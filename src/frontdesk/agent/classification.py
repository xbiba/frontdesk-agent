"""Classifier output contract and its validation (spec §6, second level).

The model may answer only with closed vocabularies and allowlisted IDs. Code
checks everything; anything off-contract is rejected and handed to a person.
The validated fields are all that later steps — retrieval, the responder,
the order lookup — ever see of the message. Identifiers such as an order
number are read from the message by code, never taken from the model.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from frontdesk.agent.vocab import PROFILE_INTENTS, Category, Intent
from frontdesk.knowledge.facts import Catalog
from frontdesk.safety.rules import Verdict, contains_clinical_terms, evaluate
from frontdesk.tenancy.pack import Features, Language, Profile

MAX_QUESTION_CHARS: Final = 120
# ASCII digits only: `\d` would also accept other scripts' digits.
ORDER_REFERENCE: Final = re.compile(r"^ORD-[0-9]{4,8}$")
POSTAL_CODE: Final = re.compile(r"^[0-9]{4,6}$")
_DIGIT_RUN: Final = re.compile(r"\d{4,}")
_REFERENCE_IN_TEXT: Final = re.compile(r"\bORD[-\s]?([0-9]{4,8})\b", re.IGNORECASE)
_NUMBER_IN_TEXT: Final = re.compile(r"(?<![\w-])([0-9]{4,6})(?![\w-])")


class ClassifierOutput(BaseModel):
    """What the classifier model may return."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    intent: Intent
    service_ids: list[str] = Field(default_factory=list, max_length=5)
    doctor_ids: list[str] = Field(default_factory=list, max_length=5)
    problem_category_id: str | None = None
    general_question: str | None = Field(default=None, max_length=400)


class ValidatedFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: Category
    intent: Intent
    language: Language
    service_ids: tuple[str, ...] = ()
    doctor_ids: tuple[str, ...] = ()
    problem_category_id: str | None = None
    order_reference: str | None = None
    postal_code: str | None = None
    general_question: str | None = None


class Rejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str


def check_free_text(text: str) -> str | None:
    """Why this text may not leave the system as a query, or None if it may."""
    stripped = text.strip()
    if not stripped:
        return "empty_text"
    if len(stripped) > MAX_QUESTION_CHARS:
        return "text_too_long"
    if "@" in stripped or _DIGIT_RUN.search(stripped):
        return "text_has_identifiers"
    if evaluate(stripped).verdict is not Verdict.PASS:
        return "text_failed_rules"
    if contains_clinical_terms(stripped):
        return "text_has_clinical_terms"
    return None


def extract_order_details(text: str) -> tuple[str | None, str | None]:
    """Order number and postal code as written by the person; None when absent or ambiguous."""
    references = {match.group(1) for match in _REFERENCE_IN_TEXT.finditer(text)}
    numbers = {match.group(1) for match in _NUMBER_IN_TEXT.finditer(text)} - references
    reference = f"ORD-{next(iter(references))}" if len(references) == 1 else None
    postal_code = next(iter(numbers)) if len(numbers) == 1 else None
    return reference, postal_code


def validate_classification(
    output: ClassifierOutput,
    *,
    message: str,
    catalog: Catalog,
    profile: Profile,
    features: Features,
    language: Language,
) -> ValidatedFields | Rejection:
    if output.category is Category.EMERGENCY:
        return ValidatedFields(category=Category.EMERGENCY, intent=Intent.OTHER, language=language)
    problem = _contract_problem(output, catalog=catalog, profile=profile, features=features)
    if problem is not None:
        return Rejection(reason=problem)
    question: str | None = None
    if output.intent is Intent.GENERAL_QUESTION:
        if not output.general_question:
            return Rejection(reason="missing_general_question")
        reason = check_free_text(output.general_question)
        if reason is not None:
            return Rejection(reason=reason)
        question = output.general_question.strip()
    orientation = output.category is Category.ORIENTATION
    reference, postal_code = (
        extract_order_details(message) if output.intent is Intent.ORDER_STATUS else (None, None)
    )
    return ValidatedFields(
        category=output.category,
        intent=output.intent,
        language=language,
        service_ids=tuple(dict.fromkeys(output.service_ids)),
        doctor_ids=tuple(dict.fromkeys(output.doctor_ids)),
        problem_category_id=output.problem_category_id if orientation else None,
        order_reference=reference,
        postal_code=postal_code,
        general_question=question,
    )


def _contract_problem(
    output: ClassifierOutput, *, catalog: Catalog, profile: Profile, features: Features
) -> str | None:
    if output.intent not in PROFILE_INTENTS[profile]:
        return "intent_not_allowed"
    if output.category is Category.ORIENTATION:
        if not features.orientation:
            return "orientation_disabled"
        if output.problem_category_id not in catalog.problem_categories:
            return "unknown_problem_category"
    if any(service_id not in catalog.services for service_id in output.service_ids):
        return "unknown_service_id"
    if any(doctor_id not in catalog.doctors for doctor_id in output.doctor_ids):
        return "unknown_doctor_id"
    if output.intent is Intent.ORDER_STATUS and not features.order_status:
        return "order_status_disabled"
    return None
