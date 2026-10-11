"""Classifier output contract and its validation (ADR-0006: the second level, after the rules).

The model may answer only with closed vocabularies and allowlisted IDs. Code
checks everything; anything off-contract is rejected and handed to a person.
The validated fields are all that later steps — retrieval, the responder,
the order lookup — ever see of the message. Identifiers such as an order
number are read from the message by code, never taken from the model.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frontdesk.agent.vocab import PROFILE_INTENTS, Category, Intent
from frontdesk.knowledge.facts import Catalog
from frontdesk.safety.normalize import normalize
from frontdesk.safety.rules import Verdict, contains_clinical_terms, evaluate
from frontdesk.tenancy.pack import Features, Language, Profile

MAX_QUESTION_CHARS: Final = 120
# ASCII digits only: `\d` would also accept other scripts' digits.
ORDER_REFERENCE: Final = re.compile(r"^ORD-[0-9]{4,8}$")
POSTAL_CODE: Final = re.compile(r"^[0-9]{4,6}$")
_DIGIT_RUN: Final = re.compile(r"\d{4,}")
_REFERENCE_IN_TEXT: Final = re.compile(r"\bORD[-\s]?([0-9]{4,8})\b", re.IGNORECASE)
_NUMBER_IN_TEXT: Final = re.compile(r"(?<![\w-])([0-9]{4,6})(?![\w-])")
# Dates with dots or slashes (10.10.2026, 2026/10/10) are blanked before the number search: their
# year is not a postal code. Hyphenated dates are kept out by `_NUMBER_IN_TEXT` itself.
# Words that ask *when* (sq, it, en), read in the normalized message. A doctor question that asks when
# is about dates, which the documents hold, not about the doctor's record (live MCU run, 2026-10-11).
_ASKS_WHEN: Final = re.compile(
    r"\b(?:kur|deri kur|data|datat|dite|ditet|quando|fino a|date|giorni|giorno|when|until|dates|days)\b"
)
_DATE_IN_TEXT: Final = re.compile(r"(?<![\w./-])[0-9]{1,4}[./][0-9]{1,2}[./][0-9]{1,4}(?![\w/-])")


class ClassifierOutput(BaseModel):
    """What the classifier model may return."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    intent: Intent
    service_ids: list[str] = Field(default_factory=list, max_length=5)
    doctor_ids: list[str] = Field(default_factory=list, max_length=5)
    problem_category_id: str | None = None
    general_question: str | None = Field(default=None, max_length=400)

    @model_validator(mode="before")
    @classmethod
    def _emergency_wins(cls, data: object) -> object:
        """A flagged emergency stays one whatever else is wrong with the payload (ADR-0006)."""
        if (
            isinstance(data, dict)
            and str(data.get("category", "")).strip().casefold() == Category.EMERGENCY
        ):
            return {"category": Category.EMERGENCY, "intent": Intent.OTHER}
        return data


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
    if _has_markup(stripped) or not any(char.isalnum() for char in stripped):
        return "text_has_markup"
    if "@" in stripped or _DIGIT_RUN.search(stripped):
        return "text_has_identifiers"
    if evaluate(stripped).verdict is not Verdict.PASS:
        return "text_failed_rules"
    if contains_clinical_terms(stripped):
        return "text_has_clinical_terms"
    return None


def _has_markup(text: str) -> bool:
    """Angle brackets, control characters (newlines included), invisible format characters and the
    Unicode line and paragraph separators (U+2028, U+2029), which also end a line."""
    return any(
        char in "<>" or unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in text
    )


def extract_order_details(text: str) -> tuple[str | None, str | None]:
    """Order number and postal code as written by the person; None when absent or ambiguous."""
    references = {match.group(1) for match in _REFERENCE_IN_TEXT.finditer(text)}
    dateless = _DATE_IN_TEXT.sub(" ", text)
    numbers = {match.group(1) for match in _NUMBER_IN_TEXT.finditer(dateless)} - references
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
    output = _repaired(output, message=message, catalog=catalog, features=features)
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


def _repaired(
    output: ClassifierOutput, *, message: str, catalog: Catalog, features: Features
) -> ClassifierOutput:
    """What the model was seen to leave out or write wrong, repaired where that is safe (the first
    live run, 2026-10-11): a service written by its name instead of its ID; a health problem that
    fits no category of the map, which gets the clinical notice; a general question left without
    its rewrite, for which the person's own words stand in, held to the same filter, unless they
    describe symptoms, which get the clinical notice. A business without services has no service
    for the model to name."""
    update: dict[str, object] = {}
    named = _service_ids_by_name(output.service_ids, catalog) if catalog.services else ()
    if named != tuple(output.service_ids):
        update["service_ids"] = list(named)
    if (
        output.category is Category.ORIENTATION
        and features.orientation
        and output.problem_category_id not in catalog.problem_categories
    ):
        update |= {
            "category": Category.CLINICAL,
            "intent": Intent.OTHER,
            "problem_category_id": None,
        }
    if output.intent is Intent.DOCTOR_INFO and _ASKS_WHEN.search(normalize(message)):
        update |= {"intent": Intent.GENERAL_QUESTION, "general_question": None}
        output = output.model_copy(update=update)
        update = {}
    if output.intent is Intent.GENERAL_QUESTION and not output.general_question and message.strip():
        if contains_clinical_terms(message):
            update |= {"category": Category.CLINICAL, "intent": Intent.OTHER}
        else:
            update["general_question"] = message.strip()
    return output.model_copy(update=update) if update else output


def _service_ids_by_name(written: list[str], catalog: Catalog) -> tuple[str, ...]:
    """Each service as its ID: an ID stays, a name of a service in any language becomes its ID,
    and anything else stays as written, for the contract check to refuse."""
    by_name = {
        normalize(name): service_id
        for service_id, names in catalog.services.items()
        for name in names.values()
    }
    return tuple(
        written_id
        if written_id in catalog.services
        else by_name.get(normalize(written_id), written_id)
        for written_id in written
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
