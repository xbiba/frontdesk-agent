"""Deterministic stand-ins for Claude in the keyless demo (route `offline-demo`).

Keyword rules, not language understanding: they exercise the real pipeline —
rules, validation, facts written by code, retrieval, claim checks, handoffs —
and say nothing about model quality. Real answers use a live route (ADR-0004).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Final

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from frontdesk.agent.vocab import ClarifyField, Intent
from frontdesk.knowledge.facts import Catalog
from frontdesk.knowledge.retrieval import STOPWORDS
from frontdesk.safety.normalize import tokens
from frontdesk.safety.rules import contains_clinical_terms

# Normalized keywords; keys shorter than four letters must match a whole word.
_INTENT_KEYWORDS: Final[tuple[tuple[Intent, tuple[str, ...]], ...]] = (
    (Intent.HUMAN, ("operator", "njeri", "human", "reception", "recepsion")),
    (Intent.PRICE, ("kushton", "cmim", "prezz", "cost", "price", "much", "tarif")),
    (Intent.HOURS, ("orar", "hapur", "apert", "chius", "open", "hours")),
    (Intent.LOCATION, ("adres", "ndodh", "indirizz", "dove", "dov", "where", "address")),
    (
        Intent.CONTACT,
        ("telefon", "phone", "numer", "numr", "email", "contatt", "contact", "kontakt"),
    ),
    (Intent.BOOKING, ("rezerv", "takim", "prenot", "appuntament", "appointment", "book")),
    (
        Intent.GREETING,
        ("pershendetje", "ciao", "buongiorno", "buonasera", "salve", "hello", "hi", "tung"),
    ),
)
_ACKNOWLEDGEMENTS: Final = frozenset(
    {
        "ok",
        "okay",
        "grazie",
        "mille",
        "faleminderit",
        "shume",
        "mire",
        "thanks",
        "thank",
        "you",
        "very",
        "much",
        "perfetto",
        "perfect",
    }
)
# A second net after the rules, as the real classifier is (ADR-0006).
_EMERGENCY_HINTS: Final = ("malore", "sto molto male", "ndihem shume keq", "feel very ill")
_ORDER_WORDS: Final = frozenset({"ordine", "order", "porosia", "porosine", "porosi"})
_FOLLOW_UP_INTENTS: Final = frozenset({Intent.HOURS, Intent.LOCATION, Intent.CONTACT})
_TOPIC_INTENTS: Final = frozenset({Intent.PRICE, Intent.SERVICE_INFO, Intent.BOOKING})
_INTENT_VALUES: Final = frozenset(intent.value for intent in Intent)
_ORDER: Final = re.compile(r"\bORD[-\s]?[0-9]{4,8}\b", re.IGNORECASE)
_PREVIOUS: Final = re.compile(r"Previous intents in this conversation: ([a-z_, ]+)\.")
_TOPIC_SERVICES: Final = re.compile(r"Topic from earlier turns: service_ids ([^;\n]*)")
_TOPIC_ID: Final = re.compile(r"([a-z0-9][a-z0-9-]*) \(")
_PENDING: Final = re.compile(r"The assistant has just asked the person for: (\w+)\.")
_PASSAGE: Final = re.compile(
    r'<evidence id="(?P<id>D-[0-9a-f]{8})" kind="document" title="(?P<title>[^"]*)">\n'
    r"(?P<content>.*?)\n</evidence>",
    re.DOTALL,
)
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+")


def classify_message(
    text: str,
    catalog: Catalog,
    previous: Sequence[Intent] = (),
    *,
    topic_services: Sequence[str] = (),
    pending: ClarifyField | None = None,
) -> dict[str, Any]:
    words = tokens(text)
    normalized = " ".join(words)
    if _ORDER.search(text) or _ORDER_WORDS.intersection(words):
        return {"category": "administrative", "intent": "order_status"}
    if any(hint in normalized for hint in _EMERGENCY_HINTS):
        return {"category": "emergency", "intent": "other"}
    if words and all(word in _ACKNOWLEDGEMENTS for word in words):
        return {"category": "administrative", "intent": "greeting"}
    intent = _intent(words)
    services = _matching(
        words, {key: list(names.values()) for key, names in catalog.services.items()}
    )
    problems = _matching(
        words,
        {
            key: [w for labels in by_language.values() for w in labels]
            for key, by_language in catalog.problem_categories.items()
        },
    )
    clinical = contains_clinical_terms(text)
    if problems and intent is None:
        return {
            "category": "orientation",
            "intent": "service_info",
            "problem_category_id": problems[0],
        }
    if intent is None and services and pending is ClarifyField.SERVICE and previous:
        intent = previous[-1] if previous[-1] in _TOPIC_INTENTS else Intent.SERVICE_INFO
    if intent is None and not (services or clinical) and len(words) <= 4 and previous:
        intent = previous[-1] if previous[-1] in _FOLLOW_UP_INTENTS else None
    if intent in _TOPIC_INTENTS and not services:
        services = list(topic_services)
    category = "clinical" if clinical or problems else "administrative"
    if intent is not None:
        return {"category": category, "intent": intent.value, "service_ids": services}
    if services:
        return {"category": category, "intent": "service_info", "service_ids": services}
    if clinical:
        return {"category": "clinical", "intent": "other"}
    if len(words) >= 3:
        return {
            "category": "administrative",
            "intent": "general_question",
            "general_question": normalized[:120],
        }
    return {"category": "uncertain", "intent": "other"}


def demo_classifier_model(catalog: Catalog) -> FunctionModel:
    def classify(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        instructions = (request.instructions if isinstance(request, ModelRequest) else None) or ""
        pending = _PENDING.search(instructions)
        output = classify_message(
            _first_prompt(messages),
            catalog,
            _previous_intents(instructions),
            topic_services=_topic_services(instructions),
            pending=ClarifyField(pending.group(1)) if pending else None,
        )
        return _output(info, output)

    return FunctionModel(classify)


def demo_responder_model() -> FunctionModel:
    """Quotes the first sentence of the first passage whose title or text shares a word with the question."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = _first_prompt(messages)
        question = _request_fields(prompt).get("question", "")
        for match in _PASSAGE.finditer(prompt):
            if _relevant(question, f"{match['title']} {match['content']}"):
                sentence = _SENTENCE_END.split(match["content"].strip())[0]
                claim = {"text": sentence, "evidence_id": match["id"], "quote": sentence}
                return _output(info, {"action": "answer", "claims": [claim]})
        return _output(info, {"action": "handoff"})

    return FunctionModel(respond)


def _intent(words: Sequence[str]) -> Intent | None:
    for intent, keys in _INTENT_KEYWORDS:
        if any(_keyword(word, key) for word in words for key in keys):
            return intent
    return None


def _keyword(word: str, key: str) -> bool:
    return word == key or (len(key) >= 4 and word.startswith(key))


def _matching(words: Sequence[str], labels: dict[str, list[str]]) -> list[str]:
    """Keys whose labels share a distinctive prefix with a word of the message.

    A prefix shared by two keys ("visit" in two visit services) picks neither.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for key, names in labels.items():
        for name in names:
            for token in tokens(name):
                if len(token) >= 3:
                    owners[token[:5]].add(key)
    distinctive = {prefix: next(iter(keys)) for prefix, keys in owners.items() if len(keys) == 1}
    return sorted(
        {key for word in words for prefix, key in distinctive.items() if word.startswith(prefix)}
    )


def _relevant(question: str, passage: str) -> bool:
    """At least one content word in common (by its first five letters)."""

    def stems(text: str) -> set[str]:
        return {word[:5] for word in tokens(text) if len(word) >= 4 and word not in STOPWORDS}

    return bool(stems(question) & stems(passage))


def _previous_intents(instructions: str) -> tuple[Intent, ...]:
    match = _PREVIOUS.search(instructions)
    names = [name.strip() for name in match.group(1).split(",")] if match else []
    return tuple(Intent(name) for name in names if name in _INTENT_VALUES)


def _topic_services(instructions: str) -> tuple[str, ...]:
    match = _TOPIC_SERVICES.search(instructions)
    return tuple(_TOPIC_ID.findall(match.group(1))) if match else ()


def _output(info: AgentInfo, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])


def _first_prompt(messages: Sequence[ModelMessage]) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ""


def _request_fields(prompt: str) -> dict[str, str]:
    body = prompt.split("</request>", 1)[0]
    pairs = (line.partition(": ") for line in body.splitlines()[1:])
    return {key.strip(): value.strip() for key, separator, value in pairs if separator}
