"""Replies about records, written by code (ADR-0009).

Every word comes from an approved record or from the fixed phrasing below, so
a price, an opening time or a list of services cannot be misstated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.knowledge.evidence import Evidence
from frontdesk.knowledge.facts import Catalog, field_label
from frontdesk.tenancy.pack import Language

FACT_INTENTS: Final = frozenset(
    {
        Intent.LOCATION,
        Intent.HOURS,
        Intent.CONTACT,
        Intent.PRICE,
        Intent.SERVICE_INFO,
        Intent.DOCTOR_INFO,
        Intent.ORDER_STATUS,
    }
)
_LOCATION_FIELDS: Final[dict[Intent, tuple[str, ...]]] = {
    Intent.LOCATION: ("address",),
    Intent.HOURS: ("hours",),
    Intent.CONTACT: ("phone", "email"),
}
_SOURCE_PREFIX: Final[dict[Intent, str]] = {
    Intent.PRICE: "price:",
    Intent.SERVICE_INFO: "service:",
    Intent.DOCTOR_INFO: "doctor:",
    Intent.ORDER_STATUS: "order:",
}
_ORIENTATION: Final[dict[str, str]] = {
    "sq": "Me këtë merren: {services}.",
    "it": "Se ne occupano: {services}.",
    "en": "These services can help: {services}.",
}
# An orientation reply always offers a person as well, because the health boundary relies on it.
# The offer is a question and promises nothing; answering "reception" is the `human` intent, which
# gets the contact. The Albanian wording still needs a native speaker's review.
_CLOSINGS: Final[dict[str, dict[str, str]]] = {
    "default": {
        "sq": "Mund t'ju ndihmoj me diçka tjetër?",
        # Neither "tu" nor "Lei": the closing fits every pack's register (Clinica MCU uses Lei).
        "it": "C'è altro in cui posso essere utile?",
        "en": "Can I help with anything else?",
    },
    "orientation": {
        "sq": "Doni të dini çmimet ose oraret, apo preferoni të flisni me recepsionin?",
        "it": "Vuoi sapere prezzi o orari, o preferisci parlare con la reception?",
        "en": "Would you like prices or opening hours, or would you rather talk to reception?",
    },
}
_CLARIFY: Final[dict[ClarifyField, dict[str, str]]] = {
    ClarifyField.SERVICE: {
        "sq": "Për cilin shërbim?",
        "it": "Per quale servizio?",
        "en": "Which service?",
    },
    ClarifyField.DOCTOR: {
        "sq": "Për cilin mjek?",
        "it": "Di quale medico?",
        "en": "Which doctor?",
    },
    ClarifyField.ORDER_DETAILS: {
        "sq": "Mund të më shkruani numrin e porosisë (ORD-…) dhe kodin postar në të njëjtin mesazh?",
        "it": "Mi scrivi il numero d'ordine (ORD-…) e il CAP nello stesso messaggio?",
        "en": "Could you send the order number (ORD-…) and the postal code in one message?",
    },
}
_EXAMPLES: Final[dict[str, str]] = {"sq": "Për shembull", "it": "Per esempio", "en": "For example"}
MAX_EXAMPLES: Final = 5


def closing(language: Language) -> str:
    return _CLOSINGS["default"][language]


def compose_facts(
    fields: ValidatedFields,
    facts: Sequence[Evidence],
    *,
    with_closing: bool = True,
    missing: Sequence[str] = (),
) -> str | None:
    lines = _fact_lines(fields, facts)
    if not lines:
        return None
    lines = [*lines, *missing]
    if not with_closing:
        return "\n".join(lines)
    kind = "orientation" if fields.category is Category.ORIENTATION else "default"
    return "\n".join(lines) + "\n\n" + _CLOSINGS[kind][fields.language]


def clarify_question(field: ClarifyField, language: Language, catalog: Catalog) -> str:
    question = _CLARIFY[field][language]
    if field is ClarifyField.SERVICE:
        names = [names.get(language, key) for key, names in sorted(catalog.services.items())]
    elif field is ClarifyField.DOCTOR:
        names = sorted(catalog.doctors.values())
    else:
        names = []
    if not names:
        return question
    return f"{question} {_EXAMPLES[language]}: {', '.join(names[:MAX_EXAMPLES])}."


def _fact_lines(fields: ValidatedFields, facts: Sequence[Evidence]) -> list[str]:
    language = fields.language
    if fields.category is Category.ORIENTATION:
        return [
            _ORIENTATION[language].format(services=item.values["services"])
            for item in _with_prefix(facts, "orientation:")
        ]
    if fields.intent in _LOCATION_FIELDS:
        location = next(iter(_with_prefix(facts, "location:")), None)
        if location is None:
            return []
        return [
            f"{field_label(key, language)}: {location.values[key]}"
            for key in _LOCATION_FIELDS[fields.intent]
        ]
    prefix = _SOURCE_PREFIX.get(fields.intent)
    return [item.content for item in _with_prefix(facts, prefix)] if prefix else []


def _with_prefix(facts: Sequence[Evidence], prefix: str) -> list[Evidence]:
    return [item for item in facts if item.source_ref.startswith(prefix)]


def unanswered(fields: ValidatedFields, facts: Sequence[Evidence]) -> tuple[str, ...]:
    """The services or doctors the person named that no record answers (a service with no price)."""
    prefix = _SOURCE_PREFIX.get(fields.intent)
    if prefix is None or fields.category is Category.ORIENTATION:
        return ()
    asked = fields.doctor_ids if fields.intent is Intent.DOCTOR_INFO else fields.service_ids
    answered = {item.source_ref for item in facts}
    return tuple(item for item in asked if f"{prefix}{item}" not in answered)
