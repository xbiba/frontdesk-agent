"""Evidence gathered by code (spec §4, step 5): records for facts, passages for documents.

Queries are built from validated fields only, never from the original message.
"""

from __future__ import annotations

from datetime import date
from typing import Final

from sqlalchemy.orm import Session

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.vocab import Category, Intent
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.evidence import Evidence
from frontdesk.knowledge.facts import (
    Catalog,
    doctor_evidence,
    location_evidence,
    orientation_evidence,
    price_evidence,
    service_evidence,
)
from frontdesk.knowledge.retrieval import chunk_evidence, hybrid_search
from frontdesk.tenancy.pack import Language

_INTENT_TERMS: Final[dict[Intent, dict[str, str]]] = {
    Intent.LOCATION: {"sq": "adresa parkim", "it": "indirizzo parcheggio", "en": "address parking"},
    Intent.HOURS: {"sq": "orari", "it": "orari apertura", "en": "opening hours"},
    Intent.CONTACT: {"sq": "kontakt telefon", "it": "contatti telefono", "en": "contact phone"},
}
_LOCATION_INTENTS: Final = frozenset({Intent.LOCATION, Intent.HOURS, Intent.CONTACT})


def build_query(fields: ValidatedFields, catalog: Catalog, language: Language) -> str | None:
    parts: list[str] = []
    if fields.general_question:
        parts.append(fields.general_question)
    parts += [
        catalog.services[s].get(language, s) for s in fields.service_ids if s in catalog.services
    ]
    parts += [catalog.doctors[d] for d in fields.doctor_ids if d in catalog.doctors]
    term = _INTENT_TERMS.get(fields.intent, {}).get(language)
    if term:
        parts.append(term)
    query = " ".join(parts).strip()
    return query or None


def fact_evidence(
    session: Session, fields: ValidatedFields, *, business_id: str, today: date
) -> list[Evidence]:
    language = fields.language
    if fields.category is Category.ORIENTATION and fields.problem_category_id:
        return _present(
            orientation_evidence(session, business_id, fields.problem_category_id, language)
        )
    if fields.intent in _LOCATION_INTENTS:
        return _present(location_evidence(session, business_id, language))
    if fields.intent is Intent.PRICE:
        return [
            item
            for service_id in fields.service_ids
            for item in _present(price_evidence(session, business_id, service_id, language, today))
        ]
    if fields.intent is Intent.SERVICE_INFO:
        return service_evidence(session, business_id, fields.service_ids, language)
    if fields.intent is Intent.DOCTOR_INFO:
        return doctor_evidence(session, business_id, fields.doctor_ids, language)
    return []


def document_evidence(
    session: Session,
    fields: ValidatedFields,
    *,
    business_id: str,
    catalog: Catalog,
    embedder: Embedder,
    today: date,
) -> list[Evidence]:
    query = build_query(fields, catalog, fields.language)
    if query is None:
        return []
    vector = embedder.embed([query])[0]
    chunks = hybrid_search(
        session, business_id=business_id, query=query, query_vector=vector, today=today
    )
    return [chunk_evidence(chunk) for chunk in chunks]


def _present(item: Evidence | None) -> list[Evidence]:
    return [item] if item is not None else []
