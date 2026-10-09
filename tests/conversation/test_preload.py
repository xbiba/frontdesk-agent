"""Search queries are built from validated fields and catalog names, never from the message."""

import uuid

from sqlalchemy.orm import Session

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.vocab import Category, Intent
from frontdesk.conversation.preload import build_query, document_evidence
from frontdesk.knowledge.facts import Catalog
from tests.conversation.helpers import TODAY, SpyEmbedder

CATALOG = Catalog(
    business_id="demo-clinic",
    version_id=uuid.uuid4(),
    services={"physiotherapy": {"it": "Fisioterapia", "en": "Physiotherapy"}},
    doctors={"dr-arben-demo": "Dr. Arben Demo"},
    problem_categories={},
)


def _fields(intent: Intent, **changes: object) -> ValidatedFields:
    return ValidatedFields(
        category=Category.ADMINISTRATIVE, intent=intent, language="it", **changes
    )


def test_the_query_is_the_question_then_the_names_then_the_intent_terms() -> None:
    fields = _fields(
        Intent.GENERAL_QUESTION,
        general_question="Come mi preparo?",
        service_ids=("physiotherapy",),
        doctor_ids=("dr-arben-demo",),
    )
    assert build_query(fields, CATALOG, "it") == "Come mi preparo? Fisioterapia Dr. Arben Demo"


def test_service_names_come_in_the_language_of_the_query_and_unknown_ids_are_left_out() -> None:
    fields = _fields(Intent.PRICE, service_ids=("physiotherapy", "unknown"))
    assert build_query(fields, CATALOG, "en") == "Physiotherapy"
    # A service with no name in the language is searched by its id.
    assert build_query(fields, CATALOG, "sq") == "physiotherapy"


def test_location_intents_add_their_search_terms_in_each_language() -> None:
    assert build_query(_fields(Intent.HOURS), CATALOG, "en") == "opening hours"
    assert build_query(_fields(Intent.CONTACT), CATALOG, "it") == "contatti telefono"
    assert build_query(_fields(Intent.LOCATION), CATALOG, "sq") == "adresa parkim"


def test_nothing_to_search_for_gives_no_query_and_no_embedding() -> None:
    fields = _fields(Intent.OTHER)
    assert build_query(fields, CATALOG, "it") is None
    embedder = SpyEmbedder()
    found = document_evidence(
        Session(),  # never used: there is no query to run
        fields,
        business_id="demo-clinic",
        catalog=CATALOG,
        embedder=embedder,
        today=TODAY,
    )
    assert found == []
    assert embedder.inputs == []
