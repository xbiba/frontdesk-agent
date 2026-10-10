"""Search queries are built from validated fields and catalog names, never from the message."""

import uuid
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.vocab import Category, Intent
from frontdesk.conversation.preload import build_query, document_evidence
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import handle_message
from frontdesk.db.engine import business_session
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import BusinessPack
from tests.conversation.helpers import (
    TODAY,
    SpyEmbedder,
    classifier_model,
    document_responder,
    make_runtime,
)

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


# --- The relevance floor reaches the search behind the preload (final review I-1) ---------------------

Activate = Callable[..., sessionmaker[Session]]
UNRELATED = "zebra quantum harmonica"  # shares no word with the demo clinic's documents


@pytest.mark.db
def test_the_floor_reaches_the_search_behind_the_preload(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    fields = _fields(Intent.GENERAL_QUESTION, general_question=UNRELATED)
    with business_session(factory, clinic_pack.id) as session:
        search = partial(
            document_evidence,
            session,
            fields,
            business_id=clinic_pack.id,
            catalog=CATALOG,
            embedder=HashEmbedder(),
            today=TODAY,
        )
        assert search()  # no floor: the nearest passages, however far they are
        assert search(min_similarity=0.9) == []


@pytest.mark.db
@pytest.mark.parametrize(("floor", "shown"), [(0.0, True), (0.9, False)])
async def test_the_floor_of_the_runtime_reaches_the_passages_the_responder_is_given(
    activate_pack: Activate, clinic_pack: BusinessPack, floor: float, shown: bool
) -> None:
    question = {
        "category": "administrative",
        "intent": "general_question",
        "general_question": UNRELATED,
    }
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(question),
        document_responder(prompts),
    )
    await handle_message(
        UNRELATED, ConversationState(turns=1), replace(runtime, min_retrieval_similarity=floor)
    )
    [prompt] = prompts
    assert ("<evidence" in prompt) is shown
