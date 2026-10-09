import uuid
from collections.abc import Sequence
from datetime import date

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.answer_check import check_answer
from frontdesk.agent.classification import ClassifierOutput, ValidatedFields
from frontdesk.agent.classifier import CLASSIFIER_LIMITS, build_classifier, classifier_deps
from frontdesk.agent.demo_models import (
    classify_message,
    demo_classifier_model,
    demo_responder_model,
)
from frontdesk.agent.deps import ResponderDeps
from frontdesk.agent.responder import (
    RESPONDER_LIMITS,
    AnswerAction,
    ResponderOutput,
    build_responder,
    render_turn_prompt,
)
from frontdesk.agent.tools import ToolServices, build_toolset
from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import BusinessPack, Profile

CATALOG = Catalog(
    business_id="demo-clinic",
    version_id=uuid.uuid4(),
    services={
        "physiotherapy": {"sq": "Fizioterapi", "it": "Fisioterapia", "en": "Physiotherapy"},
        "orthopedics-visit": {
            "sq": "Vizitë ortopedike",
            "it": "Visita ortopedica",
            "en": "Orthopedic visit",
        },
        "dermatology-visit": {
            "sq": "Vizitë dermatologjike",
            "it": "Visita dermatologica",
            "en": "Dermatology visit",
        },
    },
    doctors={},
    problem_categories={"back": {"sq": ["shpinë"], "it": ["schiena"], "en": ["back"]}},
)
PARKING = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="parking-en#1",
    version="v",
    title="Parking",
    content="Klinika Demo has six free parking spaces behind the building. Street parking is paid on weekdays.",
)


async def _classify(
    text: str,
    profile: Profile = Profile.RECEPTIONIST,
    previous: tuple[Intent, ...] = (),
    topic: tuple[str, ...] = (),
    pending: ClarifyField | None = None,
) -> ClassifierOutput:
    agent = build_classifier(demo_classifier_model(CATALOG), profile)
    deps = classifier_deps(
        CATALOG,
        business_name="Demo",
        language="it",
        previous_intents=previous,
        topic_service_ids=topic,
        pending_clarification=pending,
    )
    return (await agent.run(text, deps=deps, usage_limits=CLASSIFIER_LIMITS)).output


async def test_price_question() -> None:
    output = await _classify("Quanto costa la fisioterapia?")
    assert (output.category, output.intent, output.service_ids) == (
        Category.ADMINISTRATIVE,
        Intent.PRICE,
        ["physiotherapy"],
    )


def test_shared_words_do_not_pick_a_service() -> None:
    assert classify_message("Quanto costa una visita?", CATALOG)["service_ids"] == []
    assert classify_message("Quanto costa una visita ortopedica?", CATALOG)["service_ids"] == [
        "orthopedics-visit"
    ]


async def test_orientation_and_clinical_messages() -> None:
    orientation = await _classify("Ho mal di schiena, da chi devo andare?")
    assert (orientation.category, orientation.problem_category_id) == (Category.ORIENTATION, "back")
    clinical = await _classify("Ho la febbre alta")
    assert (clinical.category, clinical.intent) == (Category.CLINICAL, Intent.OTHER)
    mixed = await _classify("Ho la febbre, quanto costa la fisioterapia?")
    assert (mixed.category, mixed.intent) == (Category.CLINICAL, Intent.PRICE)


async def test_follow_ups_use_previous_intents_and_the_topic() -> None:
    hours = await _classify("E il sabato?", previous=(Intent.HOURS,))
    assert hours.intent is Intent.HOURS
    price = await _classify(
        "E quanto costa?", previous=(Intent.SERVICE_INFO,), topic=("physiotherapy",)
    )
    assert (price.intent, price.service_ids) == (Intent.PRICE, ["physiotherapy"])


async def test_an_answer_to_a_pending_question_keeps_the_intent() -> None:
    output = await _classify("fisioterapia", previous=(Intent.PRICE,), pending=ClarifyField.SERVICE)
    assert (output.intent, output.service_ids) == (Intent.PRICE, ["physiotherapy"])


async def test_small_talk_human_unknown_and_emergency_hints() -> None:
    assert (await _classify("ok grazie")).intent is Intent.GREETING
    assert (await _classify("Vorrei parlare con un operatore")).intent is Intent.HUMAN
    assert (await _classify("boh")).category is Category.UNCERTAIN
    assert (await _classify("Mio padre ha avuto un malore")).category is Category.EMERGENCY


async def test_order_status_leaves_the_details_to_code() -> None:
    order = await _classify("Dov'è il mio ordine ORD-10427? CAP 20121", Profile.SUPPORT_SALES)
    assert order.intent is Intent.ORDER_STATUS
    assert "10427" not in order.model_dump_json()


async def test_demo_responder_quotes_only_relevant_passages() -> None:
    factory: sessionmaker[Session] = sessionmaker()  # never used: the demo responder calls no tool
    services = ToolServices(session_factory=factory, embedder=HashEmbedder(), today=date.today)
    toolset = build_toolset(Profile.RECEPTIONIST, services)
    agent = build_responder(demo_responder_model(), Profile.RECEPTIONIST, toolset)

    async def ask(question: str) -> ResponderOutput:
        fields = ValidatedFields(
            category=Category.ADMINISTRATIVE,
            intent=Intent.GENERAL_QUESTION,
            language="en",
            general_question=question,
        )
        deps = ResponderDeps(
            business_id="demo-clinic", business_name="Demo", language="en", fields=fields
        )
        prompt = render_turn_prompt(fields, [PARKING])
        return (await agent.run(prompt, deps=deps, usage_limits=RESPONDER_LIMITS)).output

    parking = await ask("Is there parking?")
    assert parking.action is AnswerAction.ANSWER
    assert (
        parking.claims[0].quote == "Klinika Demo has six free parking spaces behind the building."
    )
    assert (await ask("Can I bring my son?")).action is AnswerAction.HANDOFF


# --- The claim checks (ADR-0009) on what the demo responder writes --------------------------------


async def _respond(question: str, passages: Sequence[Evidence]) -> ResponderOutput:
    factory: sessionmaker[Session] = sessionmaker()  # never used: the demo responder calls no tool
    services = ToolServices(session_factory=factory, embedder=HashEmbedder(), today=date.today)
    agent = build_responder(
        demo_responder_model(), Profile.RECEPTIONIST, build_toolset(Profile.RECEPTIONIST, services)
    )
    fields = ValidatedFields(
        category=Category.ADMINISTRATIVE,
        intent=Intent.GENERAL_QUESTION,
        language="en",
        general_question=question,
    )
    deps = ResponderDeps(
        business_id="demo-clinic", business_name="Demo", language="en", fields=fields
    )
    prompt = render_turn_prompt(fields, passages)
    return (await agent.run(prompt, deps=deps, usage_limits=RESPONDER_LIMITS)).output


def _passages(pack: BusinessPack) -> list[Evidence]:
    """The passages retrieval hands to the responder: one per chunk of every document of the pack."""
    return [
        Evidence.create(
            kind=EvidenceKind.DOCUMENT,
            source_ref=f"{document.key}#{piece.ordinal}",
            version="v",
            title=piece.heading,
            content=piece.content,
        )
        for document in pack.documents
        for piece in chunk_markdown(document.title, document.body)
    ]


async def test_the_claims_for_every_demo_passage_pass_the_claim_checks(
    clinic_pack: BusinessPack, shop_pack: BusinessPack
) -> None:
    passages = [*_passages(clinic_pack), *_passages(shop_pack)]
    assert len(passages) >= 10
    for passage in passages:
        output = await _respond(passage.title, [passage])
        assert output.action is AnswerAction.ANSWER, passage.title
        check = check_answer(output, [passage], embedder=HashEmbedder(), min_similarity=0.5)
        assert check.ok, (passage.title, check.reasons)
        assert check.cited == (passage,)


@pytest.mark.parametrize(
    ("title", "content", "sentence"),
    [
        pytest.param(
            "Pagamenti",
            "Non accettiamo assegni. Accettiamo carte e contanti.",
            "Non accettiamo assegni.",
            id="a-negation-inside-the-sentence-is-quoted-with-it",
        ),
        pytest.param(
            "Orari",
            "Apriamo alle 9.30 ogni giorno. Chiudiamo alle 19.00.",
            "Apriamo alle 9.30 ogni giorno.",
            id="a-time-with-a-dot-is-not-a-sentence-end",
        ),
        pytest.param(
            "Resi",
            "Il reso è possibile entro quattordici giorni dalla consegna. Scrivici il numero.",
            "Il reso è possibile entro quattordici giorni dalla consegna.",
            id="a-number-in-words-is-the-same-number-in-claim-and-quote",
        ),
        pytest.param(
            "Posti",
            "Il parcheggio ha 1.000 posti auto riservati. Altro non serve.",
            "Il parcheggio ha 1.000 posti auto riservati.",
            id="a-thousands-separator-stays-whole",
        ),
    ],
)
async def test_the_quote_is_a_whole_sentence_whatever_it_holds(
    title: str, content: str, sentence: str
) -> None:
    passage = Evidence.create(
        kind=EvidenceKind.DOCUMENT, source_ref="doc#0", version="v", title=title, content=content
    )
    output = await _respond(title, [passage])
    assert [claim.quote for claim in output.claims] == [sentence]
    check = check_answer(output, [passage], embedder=HashEmbedder(), min_similarity=0.5)
    assert check.ok, check.reasons
    assert check.text == sentence
