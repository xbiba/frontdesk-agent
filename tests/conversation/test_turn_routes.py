"""Routes of a turn that test_turn.py leaves out, and what a turn hands to the models.

The first sections cover the limits on every model call, what the validation receives and which
clarifying questions a profile's responder may ask; the rest follows the other routes.
"""

import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai import Agent, ModelResponse, ToolCallPart, UsageLimits
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classification import (
    ClassifierOutput,
    Rejection,
    ValidatedFields,
    validate_classification,
)
from frontdesk.agent.classifier import CLASSIFIER_LIMITS
from frontdesk.agent.prompts import load_prompt
from frontdesk.agent.responder import RESPONDER_LIMITS
from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.conversation.preload import fact_evidence
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import RESPONDER_CLARIFY_FIELDS, TurnOutcome, handle_message
from frontdesk.db.engine import make_engine, make_session_factory
from frontdesk.knowledge.evidence import Evidence
from frontdesk.tenancy.pack import BusinessPack, Features, FixedTextKey, Profile
from tests.conversation.helpers import (
    classifier_model,
    classifier_turns,
    document_responder,
    make_runtime,
)

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Is there parking?",
}
RETURNS = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "How do I return an item?",
}


def _text(pack: BusinessPack, language: str, key: FixedTextKey) -> str:
    return pack.fixed_texts[language][key]  # type: ignore[index]


def _handoff(pack: BusinessPack, language: str, *keys: FixedTextKey) -> str:
    return "\n\n".join(_text(pack, language, key) for key in (*keys, FixedTextKey.HANDOFF))


def _answering(args: dict[str, Any]) -> FunctionModel:
    """A model that answers every request with these tool-call arguments."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(model)


# Every model call carries its usage limits.


async def test_every_model_call_carries_its_usage_limits(
    activate_pack: Activate, clinic_pack: BusinessPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str | None, UsageLimits | None] = {}
    original = Agent.run

    async def spy(self: Agent[Any, Any], *args: Any, **kwargs: Any) -> Any:
        seen[self.name] = kwargs.get("usage_limits")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "run", spy)
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), document_responder()
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert seen == {
        "receptionist-classifier": CLASSIFIER_LIMITS,
        "receptionist-responder": RESPONDER_LIMITS,
    }


# The validation gets the pack's profile and features and the detected language.


async def test_validation_gets_the_packs_profile_and_features_and_the_detected_language(
    activate_pack: Activate, clinic_pack: BusinessPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real = validate_classification

    def spy(output: ClassifierOutput, **kwargs: Any) -> ValidatedFields | Rejection:
        calls.append(kwargs)
        return real(output, **kwargs)

    monkeypatch.setattr("frontdesk.conversation.turn.validate_classification", spy)
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    # Italian words, while the conversation so far was English and the pack defaults to Albanian.
    state = ConversationState(turns=1, language="en")
    result, _ = await handle_message("Quanto costa la fisioterapia?", state, runtime)
    assert result.language == "it"
    (call,) = calls
    assert call["profile"] is clinic_pack.profile
    assert call["features"] is clinic_pack.features
    assert call["language"] == "it"
    assert call["message"] == "Quanto costa la fisioterapia?"


async def test_the_shop_refuses_an_intent_only_the_clinic_offers(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "price"}
    runtime = make_runtime(
        shop_pack, activate_pack(shop_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Quanto costa la spedizione?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rejection == "intent_not_allowed"
    assert result.text == _text(shop_pack, "it", FixedTextKey.HANDOFF)


async def test_orientation_can_be_switched_off_in_the_pack(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    pack = clinic_pack.model_copy(update={"features": Features(orientation=False)})
    args = {"category": "orientation", "intent": "service_info", "problem_category_id": "back"}
    runtime = make_runtime(pack, factory, classifier_model(args), document_responder())
    result, _ = await handle_message(
        "Ho mal di schiena, da chi devo andare?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rejection == "orientation_disabled"
    assert result.text == _text(pack, "it", FixedTextKey.HANDOFF)


async def test_order_status_can_be_switched_off_in_the_pack(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    pack = shop_pack.model_copy(update={"features": Features(order_status=False)})
    args = {"category": "administrative", "intent": "order_status"}
    runtime = make_runtime(pack, factory, classifier_model(args), document_responder())
    result, _ = await handle_message("Dov'è il mio ordine?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rejection == "order_status_disabled"


# A responder's clarifying question is accepted only if its profile's prompt offers it.


@pytest.mark.parametrize(
    ("field", "question"),
    [
        pytest.param("service", "Which service? For example: ", id="service"),
        pytest.param("doctor", "Which doctor? For example: Dr. Arben Demo", id="doctor"),
    ],
)
async def test_the_clinic_responder_may_ask_for_a_service_or_a_doctor(
    activate_pack: Activate, clinic_pack: BusinessPack, field: str, question: str
) -> None:
    responder = _answering({"action": "clarify", "clarify": field})
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), responder
    )
    result, state = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.ASKED
    assert result.text.startswith(question)
    assert result.trace.clarify is ClarifyField(field)
    assert state.pending_clarification is ClarifyField(field)


async def test_the_clinic_responder_may_not_ask_for_order_details(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    responder = _answering({"action": "clarify", "clarify": "order_details"})
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), responder
    )
    result, state = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "en", FixedTextKey.NO_INFORMATION)
    assert result.trace.check_reasons == ("clarify_not_offered",)
    assert result.trace.clarify is None
    assert state.pending_clarification is None


@pytest.mark.parametrize("field", [field.value for field in ClarifyField])
async def test_the_shop_responder_may_not_ask_for_anything(
    activate_pack: Activate, shop_pack: BusinessPack, field: str
) -> None:
    responder = _answering({"action": "clarify", "clarify": field})
    runtime = make_runtime(
        shop_pack, activate_pack(shop_pack), classifier_model(RETURNS), responder
    )
    result, state = await handle_message(
        "How do I return an item?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(shop_pack, "en", FixedTextKey.NO_INFORMATION)
    assert result.trace.check_reasons == ("clarify_not_offered",)
    assert result.trace.clarify is None
    assert state.pending_clarification is None


def _prompt_clarify_fields(profile: Profile) -> frozenset[ClarifyField]:
    """The clarify values the profile's responder prompt tells the model it may use."""
    prompt = load_prompt(profile, "responder")
    actions = prompt.split("Return one of", 1)[1].split("Rules:", 1)[0]
    line = next((x for x in actions.splitlines() if x.startswith("- clarify:")), "")
    return frozenset(ClarifyField(value) for value in re.findall(r'"([a-z_]+)"', line))


@pytest.mark.parametrize("profile", list(Profile))
def test_the_allowed_questions_are_the_ones_the_prompt_offers(profile: Profile) -> None:
    assert RESPONDER_CLARIFY_FIELDS[profile] == _prompt_clarify_fields(profile)


# A clarifying question counts as asked only if the person was shown it.


async def test_a_dropped_question_about_a_fact_is_not_remembered_as_asked(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "clinical", "intent": "price"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, state = await handle_message(
        "Ho la febbre, quanto costa?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.CLINICAL)
    assert result.trace.clarify is None
    assert state.pending_clarification is None


async def test_a_dropped_question_from_the_responder_is_not_remembered_as_asked(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {**PARKING, "category": "clinical"}
    responder = _answering({"action": "clarify", "clarify": "service"})
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), responder
    )
    result, state = await handle_message(
        "Ho la febbre, c'è parcheggio?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.CLINICAL)
    assert result.trace.clarify is None
    assert state.pending_clarification is None


async def test_a_question_that_was_asked_reaches_the_next_classification(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    instructions: list[str] = []
    classifier = classifier_turns(
        [{"category": "administrative", "intent": "price"}, PRICE], instructions
    )
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier, document_responder()
    )
    first, state = await handle_message("Quanto costa?", ConversationState(turns=1), runtime)
    second, after = await handle_message("La fisioterapia", state, runtime)
    assert first.outcome is TurnOutcome.ASKED
    assert "The assistant has just asked the person for" not in instructions[0]
    assert "The assistant has just asked the person for: service." in instructions[1]
    assert second.text.startswith("Fisioterapia: 30 EUR (a seduta)")
    assert (after.pending_clarification, after.previous_intents) == (None, ("price", "price"))


# Routes the rules and the classifier can take besides the ones test_turn.py follows.


async def test_without_active_knowledge_no_model_runs_and_the_person_goes_to_reception(
    app_url: str, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    factory = make_session_factory(make_engine(app_url))  # migrated, but nothing activated
    runtime = make_runtime(
        clinic_pack, factory, classifier_model(None, prompts), document_responder()
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _text(clinic_pack, "en", FixedTextKey.HANDOFF)
    assert result.trace.rejection == "no_active_knowledge"
    assert prompts == []


async def test_an_emergency_found_by_the_rules_clears_the_topic(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(None), document_responder()
    )
    state = ConversationState(turns=1, topic_service_ids=("physiotherapy",), topic_age=1)
    result, after = await handle_message("mi fa male il petto", state, runtime)
    assert result.trace.rule == "emergency"
    assert (after.topic_service_ids, after.topic_doctor_ids, after.topic_age) == ((), (), 0)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"category": "emergency"}, id="no-intent"),
        pytest.param(
            {"category": " Emergency ", "intent": "nonsense", "unexpected": True}, id="off-contract"
        ),
    ],
)
async def test_a_classifier_answer_that_says_emergency_gets_the_emergency_text_even_if_malformed(
    activate_pack: Activate, clinic_pack: BusinessPack, args: dict[str, Any]
) -> None:
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder(prompts)
    )
    result, _ = await handle_message(
        "Mio padre ha avuto un malore", ConversationState(turns=1), runtime
    )
    assert result.trace.rule == "pass"  # the rules let it through; the classifier caught it
    assert result.trace.category is Category.EMERGENCY
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.EMERGENCY)
    assert prompts == []


async def test_a_phone_number_gets_the_personal_data_text_before_any_model_runs(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(None, prompts),
        document_responder(),
    )
    result, _ = await handle_message(
        "Il mio numero è +355 00 111 2222", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.FIXED
    assert result.text == _text(clinic_pack, "it", FixedTextKey.PERSONAL_DATA)
    assert result.trace.rule == "personal_data:phone"
    assert prompts == []


async def test_a_request_for_medical_advice_gets_the_clinical_text_before_any_model_runs(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(None, prompts),
        document_responder(),
    )
    result, _ = await handle_message(
        "Che farmaco devo prendere per la schiena?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.CLINICAL)
    assert result.trace.rule == "clinical_advice"
    assert prompts == []


@pytest.mark.parametrize("intent", ["greeting", "other"])
async def test_small_talk_gets_the_greeting_text(
    activate_pack: Activate, clinic_pack: BusinessPack, intent: str
) -> None:
    args = {"category": "administrative", "intent": intent}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message("Buongiorno", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.FIXED
    assert result.text == _text(clinic_pack, "it", FixedTextKey.GREETING)


async def test_a_request_for_a_person_hands_off(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "human"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Vorrei parlare con un operatore", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _text(clinic_pack, "it", FixedTextKey.HANDOFF)


@pytest.mark.parametrize(
    ("intent", "lines"),
    [
        pytest.param(
            "location", "Indirizzo: Rruga e Shembullit 1, Tiranë (adresë fiktive)", id="address"
        ),
        pytest.param("hours", "Orari: lun-ven 08:00-19:00; sab 09:00-14:00", id="hours"),
        pytest.param(
            "contact", "Telefono: +355 00 000 0000\nEmail: info@klinika-demo.example", id="contacts"
        ),
    ],
)
async def test_address_hours_and_contacts_are_written_from_the_location_record(
    activate_pack: Activate, clinic_pack: BusinessPack, intent: str, lines: str
) -> None:
    prompts: list[str] = []
    args = {"category": "administrative", "intent": intent}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder(prompts)
    )
    result, _ = await handle_message("Dove siete?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == lines
    assert result.trace.sources == ("location:main",)
    assert prompts == []


async def test_a_doctor_is_written_from_the_record_and_becomes_the_topic(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "doctor_info", "doctor_ids": ["dr-arben-demo"]}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, state = await handle_message(
        "Cosa fa il dottor Arben?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == "Dr. Arben Demo: Visita ortopedica"
    assert result.trace.sources == ("doctor:dr-arben-demo",)
    assert (state.topic_doctor_ids, state.topic_service_ids) == (("dr-arben-demo",), ())


@pytest.mark.parametrize(
    ("intent", "field", "question"),
    [
        pytest.param(
            "service_info", ClarifyField.SERVICE, "Per quale servizio? Per esempio: ", id="service"
        ),
        pytest.param(
            "doctor_info",
            ClarifyField.DOCTOR,
            "Di quale medico? Per esempio: Dr. Arben Demo, Dr.ssa Giulia Esempio, Elira Shembull.",
            id="doctor",
        ),
    ],
)
async def test_service_and_doctor_questions_without_a_name_get_a_question_from_code(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    intent: str,
    field: ClarifyField,
    question: str,
) -> None:
    args = {"category": "administrative", "intent": intent}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, state = await handle_message("Mi dica di più", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.ASKED
    assert result.text.startswith(question)
    assert state.pending_clarification is field


async def test_a_price_that_is_not_in_the_records_is_reported_as_missing(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "price", "service_ids": ["ultrasound"]}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Quanto costa l'ecografia?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.NO_INFORMATION)
    assert result.trace.sources == ()


async def test_a_responder_without_an_answer_hands_off(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        _answering({"action": "handoff"}),
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "en", FixedTextKey.NO_INFORMATION)
    assert result.trace.check_reasons == ()


async def test_the_shop_writes_its_hours_from_the_record(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "hours"}
    runtime = make_runtime(
        shop_pack, activate_pack(shop_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message("Quando siete aperti?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == "Orari: mar-sab 10:00-19:30"


async def test_the_shop_answers_a_returns_question_from_its_documents(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        shop_pack, activate_pack(shop_pack), classifier_model(RETURNS), document_responder()
    )
    result, _ = await handle_message(
        "How do I return an item?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text.startswith(
        "You can return an item within fourteen days of delivery, unused and with its label."
    )
    assert result.trace.sources == ("returns-en",)


def _classifier_args(category: Category, intent: Intent) -> dict[str, Any]:
    """A well-formed classifier answer for any pair of the closed vocabularies."""
    args: dict[str, Any] = {"category": category.value, "intent": intent.value}
    if intent in {Intent.PRICE, Intent.SERVICE_INFO, Intent.BOOKING}:
        args["service_ids"] = ["physiotherapy"]
    if intent is Intent.DOCTOR_INFO:
        args["doctor_ids"] = ["dr-arben-demo"]
    if intent is Intent.GENERAL_QUESTION:
        args["general_question"] = "Is there parking?"
    if category is Category.ORIENTATION:
        args["problem_category_id"] = "back"
    return args


async def test_only_general_questions_reach_the_responder_and_every_handoff_ends_with_contact(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    args: dict[str, Any] = {}  # the classifier reads it at each call
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder(prompts)
    )
    contact = _text(clinic_pack, "en", FixedTextKey.HANDOFF)
    reached: set[tuple[Category, Intent]] = set()
    for category in Category:
        for intent in Intent:
            args.clear()
            args.update(_classifier_args(category, intent))
            asked = len(prompts)
            state = ConversationState(turns=1)
            result, _ = await handle_message("Is there parking?", state, runtime)
            if len(prompts) > asked:
                reached.add((category, intent))
            if result.outcome is TurnOutcome.HANDOFF:
                assert result.text.endswith(contact), (category, intent)
    # Orientation and emergencies never reach it; the rest do only with a general question.
    assert reached == {
        (Category.ADMINISTRATIVE, Intent.GENERAL_QUESTION),
        (Category.CLINICAL, Intent.GENERAL_QUESTION),
        (Category.UNCERTAIN, Intent.GENERAL_QUESTION),
    }


async def test_a_price_question_says_which_service_has_no_price(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {
        "category": "administrative",
        "intent": "price",
        "service_ids": ["physiotherapy", "osteopathy"],  # osteopathy has no price record
    }
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Quanto costano fisioterapia e osteopatia?", ConversationState(turns=1), runtime
    )
    no_information = _text(clinic_pack, "it", FixedTextKey.NO_INFORMATION)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == (
        f"Fisioterapia: 30 EUR (a seduta)\nOsteopatia: {no_information}\n\n"
        f"{_text(clinic_pack, 'it', FixedTextKey.HANDOFF)}"
    )
    assert result.trace.sources == ("price:physiotherapy",)


# A pack without a next step adds no closing: the generic question read as a bot.
async def test_a_price_question_with_every_record_ends_with_the_records(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {
        "category": "administrative",
        "intent": "price",
        "service_ids": ["physiotherapy", "orthopedics-visit"],
    }
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Quanto costano fisioterapia e visita ortopedica?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == "Fisioterapia: 30 EUR (a seduta)\nVisita ortopedica: 40 EUR"


@pytest.mark.parametrize(
    ("message", "args"),
    [
        pytest.param("mi fa male il petto", None, id="rule"),
        pytest.param(
            "mio padre ha avuto un malore",
            {"category": "emergency", "intent": "other"},
            id="classifier",
        ),
        pytest.param(
            "Vorrei sapere gli orari. " * 100 + "Aiuto, mi fa male il petto", None, id="long"
        ),
    ],
)
async def test_the_emergency_text_comes_first_on_the_first_turn(
    activate_pack: Activate, clinic_pack: BusinessPack, message: str, args: dict[str, Any] | None
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(message, ConversationState(), runtime)  # turns == 0
    emergency = _text(clinic_pack, "it", FixedTextKey.EMERGENCY)
    disclosure = _text(clinic_pack, "it", FixedTextKey.DISCLOSURE)
    handoff = _text(clinic_pack, "it", FixedTextKey.HANDOFF)
    assert result.text == f"{emergency}\n\n{disclosure}\n\n{handoff}"


@pytest.mark.parametrize("intent", ["booking", "human"])
async def test_a_clinical_message_that_asks_for_a_person_keeps_the_clinical_text(
    activate_pack: Activate, clinic_pack: BusinessPack, intent: str
) -> None:
    args = {"category": "clinical", "intent": intent}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Ho la febbre da tre giorni, vorrei un operatore", ConversationState(turns=1), runtime
    )
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.CLINICAL)


async def test_a_doctor_question_says_which_doctor_has_no_record(
    activate_pack: Activate, clinic_pack: BusinessPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = fact_evidence

    def without_elira(*args: Any, **kwargs: Any) -> list[Evidence]:
        # As if the knowledge changed between reading the catalog and reading the records.
        return [
            item for item in real(*args, **kwargs) if item.source_ref != "doctor:ft-elira-shembull"
        ]

    monkeypatch.setattr("frontdesk.conversation.turn.fact_evidence", without_elira)
    args = {
        "category": "administrative",
        "intent": "doctor_info",
        "doctor_ids": ["dr-arben-demo", "ft-elira-shembull"],
    }
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Cosa fanno il dottor Arben ed Elira?", ConversationState(turns=1), runtime
    )
    no_information = _text(clinic_pack, "it", FixedTextKey.NO_INFORMATION)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == (
        f"Dr. Arben Demo: Visita ortopedica\nElira Shembull: {no_information}\n\n"
        f"{_text(clinic_pack, 'it', FixedTextKey.HANDOFF)}"
    )
    assert result.trace.sources == ("doctor:dr-arben-demo",)


async def test_an_orientation_answer_is_complete_even_if_a_service_is_named_too(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {
        "category": "orientation",
        "intent": "service_info",
        "problem_category_id": "back",
        "service_ids": ["physiotherapy"],
    }
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Ho mal di schiena, da chi devo andare?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == (
        "Se ne occupano: Visita ortopedica, Osteopatia.\n\n"
        "Vuoi sapere prezzi o orari, o preferisci parlare con la reception?"
    )


async def test_choosing_the_reception_after_an_orientation_gives_the_contact_and_promises_nothing(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    orientation = {
        "category": "orientation",
        "intent": "service_info",
        "problem_category_id": "back",
    }
    human = {"category": "administrative", "intent": "human"}
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_turns([orientation, human], []),
        document_responder(),
    )
    first, state = await handle_message(
        "Ho mal di schiena, da chi devo andare?", ConversationState(turns=1), runtime
    )
    assert "la reception?" in first.text  # the offer
    second, _ = await handle_message("Preferisco la reception", state, runtime)
    assert second.outcome is TurnOutcome.HANDOFF
    assert second.trace.intent is Intent.HUMAN
    assert second.text == _text(clinic_pack, "it", FixedTextKey.HANDOFF)
    assert "+355 00 000 0000" in second.text  # how to reach a person; nobody is said to call
