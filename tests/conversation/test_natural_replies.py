"""A conversation like a good receptionist's (Albi, 2026-11-07: "una conversazione naturale ...
mostra la possibilità di chiamare una persona solo quando è richiesto"). Reception's contact comes
when the person asks for it, for an emergency or a medical question, or when a booking needs the
staff; an answer the bot does not have offers the contact without writing it; a failure says to
try again in a moment; a greeting is the introduction alone; and no answer ends with a tail."""

import asyncio
from collections.abc import Callable

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, handle_message
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "C'è parcheggio?",
}
STEP = "Per prenotare una visita, chiama la reception al +355 00 000 0000."
PHONE_DIGITS = "3550000000"


def _text(pack: BusinessPack, language: str, key: FixedTextKey) -> str:
    return pack.fixed_texts[language][key]  # type: ignore[index]


def _digits(text: str) -> str:
    return "".join(char for char in text if char.isdigit())


def _with_next_step(pack: BusinessPack) -> BusinessPack:
    texts = {
        language: {**texts, FixedTextKey.NEXT_STEP: STEP}
        for language, texts in pack.fixed_texts.items()
    }
    return pack.model_copy(update={"fixed_texts": texts})


async def test_a_first_greeting_is_the_introduction_alone(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    greeting = {"category": "administrative", "intent": "greeting"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(greeting), document_responder()
    )
    result, _ = await handle_message("Ciao", ConversationState(), runtime)
    assert result.text == _text(clinic_pack, "it", FixedTextKey.DISCLOSURE)
    assert PHONE_DIGITS not in _digits(result.text)


async def test_an_answer_ends_with_the_answer_and_no_tail(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = _with_next_step(clinic_pack)
    runtime = make_runtime(pack, activate_pack(pack), classifier_model(PRICE), document_responder())
    result, _ = await handle_message("Quanto costa?", ConversationState(turns=1), runtime)
    assert (result.outcome, result.text) == (TurnOutcome.REPLIED, "Fisioterapia: 30 EUR (a seduta)")


async def test_a_document_answer_ends_with_the_answer(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = _with_next_step(clinic_pack)
    runtime = make_runtime(
        pack, activate_pack(pack), classifier_model(PARKING), document_responder()
    )
    result, _ = await handle_message("C'è parcheggio?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert STEP not in result.text


async def test_what_the_bot_does_not_know_it_says_and_offers_the_contact_without_the_number(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    def no_passage(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"action": "handoff"})])

    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        FunctionModel(no_passage),
    )
    result, _ = await handle_message("C'è parcheggio?", ConversationState(turns=1), runtime)
    no_information = _text(clinic_pack, "it", FixedTextKey.NO_INFORMATION)
    offer = _text(clinic_pack, "it", FixedTextKey.CONTACT_OFFER)
    assert (result.outcome, result.text) == (TurnOutcome.HANDOFF, f"{no_information} {offer}")
    assert PHONE_DIGITS not in _digits(result.text)


async def test_a_message_the_classifier_cannot_read_gets_a_question_back(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    unclear = {"category": "uncertain", "intent": "other"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(unclear), document_responder()
    )
    result, _ = await handle_message("boh, quella cosa", ConversationState(turns=1), runtime)
    assert result.text == _text(clinic_pack, "it", FixedTextKey.CLARIFICATION)
    assert PHONE_DIGITS not in _digits(result.text)


async def test_asking_for_a_person_gives_the_contact(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    human = {"category": "administrative", "intent": "human"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(human), document_responder()
    )
    result, _ = await handle_message(
        "Posso parlare con qualcuno?", ConversationState(turns=1), runtime
    )
    assert (result.outcome, result.text) == (
        TurnOutcome.HANDOFF,
        _text(clinic_pack, "it", FixedTextKey.HANDOFF),
    )


async def test_a_crash_says_to_try_again_without_the_contact(
    activate_pack: Activate, clinic_pack: BusinessPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("frontdesk.conversation.turn.fact_evidence", broken)
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    result, _ = await handle_message("Quanto costa?", ConversationState(turns=1), runtime)
    assert (result.trace.rule, result.outcome) == ("error", TurnOutcome.FIXED)
    assert result.text == _text(clinic_pack, "it", FixedTextKey.UNAVAILABLE)


async def test_a_timeout_says_to_try_again(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"action": "handoff"})])

    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        FunctionModel(slow),
        timeout_seconds=0.5,
    )
    result, _ = await handle_message("C'è parcheggio?", ConversationState(turns=1), runtime)
    assert (result.trace.rule, result.outcome) == ("timeout", TurnOutcome.FIXED)
    assert result.text == _text(clinic_pack, "it", FixedTextKey.UNAVAILABLE)


async def test_a_first_answer_carries_the_brief_disclosure_when_the_pack_has_one(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    brief = "Ti risponde l'assistente automatico di Klinika Demo."
    texts = {
        language: {**texts, FixedTextKey.DISCLOSURE_BRIEF: brief}
        for language, texts in clinic_pack.fixed_texts.items()
    }
    pack = clinic_pack.model_copy(update={"fixed_texts": texts})
    runtime = make_runtime(pack, activate_pack(pack), classifier_model(PRICE), document_responder())
    result, _ = await handle_message("Quanto costa?", ConversationState(), runtime)
    assert result.text == f"{brief}\n\nFisioterapia: 30 EUR (a seduta)"
