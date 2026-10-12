"""The voice in a turn (Albi's live chat, 2026-10-12: "it reads like a stupid chatbot"): the model
rewrites what code wrote, in context; code keeps the facts, and the fixed texts that must stay fixed
never reach it."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai.models import Model
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classifier import build_classifier
from frontdesk.agent.voice import build_voice
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnRuntime, handle_message
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import (
    classifier_turns,
    document_responder,
    make_runtime,
    voice_model,
)

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
STEP = "Per prenotare una visita, chiama la reception al +355 00 000 0000."
HANDOFF = "Per parlare con una persona, chiama la reception al +355 00 000 0000."
GREET = {"category": "administrative", "intent": "greeting"}
HUMAN = {"category": "administrative", "intent": "human"}
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["orthopedics-visit"]}


@pytest.fixture
def pack(clinic_pack: BusinessPack) -> BusinessPack:
    texts = {
        language: {**texts, FixedTextKey.NEXT_STEP: STEP}
        for language, texts in clinic_pack.fixed_texts.items()
    }
    return clinic_pack.model_copy(update={"fixed_texts": texts})


@pytest.fixture
def base(activate_pack: Activate, pack: BusinessPack) -> TurnRuntime:
    return make_runtime(pack, activate_pack(pack), classifier_turns([], []), document_responder())


def _runtime(base: TurnRuntime, outputs: list[dict[str, Any]], voice: Model) -> TurnRuntime:
    profile = base.pack.profile
    return replace(
        base,
        classifier=build_classifier(classifier_turns(outputs, []), profile),
        voice=build_voice(voice, profile),
    )


async def _say(runtime: TurnRuntime, message: str, turns: int = 1) -> Any:
    return await handle_message(message, ConversationState(turns=turns), runtime)


async def test_small_talk_gets_the_voice(base: TurnRuntime) -> None:
    voiced = "Bene, grazie! Come posso aiutarti? Puoi chiedere di servizi, medici o prezzi."
    result, _ = await _say(_runtime(base, [GREET], voice_model(voiced)), "come va?")
    assert result.text == voiced
    assert result.trace.voice == "rewritten"


async def test_a_rewrite_that_changes_the_phone_keeps_the_code_text(base: TurnRuntime) -> None:
    wrong = voice_model("Certo, chiama la reception al +355 00 000 0001.")
    result, _ = await _say(_runtime(base, [HUMAN], wrong), "vorrei parlare con un operatore")
    assert (result.outcome, result.text) == (TurnOutcome.HANDOFF, HANDOFF)
    assert result.trace.voice == "numbers"


async def test_a_voice_that_fails_keeps_the_code_text(base: TurnRuntime) -> None:
    def broken(message: str) -> str:
        raise RuntimeError("the model is down")

    result, _ = await _say(
        _runtime(base, [HUMAN], voice_model(broken)), "vorrei parlare con un operatore"
    )
    assert result.text == HANDOFF
    assert result.trace.voice == "failed"


async def test_an_emergency_never_reaches_the_voice(base: TurnRuntime) -> None:
    calls: list[tuple[str, str]] = []
    runtime = _runtime(base, [], voice_model("no", calls))
    result, _ = await _say(runtime, "Ho un forte dolore al petto e non respiro")
    assert result.text.startswith(runtime.pack.fixed_texts["it"][FixedTextKey.EMERGENCY])  # type: ignore[index]
    assert calls == []


async def test_the_voice_reads_the_message_the_draft_and_the_previous_reply(
    base: TurnRuntime,
) -> None:
    calls: list[tuple[str, str]] = []
    runtime = _runtime(base, [HUMAN], voice_model(HANDOFF, calls))
    state = ConversationState(turns=2, last_reply="Visita ortopedica: 40 EUR")
    await handle_message("vorrei parlare con qualcuno", state, runtime)
    ((message, instructions),) = calls
    assert message == "vorrei parlare con qualcuno"
    assert f"The draft:\n<<<\n{HANDOFF}\n>>>" in instructions
    assert "The assistant's previous message:\n<<<\nVisita ortopedica: 40 EUR\n>>>" in instructions


async def test_the_first_reply_still_opens_with_the_disclosure(base: TurnRuntime) -> None:
    voiced = "Bene, grazie! Come posso aiutarti?"
    result, _ = await _say(_runtime(base, [GREET], voice_model(voiced)), "come va?", turns=0)
    disclosure = base.pack.fixed_texts["it"][FixedTextKey.DISCLOSURE]  # type: ignore[index]
    assert result.text == f"{disclosure}\n\n{voiced}"


async def test_a_next_step_said_in_other_words_still_counts_as_said(base: TurnRuntime) -> None:
    voiced = (
        "La visita ortopedica costa 40 EUR. Per prenotarla chiama la reception al +355 00 000 0000."
    )
    runtime = _runtime(base, [PRICE], voice_model(voiced))
    result, state = await _say(runtime, "quanto costa la visita ortopedica?")
    assert result.text == voiced
    assert state.next_step_shown
