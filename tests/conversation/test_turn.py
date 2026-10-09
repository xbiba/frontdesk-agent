import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.vocab import ClarifyField
from frontdesk.conversation.compose import closing
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, handle_message
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import (
    SpyEmbedder,
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


def _text(pack: BusinessPack, language: str, key: FixedTextKey) -> str:
    return pack.fixed_texts[language][key]  # type: ignore[index]


def _handoff(pack: BusinessPack, language: str, *keys: FixedTextKey) -> str:
    return "\n\n".join(_text(pack, language, key) for key in (*keys, FixedTextKey.HANDOFF))


async def test_prices_are_written_from_the_record_without_the_responder(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PRICE),
        document_responder(prompts),
    )
    result, state = await handle_message(
        "Quanto costa la fisioterapia?", ConversationState(), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == (
        f"{_text(clinic_pack, 'it', FixedTextKey.DISCLOSURE)}\n\n"
        f"Fisioterapia: 30 EUR (a seduta)\n\n{closing('it')}"
    )
    assert result.trace.sources == ("price:physiotherapy",)
    assert prompts == []
    assert (state.turns, state.language, state.topic_service_ids) == (1, "it", ("physiotherapy",))


async def test_a_turn_reads_no_prompt_file_so_losing_one_costs_no_reply(
    activate_pack: Activate, clinic_pack: BusinessPack, agents_copy: Path
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), document_responder()
    )
    shutil.rmtree(agents_copy)  # every prompt file is gone once the agents are built
    emergency, _ = await handle_message("mi fa male il petto", ConversationState(turns=1), runtime)
    assert emergency.text == _handoff(clinic_pack, "it", FixedTextKey.EMERGENCY)
    assert emergency.trace.rule == "emergency"
    answer, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert answer.outcome is TurnOutcome.REPLIED
    assert answer.trace.rule == "pass"


async def test_albanian_without_diacritics_gets_albanian_texts(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    result, _ = await handle_message("sa kushton fizioterapia", ConversationState(turns=1), runtime)
    assert result.language == "sq"
    assert result.text.startswith("Fizioterapi: 30 EUR (për seancë)")


async def test_follow_ups_reuse_the_topic(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    instructions: list[str] = []
    classifier = classifier_turns(
        [
            {
                "category": "administrative",
                "intent": "service_info",
                "service_ids": ["physiotherapy"],
            },
            PRICE,
        ],
        instructions,
    )
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier, document_responder()
    )
    _, state = await handle_message(
        "Mi parli della fisioterapia?", ConversationState(turns=1), runtime
    )
    second, _ = await handle_message("E quanto costa?", state, runtime)
    assert "Topic from earlier turns: service_ids physiotherapy (Fisioterapia)." in instructions[1]
    assert second.text.startswith("Fisioterapia: 30 EUR (a seduta)")


async def test_a_price_without_a_service_gets_a_question_from_code(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "price"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, state = await handle_message("Quanto costa?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.ASKED
    assert result.text.startswith("Per quale servizio? Per esempio: ")
    assert state.pending_clarification is ClarifyField.SERVICE


async def test_rule_emergencies_never_reach_a_model(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(None, prompts),
        document_responder(),
    )
    result, _ = await handle_message("mi fa male il petto", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.EMERGENCY)
    assert result.trace.rule == "emergency"
    assert prompts == []


async def test_emergencies_flagged_by_the_classifier_take_the_same_fixed_path(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    args = {"category": "emergency", "intent": "price", "service_ids": ["physiotherapy"]}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder(prompts)
    )
    state = ConversationState(turns=1, topic_service_ids=("physiotherapy",))
    result, after = await handle_message("mi sento strano, aiuto", state, runtime)
    assert result.text == _handoff(clinic_pack, "it", FixedTextKey.EMERGENCY)
    assert result.trace.rule == "pass"
    assert prompts == []
    assert after.topic_service_ids == ()


@pytest.mark.parametrize(
    ("message", "outcome"),
    [
        ("", TurnOutcome.FIXED),
        ("   ", TurnOutcome.FIXED),
        ("👍👍", TurnOutcome.FIXED),
        ("?!", TurnOutcome.FIXED),
        ("a" * 2001, TurnOutcome.HANDOFF),
    ],
)
async def test_empty_emoji_and_long_messages_skip_the_models(
    activate_pack: Activate, clinic_pack: BusinessPack, message: str, outcome: TurnOutcome
) -> None:
    prompts: list[str] = []
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(None, prompts),
        document_responder(),
    )
    result, _ = await handle_message(message, ConversationState(turns=1, language="it"), runtime)
    assert result.outcome is outcome
    assert prompts == []


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        (
            {"category": "administrative", "intent": "price", "service_ids": ["returns-it"]},
            "unknown_service_id",
        ),
        (
            {
                "category": "administrative",
                "intent": "general_question",
                "general_question": "call me on +355 00 111 2222",
            },
            "text_has_identifiers",
        ),
        (
            {
                "category": "administrative",
                "intent": "general_question",
                "general_question": "my back pain, is it bad?",
            },
            "text_has_clinical_terms",
        ),
        (
            {"category": "orientation", "intent": "service_info", "problem_category_id": "heart"},
            "unknown_problem_category",
        ),
        ({"category": "administrative", "intent": "order_status"}, "intent_not_allowed"),
        (
            {"category": "administrative", "intent": "price", "order_reference": "ORD-10427"},
            "classifier_error",
        ),
    ],
)
async def test_manipulated_classifier_output_goes_to_reception(
    activate_pack: Activate, clinic_pack: BusinessPack, args: dict[str, Any], reason: str
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message("Ho una domanda", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rejection == reason
    assert result.text == _text(clinic_pack, "it", FixedTextKey.HANDOFF)


async def test_document_answers_are_quoted_and_checked(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), document_responder()
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text.startswith("Klinika Demo has six free parking spaces behind the building.")
    assert result.trace.sources == ("parking-en",)


def _quote_elsewhere(passage_id: str, content: str) -> dict[str, Any]:
    return {
        "text": "Parking is free.",
        "evidence_id": passage_id,
        "quote": "Parking is free on every day.",
    }


def _contradiction(passage_id: str, content: str) -> dict[str, Any]:
    return {
        "text": "Il parcheggio in strada è sempre gratuito.",
        "evidence_id": passage_id,
        "quote": "Street parking is paid on weekdays.",
    }


def _invented_number(passage_id: str, content: str) -> dict[str, Any]:
    return {
        "text": "There are 12 free parking spaces.",
        "evidence_id": passage_id,
        "quote": "six free parking spaces behind the building",
    }


def _price(passage_id: str, content: str) -> dict[str, Any]:
    return {
        "text": "Parking costs 2 euros.",
        "evidence_id": passage_id,
        "quote": "Street parking is paid on weekdays.",
    }


def _not_provided(passage_id: str, content: str) -> dict[str, Any]:
    return {
        "text": "There is parking.",
        "evidence_id": "D-00000000",
        "quote": "six free parking spaces behind the building",
    }


def _with_a_link(passage_id: str, content: str) -> dict[str, Any]:
    sentence = content.split(". ")[0].rstrip(".")
    return {
        "text": f"{sentence}. Book a space at www.parking-demo.example.",
        "evidence_id": passage_id,
        "quote": sentence,
    }


def _with_an_address(passage_id: str, content: str) -> dict[str, Any]:
    sentence = content.split(". ")[0].rstrip(".")
    return {
        "text": f"{sentence}. Write to parking@parking-demo.example.",
        "evidence_id": passage_id,
        "quote": sentence,
    }


def _with_a_markdown_link(passage_id: str, content: str) -> dict[str, Any]:
    sentence = content.split(". ")[0].rstrip(".")
    return {
        "text": f"{sentence}. See [the map](https://parking-demo.example/map).",
        "evidence_id": passage_id,
        "quote": sentence,
    }


@pytest.mark.eval
@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        (_quote_elsewhere, "quote_not_in_source"),
        (_contradiction, "unsupported_claim"),
        (_invented_number, "number_not_in_quote"),
        (_price, "price_in_answer"),
        (_not_provided, "unknown_evidence"),
        (_with_a_link, "link_not_in_quote"),
        (_with_an_address, "link_not_in_quote"),
        (_with_a_markdown_link, "link_not_in_quote"),
    ],
)
async def test_plausible_but_wrong_answers_never_reach_the_person(
    activate_pack: Activate, clinic_pack: BusinessPack, claim: Any, reason: str
) -> None:
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        document_responder(claim=claim),
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _handoff(clinic_pack, "en", FixedTextKey.NO_INFORMATION)
    assert reason in result.trace.check_reasons
    assert "D-00000000" not in result.trace.evidence_ids


async def test_clinical_message_with_a_practical_question(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {**PRICE, "category": "clinical"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Ho la febbre, quanto costa la fisioterapia?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == "Fisioterapia: 30 EUR (a seduta)\n\n" + _handoff(
        clinic_pack, "it", FixedTextKey.CLINICAL
    )


async def test_uncertain_messages_go_to_reception(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "uncertain", "intent": "other"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message("boh", ConversationState(turns=1, language="it"), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _text(clinic_pack, "it", FixedTextKey.HANDOFF)


async def test_orientation_is_written_from_the_map(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    prompts: list[str] = []
    args = {"category": "orientation", "intent": "service_info", "problem_category_id": "back"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder(prompts)
    )
    result, _ = await handle_message(
        "Ho mal di schiena, da chi devo andare?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text.startswith("Se ne occupano: Visita ortopedica, Osteopatia.")
    assert result.trace.sources == ("orientation:back",)
    assert prompts == []


async def test_bookings_go_to_reception_until_m4(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    args = {"category": "administrative", "intent": "booking", "service_ids": ["physiotherapy"]}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(args), document_responder()
    )
    result, _ = await handle_message(
        "Vorrei prenotare la fisioterapia", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _text(clinic_pack, "it", FixedTextKey.HANDOFF)


def _looping(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart("search_information", {"question": "Is there parking?"})]
    )


def _unavailable(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=503, model_name="test", body="SENTINEL-ERROR")


def _refusing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("I can't help with that.")])


@pytest.mark.parametrize(
    ("stage", "function", "reason"),
    [
        ("responder", _looping, "responder_error"),
        ("responder", _unavailable, "responder_error"),
        ("responder", _refusing, "responder_error"),
        ("classifier", _unavailable, "classifier_error"),
    ],
)
async def test_model_failures_hand_off(
    activate_pack: Activate, clinic_pack: BusinessPack, stage: str, function: Any, reason: str
) -> None:
    failing = FunctionModel(function)
    classifier = failing if stage == "classifier" else classifier_model(PARKING)
    responder = failing if stage == "responder" else document_responder()
    runtime = make_runtime(clinic_pack, activate_pack(clinic_pack), classifier, responder)
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == _text(clinic_pack, "en", FixedTextKey.HANDOFF)
    assert result.trace.rejection == reason


async def test_unexpected_errors_hand_off(
    activate_pack: Activate, clinic_pack: BusinessPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("frontdesk.conversation.turn.fact_evidence", broken)
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    result, _ = await handle_message(
        "Quanto costa la fisioterapia?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rule == "error"
    assert result.text == _text(clinic_pack, "it", FixedTextKey.HANDOFF)


async def test_slow_turns_time_out(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
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
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.rule == "timeout"


async def test_message_text_never_reaches_responder_embedder_or_later_turns(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    classifier_prompts: list[str] = []
    responder_prompts: list[str] = []
    spy = SpyEmbedder()
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING, classifier_prompts),
        document_responder(responder_prompts),
        spy,
    )
    _, state = await handle_message(
        "Is there parking? SENTINELXYZ", ConversationState(turns=1), runtime
    )
    await handle_message("And for disabled people?", state, runtime)
    assert "SENTINELXYZ" in classifier_prompts[0]
    assert all("SENTINELXYZ" not in prompt for prompt in classifier_prompts[1:])
    assert responder_prompts
    assert all("SENTINELXYZ" not in prompt for prompt in responder_prompts)
    assert spy.inputs
    assert all("SENTINELXYZ" not in text for text in spy.inputs)
