"""A refused answer is kept for the reviewer only when the runtime asks for it (M3 design §4.4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import (
    DraftClaim,
    RejectedDraft,
    TurnOutcome,
    TurnRuntime,
    handle_message,
)
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Is there free parking?",
}
INVENTED = "There are fifty free parking spaces."


def _invented(passage_id: str, content: str) -> dict[str, str]:
    return {"text": INVENTED, "evidence_id": passage_id, "quote": "free parking spaces"}


@pytest.mark.parametrize("keep", [True, False])
async def test_a_refused_answer_is_kept_only_when_asked(
    activate_pack: Activate, clinic_pack: BusinessPack, keep: bool
) -> None:
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        document_responder(claim=_invented),
    )
    result, _ = await handle_message(
        "Is there free parking?", ConversationState(turns=1), replace(runtime, keep_drafts=keep)
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert "number_not_in_quote" in result.trace.check_reasons
    if not keep:
        assert result.draft is None
        return
    assert result.draft is not None
    [claim] = result.draft.claims
    assert result.draft.action == "answer"
    assert claim.text == INVENTED
    assert claim.quote == "free parking spaces"
    assert claim.evidence_id.startswith("D-")


async def test_an_answer_that_passes_keeps_no_draft(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), document_responder()
    )
    result, _ = await handle_message(
        "Is there free parking?", ConversationState(turns=1), replace(runtime, keep_drafts=True)
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.draft is None


def test_a_draft_is_plain_frozen_data() -> None:
    draft = RejectedDraft(
        action="answer", claims=(DraftClaim("t", "D-00000000", "q"),), clarify=None
    )
    with pytest.raises(AttributeError):
        draft.action = "handoff"  # type: ignore[misc]


def _answering(args: dict[str, Any]) -> FunctionModel:
    """A responder that gives these tool-call arguments for every request."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(model)


def _keeping(
    pack: BusinessPack,
    factory: sessionmaker[Session],
    classifier: dict[str, Any],
    responder: FunctionModel,
) -> TurnRuntime:
    runtime = make_runtime(pack, factory, classifier_model(classifier), responder)
    return replace(runtime, keep_drafts=True)


async def test_a_question_the_profile_does_not_offer_keeps_its_draft(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    responder = _answering({"action": "clarify", "clarify": "order_details"})
    runtime = _keeping(clinic_pack, activate_pack(clinic_pack), PARKING, responder)
    result, _ = await handle_message("Is there free parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.check_reasons == ("clarify_not_offered",)
    assert result.draft == RejectedDraft(action="clarify", claims=(), clarify="order_details")


async def test_a_question_the_profile_offers_was_not_refused_so_no_draft_is_kept(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    responder = _answering({"action": "clarify", "clarify": "service"})
    runtime = _keeping(clinic_pack, activate_pack(clinic_pack), PARKING, responder)
    result, _ = await handle_message("Is there free parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.ASKED
    assert result.draft is None


async def test_a_handoff_the_responder_chose_was_not_refused_so_no_draft_is_kept(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = _keeping(
        clinic_pack, activate_pack(clinic_pack), PARKING, _answering({"action": "handoff"})
    )
    result, _ = await handle_message("Is there free parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.trace.check_reasons == ()
    assert result.draft is None


async def test_a_draft_survives_the_clinical_notice_that_follows_an_administrative_answer(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    clinical_parking = {**PARKING, "category": "clinical"}
    runtime = _keeping(
        clinic_pack,
        activate_pack(clinic_pack),
        clinical_parking,
        document_responder(claim=_invented),
    )
    result, _ = await handle_message(
        "Ho la febbre, c'è parcheggio gratis?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text.startswith(clinic_pack.fixed_texts["it"][FixedTextKey.CLINICAL])
    assert "number_not_in_quote" in result.trace.check_reasons
    assert result.draft is not None
    assert result.draft.claims[0].text == INVENTED


async def test_a_draft_survives_the_disclosure_that_opens_a_conversation(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = _keeping(
        clinic_pack,
        activate_pack(clinic_pack),
        PARKING,
        document_responder(claim=_invented),
    )
    result, _ = await handle_message("Is there free parking?", ConversationState(turns=0), runtime)
    disclosure = clinic_pack.fixed_texts[result.language][FixedTextKey.DISCLOSURE]
    assert result.text.startswith(disclosure)
    assert result.draft is not None
    assert result.draft.claims[0].text == INVENTED


async def test_a_runtime_keeps_no_refused_answer_unless_it_is_asked_to(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    # No `replace(keep_drafts=...)` here: the default decides. The live evaluation asks, nothing
    # else does, because a draft is model text derived from a message.
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(PARKING),
        document_responder(claim=_invented),
    )
    assert runtime.keep_drafts is False
    result, _ = await handle_message("Is there free parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert "number_not_in_quote" in result.trace.check_reasons
    assert result.draft is None
