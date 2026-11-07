"""The pack's next step (how to book, with the contact) ends a booking question, never an answer:
first instead of a generic closing question (Albi, 2026-10-11), then not at all (Albi, 2026-11-07:
no tail, the contact only when it is asked for)."""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, handle_message
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
STEP = "Per prenotare una visita, chiama la reception."
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "C'è parcheggio?",
}


def _with_next_step(pack: BusinessPack) -> BusinessPack:
    texts = {
        language: {**texts, FixedTextKey.NEXT_STEP: f"{STEP} ({language})"}
        for language, texts in pack.fixed_texts.items()
    }
    return pack.model_copy(update={"fixed_texts": texts})


async def test_an_answer_never_ends_with_the_next_step(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    # Albi, 2026-11-07: no tail after an answer; the next step ends a booking question only.
    pack = _with_next_step(clinic_pack)
    runtime = make_runtime(pack, activate_pack(pack), classifier_model(PRICE), document_responder())
    first, state = await handle_message("Quanto costa?", ConversationState(turns=1), runtime)
    assert first.text == "Fisioterapia: 30 EUR (a seduta)"
    assert not state.next_step_shown


async def test_no_next_step_after_the_contact_itself(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = _with_next_step(clinic_pack)
    args = {"category": "administrative", "intent": "contact"}
    runtime = make_runtime(pack, activate_pack(pack), classifier_model(args), document_responder())
    result, state = await handle_message("Come vi contatto?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert STEP not in result.text
    assert not state.next_step_shown


async def test_no_next_step_after_an_answer_that_hands_off(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    # One service has no price: the reply names it and offers the contact, not the next step.
    pack = _with_next_step(clinic_pack)
    args = {**PRICE, "service_ids": ["physiotherapy", "osteopathy"]}
    runtime = make_runtime(pack, activate_pack(pack), classifier_model(args), document_responder())
    result, state = await handle_message("Quanto costano?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text.endswith(pack.fixed_texts["it"][FixedTextKey.CONTACT_OFFER])  # type: ignore[index]
    assert STEP not in result.text
    assert not state.next_step_shown


async def test_no_next_step_after_a_document_answer_that_gives_the_contact(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    pack = _with_next_step(shop_pack)
    args = {
        "category": "administrative",
        "intent": "general_question",
        "general_question": "Come faccio un reso?",
    }
    runtime = make_runtime(pack, activate_pack(pack), classifier_model(args), document_responder())
    result, _ = await handle_message("Come faccio un reso?", ConversationState(turns=1), runtime)
    assert result.text.endswith(pack.fixed_texts["it"][FixedTextKey.HANDOFF])  # type: ignore[index]
    assert STEP not in result.text
