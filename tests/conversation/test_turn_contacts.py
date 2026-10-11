"""A document answer that sends the person to a contact gives it: the system writes it from the
pack, as it does every contact (live review of 2026-10-11: "write to the address given", and no
address)."""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, handle_message
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
RETURNS = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Come faccio un reso?",
}
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Is there parking?",
}


async def test_an_answer_from_a_passage_that_names_the_business_contact_gives_it(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        shop_pack, activate_pack(shop_pack), classifier_model(RETURNS), document_responder()
    )
    result, _ = await handle_message("Come faccio un reso?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    contact = shop_pack.fixed_texts["it"][FixedTextKey.HANDOFF]  # type: ignore[index]
    assert contact in result.text
    assert shop_pack.location.email in result.text


async def test_an_answer_from_a_passage_without_a_contact_adds_none(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PARKING), document_responder()
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert clinic_pack.location.phone not in result.text
    assert clinic_pack.location.email not in result.text
