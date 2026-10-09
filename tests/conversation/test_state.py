from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.conversation.state import TOPIC_TURNS, ConversationState


def _advance(state: ConversationState, **changes: object) -> ConversationState:
    values: dict[str, object] = {
        "language": "it",
        "intent": Intent.HOURS,
        "category": Category.ADMINISTRATIVE,
        "service_ids": (),
        "doctor_ids": (),
        "clarify": None,
    }
    return state.advanced(**{**values, **changes})  # type: ignore[arg-type]


def test_conversations_get_distinct_ids() -> None:
    assert ConversationState().conversation_id != ConversationState().conversation_id


def test_topic_is_kept_for_a_few_turns_then_dropped() -> None:
    state = _advance(
        ConversationState(), intent=Intent.SERVICE_INFO, service_ids=("physiotherapy",)
    )
    assert state.topic_service_ids == ("physiotherapy",)
    for _ in range(TOPIC_TURNS - 1):
        state = _advance(state)
        assert state.topic_service_ids == ("physiotherapy",)
    assert _advance(state).topic_service_ids == ()


def test_new_ids_replace_the_topic_and_emergencies_clear_it() -> None:
    state = _advance(ConversationState(), service_ids=("physiotherapy",))
    state = _advance(state, doctor_ids=("dr-arben-demo",))
    assert (state.topic_service_ids, state.topic_doctor_ids) == ((), ("dr-arben-demo",))
    cleared = _advance(state, category=Category.EMERGENCY, intent=Intent.OTHER)
    assert (cleared.topic_service_ids, cleared.topic_doctor_ids) == ((), ())


def test_pending_clarification_lasts_one_turn_and_intents_are_bounded() -> None:
    state = _advance(ConversationState(), clarify=ClarifyField.SERVICE)
    assert state.pending_clarification is ClarifyField.SERVICE
    assert _advance(state).pending_clarification is None
    for _ in range(10):
        state = _advance(state)
    assert len(state.previous_intents) == 5
