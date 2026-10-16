"""The worker's side (M4 design §4 steps 4-8, D6, D8-D12): one turn under a lease, the burst
window, the staff's pause, replies sent at most once, and the limits."""

import logging
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import (
    TESTER_WELCOME,
    Inbox,
    InStatus,
    OutStatus,
    conversation_key,
    queue_reply,
)
from frontdesk.channels.messages import Inbound, InboundKind, SendOutcome, SendStatus
from frontdesk.channels.requests import Decision
from frontdesk.channels.runner import MAX_REPLIES_A_DAY, Outcome, Runner, Turn
from frontdesk.channels.whatsapp import TEMPLATE
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnResult
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import ChannelConversation, ChannelMessage
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.channels.fakes import NOW, Clock, FakeSender, FakeTurn

pytestmark = pytest.mark.db
BUSINESS = "demo-clinic"
CHANNEL = "instagram"
ACCOUNT = "17841400000000001"
ALICE = "1111111111111111"
CODE = "mcu-prova-1234"
KEY = conversation_key(BUSINESS, CHANNEL, ALICE)
LONG = " ".join(f"Frase numero {n}." for n in range(200))


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def inbox(factory: sessionmaker[Session], clock: Clock) -> Inbox:
    """Alice is a tester: her pairing code left the welcome queued."""
    inbox = Inbox(
        factory,
        business_id=BUSINESS,
        channel=CHANNEL,
        account_id=ACCOUNT,
        testers_only=True,
        pairing_code=CODE,
        now=clock,
    )
    inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.TEXT, "pair", CODE, NOW)])
    return inbox


def _runner(
    factory: sessionmaker[Session],
    clock: Clock,
    pack: BusinessPack,
    sender: FakeSender,
    turn: Turn,
    *,
    off: bool = False,
) -> Runner:
    return Runner(factory, pack=pack, channel=CHANNEL, sender=sender, turn=turn, now=clock, off=off)


async def _welcomed(runner: Runner) -> None:
    assert await runner.process(KEY) is Outcome.IDLE  # the run sends the queued welcome


def _say(
    inbox: Inbox, mid: str, text: str | None = None, kind: InboundKind = InboundKind.TEXT
) -> None:
    inbox.accept([Inbound(ACCOUNT, ALICE, kind, mid, text, NOW)])


def _staff(inbox: Inbox) -> None:
    inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.ECHO, "staff-1", "Sono Anna", NOW)])


def _messages(factory: sessionmaker[Session], direction: str) -> list[ChannelMessage]:
    with business_session(factory, BUSINESS) as session:
        return list(
            session.scalars(
                select(ChannelMessage)
                .where(ChannelMessage.direction == direction)
                .order_by(ChannelMessage.created_at)
            )
        )


def _conversation(factory: sessionmaker[Session]) -> ChannelConversation:
    with business_session(factory, BUSINESS) as session:
        conversation = session.get(ChannelConversation, (BUSINESS, KEY))
    assert conversation is not None
    return conversation


def _replies(sender: FakeSender) -> list[str]:
    return [text for _, text in sender.sent[1:]]  # what came after the welcome


async def test_a_turn_answers_and_the_reply_is_sent_once(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender, turn = FakeSender(), FakeTurn()
    runner = _runner(factory, clock, clinic_pack, sender, turn)
    await _welcomed(runner)
    _say(inbox, "m1", "Siete aperti sabato?")
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.DONE
    assert await runner.process(KEY) is Outcome.IDLE  # a second task finds nothing
    assert sender.sent[1:] == [(ALICE, "Risposta: Siete aperti sabato?")]
    reply = _messages(factory, "out")[-1]
    assert (reply.status, reply.external_id) == (OutStatus.SENT, "out-2")
    (question,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert (question.status, question.text) == (InStatus.READ, None)
    conversation = _conversation(factory)
    assert conversation.state is not None
    assert conversation.state["turns"] == 1
    assert conversation.lease_until is None


async def test_a_message_younger_than_the_burst_waits(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()
    runner = _runner(factory, clock, clinic_pack, FakeSender(), turn)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=1)
    assert await runner.process(KEY) is Outcome.WAIT
    assert turn.texts == []
    clock.advance(seconds=2)
    assert await runner.process(KEY) is Outcome.DONE


async def test_a_burst_across_the_window_becomes_one_turn(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()
    runner = _runner(factory, clock, clinic_pack, FakeSender(), turn)
    _say(inbox, "m1", "Buongiorno")
    clock.advance(seconds=2.5)
    _say(inbox, "m2", "siete aperti sabato?")
    clock.advance(seconds=0.5)  # the first message's task
    assert await runner.process(KEY) is Outcome.WAIT
    clock.advance(seconds=2.5)  # the second message's task
    assert await runner.process(KEY) is Outcome.DONE
    assert turn.texts == ["Buongiorno\nsiete aperti sabato?"]


async def test_a_conversation_held_by_another_worker_is_busy(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    with business_session(factory, BUSINESS) as session:
        session.execute(update(ChannelConversation).values(lease_until=NOW + timedelta(seconds=60)))
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    assert await runner.process(KEY) is Outcome.BUSY
    assert sender.sent == []


async def test_an_unknown_conversation_is_idle(
    factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    runner = _runner(factory, clock, clinic_pack, FakeSender(), FakeTurn())
    assert await runner.process(conversation_key(BUSINESS, CHANNEL, "nobody")) is Outcome.IDLE


async def test_an_empty_reply_sends_nothing(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn(reply=lambda text: "  \n "))
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.DONE
    assert _replies(sender) == []
    assert len(_messages(factory, "out")) == 1  # the welcome only


async def test_a_long_reply_goes_in_parts_in_order(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn(reply=lambda text: LONG))
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    await runner.process(KEY)
    parts = _replies(sender)
    assert len(parts) >= 3
    assert all(len(part.encode()) <= 1000 for part in parts)
    assert " ".join(parts).split() == LONG.split()


async def test_a_failed_part_stops_the_rest(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    clinic_pack: BusinessPack,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sender = FakeSender(
        SendOutcome(SendStatus.SENT, message_id="out-1"),
        SendOutcome(SendStatus.FAILED, reason="status_400"),
    )
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn(reply=lambda text: LONG))
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.runner"):
        await runner.process(KEY)
    statuses = [m.status for m in _messages(factory, "out")[1:]]
    assert statuses[0] == OutStatus.FAILED
    assert set(statuses[1:]) == {OutStatus.DROPPED}
    assert len(sender.sent) == 2
    assert any(
        r.levelno == logging.ERROR and r.msg == "channel.send_failed" for r in caplog.records
    )


async def test_an_uncertain_send_is_never_sent_again(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender(SendOutcome(SendStatus.UNCERTAIN, reason="transport"))
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    assert await runner.process(KEY) is Outcome.IDLE
    assert await runner.process(KEY) is Outcome.IDLE
    assert len(sender.sent) == 1
    assert _messages(factory, "out")[0].status == OutStatus.UNCERTAIN


async def test_a_crash_after_the_turn_sends_the_reply_on_the_next_run(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    clinic_pack: BusinessPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)

    async def crash(key: str, user_id: str) -> None:
        raise RuntimeError("the worker stopped")

    monkeypatch.setattr(runner, "_deliver", crash)
    with pytest.raises(RuntimeError, match="stopped"):
        await runner.process(KEY)
    assert _messages(factory, "out")[-1].status == OutStatus.QUEUED
    monkeypatch.undo()
    assert await runner.process(KEY) is Outcome.IDLE
    assert _replies(sender) == ["Risposta: Ciao"]


async def test_a_turn_that_keeps_failing_ends_in_the_handoff_text(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    async def broken(text: str, state: ConversationState) -> tuple[TurnResult, ConversationState]:
        raise RuntimeError("model down")

    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, broken)
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="model down"):
            await runner.process(KEY)
    assert await runner.process(KEY) is Outcome.FAILED
    handoff = clinic_pack.fixed_texts[clinic_pack.default_language][FixedTextKey.HANDOFF]
    assert sender.sent[-1] == (ALICE, handoff)
    (question,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert (question.status, question.attempts, question.text) == (InStatus.FAILED, 3, None)


async def test_the_staff_pause_discards_waiting_messages(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender, turn = FakeSender(), FakeTurn()
    runner = _runner(factory, clock, clinic_pack, sender, turn)
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    _staff(inbox)
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.PAUSED
    assert turn.texts == []
    assert _replies(sender) == []
    (question,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert (question.status, question.text) == (InStatus.DISCARDED, None)


async def test_after_the_pause_the_bot_starts_over(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()
    runner = _runner(factory, clock, clinic_pack, FakeSender(), turn)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    await runner.process(KEY)
    _staff(inbox)
    clock.advance(hours=25)
    _say(inbox, "m2", "Ci siete ancora?")
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.DONE
    assert [state.turns for state in turn.states] == [0, 0]
    assert _conversation(factory).paused_until is None


async def test_a_state_older_than_a_week_starts_over(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()
    runner = _runner(factory, clock, clinic_pack, FakeSender(), turn)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    await runner.process(KEY)
    clock.advance(days=8)
    _say(inbox, "m2", "Buongiorno")
    clock.advance(seconds=3)
    await runner.process(KEY)
    assert [state.turns for state in turn.states] == [0, 0]


async def test_the_switch_drops_replies(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn(), off=True)
    await runner.process(KEY)
    assert sender.sent == []
    (welcome,) = _messages(factory, "out")
    assert (welcome.status, welcome.reason) == (OutStatus.DROPPED, "switch")


async def test_a_reply_after_the_24_hour_window_is_not_sent(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(hours=24, minutes=1)
    assert await runner.process(KEY) is Outcome.DONE
    assert _replies(sender) == []
    reply = _messages(factory, "out")[-1]
    assert (reply.status, reply.reason) == (OutStatus.DROPPED, "window")


async def test_attachments_get_the_fixed_text_once(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender, turn = FakeSender(), FakeTurn()
    runner = _runner(factory, clock, clinic_pack, sender, turn)
    await _welcomed(runner)
    _say(inbox, "a1", kind=InboundKind.ATTACHMENT)
    _say(inbox, "a2", kind=InboundKind.ATTACHMENT)
    _say(inbox, "m1", "Che ne dice?")
    clock.advance(seconds=3)
    await runner.process(KEY)
    attachment = clinic_pack.fixed_texts["it"][FixedTextKey.ATTACHMENT]
    assert _replies(sender) == ["Risposta: Che ne dice?", attachment]
    assert turn.texts == ["Che ne dice?"]


async def test_only_one_slow_down_reply_a_minute(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    turn = FakeTurn(reply=lambda text: "Rallenti, per favore.", rule="rate_limit")
    runner = _runner(factory, clock, clinic_pack, sender, turn)
    await _welcomed(runner)
    for mid in ("m1", "m2"):
        _say(inbox, mid, "ancora")
        clock.advance(seconds=3)
        await runner.process(KEY)
    assert _replies(sender) == ["Rallenti, per favore."]
    clock.advance(minutes=1)
    _say(inbox, "m3", "ancora")
    clock.advance(seconds=3)
    await runner.process(KEY)
    assert len(_replies(sender)) == 2


async def test_at_most_sixty_replies_a_day(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    clinic_pack: BusinessPack,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    await _welcomed(runner)
    with business_session(factory, BUSINESS) as session:
        parts = [("reply", f"risposta {n}") for n in range(MAX_REPLIES_A_DAY)]
        queue_reply(
            session,
            business_id=BUSINESS,
            channel=CHANNEL,
            conversation_id=KEY,
            parts=parts,
            now=NOW,
        )
        session.flush()
        session.execute(
            update(ChannelMessage)
            .where(ChannelMessage.status == OutStatus.QUEUED)
            .values(status=OutStatus.SENT)
        )
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    turn = FakeTurn()
    runner = _runner(factory, clock, clinic_pack, sender, turn)
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.runner"):
        assert await runner.process(KEY) is Outcome.DONE
    assert _replies(sender) == []
    assert any(r.levelno == logging.ERROR and r.msg == "channel.reply_cap" for r in caplog.records)
    # Final review of M4, M8: past the cap no turn runs, so no model call is paid for it.
    assert turn.texts == []
    (message,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert (message.status, message.text) == (InStatus.READ, None)


async def test_the_switch_runs_no_turn_and_leaves_the_messages_waiting(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    # Final review of M4, M3: switched off, a stored message still went to the model, and was
    # never answered once the switch was back on.
    sender, turn = FakeSender(), FakeTurn()
    await _welcomed(_runner(factory, clock, clinic_pack, sender, turn))
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    off = _runner(factory, clock, clinic_pack, sender, turn, off=True)
    assert await off.process(KEY) is Outcome.OFF
    assert turn.texts == []
    (message,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert (message.status, message.attempts) == (InStatus.PENDING, 0)
    assert await _runner(factory, clock, clinic_pack, sender, turn).process(KEY) is Outcome.DONE
    assert _replies(sender) == ["Risposta: Ciao"]


async def test_a_worker_that_lost_its_lease_queues_and_sends_nothing(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    other = uuid.uuid4()
    turn = FakeTurn()

    async def overrun(text: str, state: ConversationState) -> tuple[TurnResult, ConversationState]:
        # The turn outlived the lease, and another worker took the conversation meanwhile.
        with business_session(factory, BUSINESS) as session:
            session.execute(
                update(ChannelConversation).values(
                    lease_owner=other, lease_until=NOW + timedelta(minutes=5)
                )
            )
        return await turn(text, state)

    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, overrun)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.BUSY
    assert sender.sent == []
    assert [m.kind for m in _messages(factory, "out")] == ["fixed"]  # the welcome only
    (question,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert question.status == InStatus.PENDING  # left to the worker that holds the lease
    assert _conversation(factory).lease_owner == other  # whose lease stays


async def test_a_reply_listed_by_two_workers_is_sent_once(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    clinic_pack: BusinessPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = FakeSender()
    first = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    second = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    listed = first._queued(KEY)  # what a second worker saw before the first one sent
    await first._deliver(KEY, ALICE)
    monkeypatch.setattr(second, "_queued", lambda key: listed)
    await second._deliver(KEY, ALICE)
    assert sender.sent == [(ALICE, TESTER_WELCOME)]


async def test_a_staff_reply_during_the_turn_keeps_the_pause(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()

    async def staff_meanwhile(
        text: str, state: ConversationState
    ) -> tuple[TurnResult, ConversationState]:
        _staff(inbox)  # the staff answers in the thread while the model composes
        return await turn(text, state)

    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, staff_meanwhile)
    await _welcomed(runner)
    _say(inbox, "m1", "Ciao")
    clock.advance(seconds=3)
    await runner.process(KEY)
    assert _replies(sender) == []
    assert _conversation(factory).paused_until is not None


# --- WhatsApp W1: taps decided by code, templates, the staff's fixed reply ------------------------
ONLY_BUTTONS = "Per ora rispondo solo ai pulsanti delle richieste di prenotazione."
TAP = "request:" + "ab" * 16 + ":confirm"


async def test_a_tap_is_decided_by_code_and_never_reaches_the_turn(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    taps: list[tuple[str, str]] = []

    async def buttons(user_id: str, text: str) -> Decision:
        taps.append((user_id, text))
        return Decision("Confermata: venerdì 23 ottobre, 10:00.")

    sender, turn = FakeSender(), FakeTurn()
    runner = Runner(
        factory,
        pack=clinic_pack,
        channel=CHANNEL,
        sender=sender,
        turn=turn,
        now=clock,
        buttons=buttons,
    )
    await _welcomed(runner)
    _say(inbox, "b1", TAP, kind=InboundKind.BUTTON)
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.DONE
    assert turn.texts == []
    assert taps == [(ALICE, TAP)]
    assert _replies(sender) == ["Confermata: venerdì 23 ottobre, 10:00."]
    (tap,) = [m for m in _messages(factory, "in") if m.external_id == "b1"]
    assert (tap.status, tap.text) == (InStatus.READ, None)


async def test_a_template_part_goes_out_outside_the_24_hour_window(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    sender = FakeSender()
    runner = _runner(factory, clock, clinic_pack, sender, FakeTurn())
    await _welcomed(runner)
    clock.advance(hours=30)
    with business_session(factory, BUSINESS) as session:
        queue_reply(
            session,
            business_id=BUSINESS,
            channel=CHANNEL,
            conversation_id=KEY,
            parts=[(TEMPLATE, '{"name": "t"}'), ("fixed", "Una risposta")],
            now=clock(),
        )
    await runner.process(KEY)
    assert sender.sent[1:] == [(ALICE, '{"name": "t"}')]
    assert sender.kinds[1:] == [TEMPLATE]  # the text after it waits for no window: dropped


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([("m1", "Ciao", InboundKind.TEXT)], id="text"),
        pytest.param([("m1", None, InboundKind.ATTACHMENT)], id="voice-note"),
        pytest.param(
            [("m1", "Ciao", InboundKind.TEXT), ("m2", None, InboundKind.ATTACHMENT)], id="both"
        ),
    ],
)
async def test_a_staff_text_or_voice_note_gets_the_fixed_reply(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    clinic_pack: BusinessPack,
    messages: list[tuple[str, str | None, InboundKind]],
) -> None:
    # Review Focus 5: one fixed reply, no model call, no silence.
    sender, turn = FakeSender(), FakeTurn()
    runner = Runner(
        factory,
        pack=clinic_pack,
        channel=CHANNEL,
        sender=sender,
        turn=turn,
        now=clock,
        fixed_reply=ONLY_BUTTONS,
    )
    await _welcomed(runner)
    for mid, text, kind in messages:
        _say(inbox, mid, text, kind=kind)
    clock.advance(seconds=3)
    assert await runner.process(KEY) is Outcome.DONE
    assert turn.texts == []
    assert _replies(sender) == [ONLY_BUTTONS]
