"""The limits of a channel anyone may write to (the site's chat, design of 2026-11-07 §3): past the
messages of a conversation, at the third identical message in a row, past the channel's turns of
the day, the handoff text goes out and the turn is never called."""

import logging

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import Inbox, OutStatus, conversation_key, queue_reply
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.channels.runner import ChannelLimits, Outcome, Runner, text_digest
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import ChannelConversation, ChannelMessage
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.channels.fakes import NOW, Clock, FakeSender, FakeTurn

pytestmark = pytest.mark.db
BUSINESS = "demo-clinic"
CHANNEL = "web"
SESSION = "5f0c6a1e-2b7d-4c59-9d2e-0123456789ab"
KEY = conversation_key(BUSINESS, CHANNEL, SESSION)
LIMITS = ChannelLimits(per_conversation=4, repeats=2, turns_a_day=5)


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def inbox(factory: sessionmaker[Session], clock: Clock) -> Inbox:
    return Inbox(
        factory,
        business_id=BUSINESS,
        channel=CHANNEL,
        account_id=BUSINESS,
        testers_only=False,
        pairing_code=None,
        now=clock,
    )


def _runner(
    factory: sessionmaker[Session], clock: Clock, pack: BusinessPack, turn: FakeTurn
) -> tuple[Runner, FakeSender]:
    sender = FakeSender()
    runner = Runner(
        factory, pack=pack, channel=CHANNEL, sender=sender, turn=turn, now=clock, limits=LIMITS
    )
    return runner, sender


async def _say(inbox: Inbox, runner: Runner, clock: Clock, mid: str, text: str) -> Outcome:
    inbox.accept([Inbound(BUSINESS, SESSION, InboundKind.TEXT, mid, text)])
    clock.advance(seconds=3)
    return await runner.process(KEY)


def _handoff(pack: BusinessPack, language: str = "it") -> str:
    """In the conversation's language: the fake turn answers in Italian."""
    return pack.fixed_texts[language][FixedTextKey.HANDOFF]  # type: ignore[index]


async def test_the_third_identical_message_gets_the_handoff_without_the_model(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    clinic_pack: BusinessPack,
    caplog: pytest.LogCaptureFixture,
) -> None:
    turn = FakeTurn()
    runner, sender = _runner(factory, clock, clinic_pack, turn)
    await _say(inbox, runner, clock, "m1", "Quanto costa?")
    await _say(inbox, runner, clock, "m2", "quanto  COSTA?")
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.runner"):
        assert await _say(inbox, runner, clock, "m3", "Quanto costa?") is Outcome.DONE
    assert turn.texts == ["Quanto costa?", "quanto  COSTA?"]
    assert sender.sent[-1] == (SESSION, _handoff(clinic_pack))
    assert sender.kinds[-1] == "fixed"
    limited = [r for r in caplog.records if r.msg == "channel.limited"]
    assert [r.fields["reason"] for r in limited] == ["repeated"]  # type: ignore[attr-defined]
    # Another question starts the count again.
    await _say(inbox, runner, clock, "m4", "Dove siete?")
    assert turn.texts[-1] == "Dove siete?"


async def test_the_digest_keeps_no_text(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    runner, _ = _runner(factory, clock, clinic_pack, FakeTurn())
    await _say(inbox, runner, clock, "m1", "Quanto costa?")
    with business_session(factory, BUSINESS) as session:
        conversation = session.get(ChannelConversation, (BUSINESS, KEY))
    assert conversation is not None
    assert conversation.last_digest == text_digest(KEY, "quanto costa?")
    assert conversation.repeats == 1
    assert text_digest(KEY, "x") != text_digest(conversation_key(BUSINESS, CHANNEL, "other"), "x")


async def test_a_conversation_past_its_messages_gets_the_handoff(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()
    runner, sender = _runner(factory, clock, clinic_pack, turn)
    for n in range(LIMITS.per_conversation):
        await _say(inbox, runner, clock, f"m{n}", f"Domanda {n}")
    assert len(turn.texts) == LIMITS.per_conversation
    await _say(inbox, runner, clock, "last", "Un'altra domanda")
    assert len(turn.texts) == LIMITS.per_conversation
    assert sender.sent[-1][1] == _handoff(clinic_pack)


async def test_the_channel_past_its_turns_of_the_day_gets_the_handoff(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    other = inbox.open("someone-else-entirely")
    with business_session(factory, BUSINESS) as session:
        parts = [("reply", f"risposta {n}") for n in range(LIMITS.turns_a_day)]
        queue_reply(
            session,
            business_id=BUSINESS,
            channel=CHANNEL,
            conversation_id=other,
            parts=parts,
            now=NOW,
        )
        session.flush()
        session.execute(update(ChannelMessage).values(status=OutStatus.SENT))
    turn = FakeTurn()
    runner, sender = _runner(factory, clock, clinic_pack, turn)
    await _say(inbox, runner, clock, "m1", "Ciao")
    assert turn.texts == []
    assert sender.sent == [(SESSION, _handoff(clinic_pack, clinic_pack.default_language))]
    # A day later the channel answers again.
    clock.advance(days=1)
    await _say(inbox, runner, clock, "m2", "Ciao di nuovo")
    assert turn.texts == ["Ciao di nuovo"]


async def test_a_channel_without_limits_never_counts(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock, clinic_pack: BusinessPack
) -> None:
    turn = FakeTurn()
    runner = Runner(
        factory, pack=clinic_pack, channel=CHANNEL, sender=FakeSender(), turn=turn, now=clock
    )
    for n in range(3):
        await _say(inbox, runner, clock, f"m{n}", "Ciao")
    assert turn.texts == ["Ciao", "Ciao", "Ciao"]
    with business_session(factory, BUSINESS) as session:
        digest = session.scalar(select(ChannelConversation.last_digest))
    assert digest is None
