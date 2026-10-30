"""The inbox's schedules (M4 design §6, D8): the sweep gives back lost work and turns sends that
never came back into uncertain ones; the purge keeps what the design says, and no longer."""

import logging

import pytest
from sqlalchemy import Update, select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import Inbox, InStatus, OutStatus, conversation_key, queue_reply
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.channels.whatsapp import TEMPLATE
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import ChannelConversation, ChannelMessage
from tests.channels.fakes import NOW, Clock

pytestmark = pytest.mark.db
BUSINESS = "demo-clinic"
CHANNEL = "instagram"
ACCOUNT = "17841400000000001"
ALICE = "1111111111111111"
CODE = "demo-prova-1234"
KEY = conversation_key(BUSINESS, CHANNEL, ALICE)


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def inbox(factory: sessionmaker[Session], clock: Clock) -> Inbox:
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


def _say(inbox: Inbox, mid: str) -> None:
    inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.TEXT, mid, "Ciao", NOW)])


def _run(factory: sessionmaker[Session], statement: Update) -> None:
    with business_session(factory, BUSINESS) as session:
        session.execute(statement)


def _messages(factory: sessionmaker[Session], direction: str) -> list[ChannelMessage]:
    with business_session(factory, BUSINESS) as session:
        return list(
            session.scalars(select(ChannelMessage).where(ChannelMessage.direction == direction))
        )


def _conversation(factory: sessionmaker[Session]) -> ChannelConversation | None:
    with business_session(factory, BUSINESS) as session:
        return session.get(ChannelConversation, (BUSINESS, KEY))


def test_a_send_that_never_came_back_becomes_uncertain(
    inbox: Inbox,
    factory: sessionmaker[Session],
    clock: Clock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    replies = update(ChannelMessage).where(ChannelMessage.direction == "out")
    _run(factory, replies.values(status=OutStatus.SENDING, updated_at=NOW))
    clock.advance(minutes=3)
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.inbox"):
        swept = inbox.sweep()
    assert swept.uncertain == 1
    (welcome,) = _messages(factory, "out")
    assert (welcome.status, welcome.reason) == (OutStatus.UNCERTAIN, "no_answer")
    assert any(
        r.levelno == logging.ERROR and r.msg == "channel.send_uncertain" for r in caplog.records
    )


def test_work_that_lost_its_task_is_given_back(inbox: Inbox, clock: Clock) -> None:
    _say(inbox, "m1")
    clock.advance(minutes=3)
    assert inbox.sweep().work == (KEY,)


def test_fresh_work_is_left_to_its_task(inbox: Inbox, clock: Clock) -> None:
    _say(inbox, "m1")
    clock.advance(seconds=30)
    assert inbox.sweep().work == ()


def test_sent_texts_are_forgotten_after_a_day(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    _run(
        factory,
        update(ChannelMessage)
        .where(ChannelMessage.direction == "out")
        .values(status=OutStatus.SENT),
    )
    clock.advance(hours=25)
    inbox.purge()
    (welcome,) = _messages(factory, "out")
    assert welcome.text is None


def test_a_reply_still_queued_after_a_day_is_dropped(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    clock.advance(hours=25)
    inbox.purge()
    (welcome,) = _messages(factory, "out")
    assert (welcome.status, welcome.reason, welcome.text) == (OutStatus.DROPPED, "expired", None)


def test_a_message_waiting_for_a_day_loses_its_text(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    _say(inbox, "m1")
    clock.advance(hours=25)
    inbox.purge()
    (message,) = [m for m in _messages(factory, "in") if m.external_id == "m1"]
    assert (message.status, message.text, message.reason) == (InStatus.FAILED, None, "expired")


def test_message_rows_go_after_seven_days(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    clock.advance(days=8)
    assert inbox.purge().messages == 2  # the pairing code and the welcome
    assert _messages(factory, "in") == []
    assert _messages(factory, "out") == []


def test_a_quiet_conversation_forgets_its_state_then_goes(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    _run(factory, update(ChannelConversation).values(state={"conversation_id": KEY, "turns": 1}))
    clock.advance(days=8)
    inbox.purge()
    conversation = _conversation(factory)
    assert conversation is not None
    assert conversation.state is None
    clock.advance(days=23)
    assert inbox.purge().conversations == 1
    assert _conversation(factory) is None


def test_after_an_uncertain_part_the_rest_of_the_reply_is_never_sent(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    # Final review of M4, M5: the worker died while sending part 1 of 2; the sweep made part 1
    # uncertain and gave the conversation back, and part 2 went out alone.
    with business_session(factory, BUSINESS) as session:
        queue_reply(
            session,
            business_id=BUSINESS,
            channel=CHANNEL,
            conversation_id=KEY,
            parts=[("reply", "parte 1"), ("reply", "parte 2")],
            now=NOW,
        )
    part_one = (
        update(ChannelMessage)
        .where(ChannelMessage.direction == "out")
        .where(ChannelMessage.text == "parte 1")
    )
    _run(factory, part_one.values(status=OutStatus.SENDING, updated_at=NOW))
    clock.advance(minutes=3)
    inbox.sweep()
    statuses = {m.text: (m.status, m.reason) for m in _messages(factory, "out")}
    assert statuses["parte 1"] == (OutStatus.UNCERTAIN, "no_answer")
    assert statuses["parte 2"] == (OutStatus.DROPPED, "previous_uncertain")


def test_an_uncertain_part_leaves_the_templates_queued(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    # Final review of W1, I2: a template is a request of its own, not the rest of a reply.
    with business_session(factory, BUSINESS) as session:
        queue_reply(
            session,
            business_id=BUSINESS,
            channel=CHANNEL,
            conversation_id=KEY,
            parts=[("reply", "parte 1"), ("reply", "parte 2"), (TEMPLATE, "richiesta")],
            now=NOW,
        )
    part_one = (
        update(ChannelMessage)
        .where(ChannelMessage.direction == "out")
        .where(ChannelMessage.text == "parte 1")
    )
    _run(factory, part_one.values(status=OutStatus.SENDING, updated_at=NOW))
    clock.advance(minutes=3)
    assert inbox.sweep().work == (KEY,)
    statuses = {m.text: (m.status, m.reason) for m in _messages(factory, "out")}
    assert statuses["parte 2"] == (OutStatus.DROPPED, "previous_uncertain")
    assert statuses["richiesta"] == (OutStatus.QUEUED, None)
