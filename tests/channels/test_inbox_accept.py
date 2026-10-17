"""The inbox's webhook side (M4 design §4, D4, D6): a message stored once, testers only, the bot's
echoes told apart from the staff's."""

import logging
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import (
    PAUSE,
    TESTER_WELCOME,
    Inbox,
    InStatus,
    OutStatus,
    conversation_key,
    queue_reply,
    valid_key,
)
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import ChannelConversation, ChannelMessage

BUSINESS = "demo-clinic"
CHANNEL = "instagram"
ACCOUNT = "17841400000000001"
ALICE = "1111111111111111"
BOB = "2222222222222222"
CODE = "mcu-prova-1234"
NOW = datetime(2026, 10, 14, 9, tzinfo=UTC)
KEY = conversation_key(BUSINESS, CHANNEL, ALICE)
INBOX_LOGGER = "frontdesk.channels.inbox"


class Clock:
    def __init__(self) -> None:
        self.at = NOW

    def __call__(self) -> datetime:
        return self.at


def _text(text: str, mid: str, user: str = ALICE) -> Inbound:
    return Inbound(ACCOUNT, user, InboundKind.TEXT, mid, text, NOW)


def _echo(text: str | None, mid: str, user: str = ALICE) -> Inbound:
    return Inbound(ACCOUNT, user, InboundKind.ECHO, mid, text, NOW)


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
        account_id=ACCOUNT,
        testers_only=True,
        pairing_code=CODE,
        now=clock,
    )


def _messages(factory: sessionmaker[Session]) -> list[ChannelMessage]:
    with business_session(factory, BUSINESS) as session:
        return list(session.scalars(select(ChannelMessage).order_by(ChannelMessage.created_at)))


def _conversation(factory: sessionmaker[Session]) -> ChannelConversation | None:
    with business_session(factory, BUSINESS) as session:
        return session.get(ChannelConversation, (BUSINESS, KEY))


def _paused_until(factory: sessionmaker[Session]) -> datetime | None:
    conversation = _conversation(factory)
    assert conversation is not None
    return conversation.paused_until


def _paired(inbox: Inbox) -> None:
    assert inbox.accept([_text(CODE, "pair")]).work == (KEY,)


def _sent(
    factory: sessionmaker[Session],
    text: str,
    *,
    status: OutStatus,
    external_id: str | None,
    at: datetime,
) -> None:
    with business_session(factory, BUSINESS) as session:
        queue_reply(
            session,
            business_id=BUSINESS,
            channel=CHANNEL,
            conversation_id=KEY,
            parts=[("reply", text)],
            now=at,
        )
        session.flush()
        session.execute(
            update(ChannelMessage)
            .where(ChannelMessage.text == text)
            .values(status=status, external_id=external_id, updated_at=at)
        )


def _fields(caplog: pytest.LogCaptureFixture, event: str) -> list[object]:
    return [getattr(record, "fields", None) for record in caplog.records if record.msg == event]


def test_the_conversation_key_is_stable_and_says_nothing() -> None:
    assert conversation_key(BUSINESS, CHANNEL, ALICE) == KEY
    assert conversation_key(BUSINESS, CHANNEL, BOB) != KEY
    assert KEY.startswith("instagram-")
    assert len(KEY) == 50
    assert ALICE not in KEY
    assert valid_key(KEY)
    assert not valid_key("instagram-xyz")
    assert not valid_key("../tasks/sweep")


@pytest.mark.db
def test_a_message_of_someone_who_is_not_a_tester_is_not_stored(
    inbox: Inbox, factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=INBOX_LOGGER):
        assert inbox.accept([_text("Siete aperti?", "m1")]).work == ()
    assert _messages(factory) == []
    assert _conversation(factory) is None
    assert _fields(caplog, "channel.not_a_tester") == [{"channel": CHANNEL, "count": 1}]


@pytest.mark.db
def test_the_pairing_code_makes_a_tester_and_queues_the_welcome(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    assert inbox.accept([_text(f"  {CODE} ", "pair")]).work == (KEY,)
    assert inbox.count_testers() == 1
    stored = [(m.direction, m.status, m.text) for m in _messages(factory)]
    assert stored == [("in", InStatus.READ, None), ("out", OutStatus.QUEUED, TESTER_WELCOME)]


@pytest.mark.db
def test_the_pairing_code_delivered_again_is_not_a_turn(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    assert inbox.accept([_text(CODE, "pair")]).work == ()
    assert len(_messages(factory)) == 2


@pytest.mark.db
def test_a_tester_who_sends_the_code_again_sends_nothing_to_the_model(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    # Final review of M4, M7: the code is a secret; once paired it was an ordinary text.
    _paired(inbox)
    assert inbox.accept([_text(f" {CODE}", "again")]).work == ()
    again = [m for m in _messages(factory) if m.external_id == "again"]
    assert [(m.status, m.text) for m in again] == [(InStatus.READ, None)]


@pytest.mark.db
def test_a_tester_message_is_stored_once_and_gives_work_until_read(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    assert inbox.accept([_text("Siete aperti sabato?", "m1")]).work == (KEY,)
    assert inbox.accept([_text("Siete aperti sabato?", "m1")]).work == (KEY,)  # Meta retried
    waiting = [m.text for m in _messages(factory) if m.status == InStatus.PENDING]
    assert waiting == ["Siete aperti sabato?"]


@pytest.mark.db
def test_a_redelivered_message_after_its_turn_gives_no_work(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    inbox.accept([_text("Siete aperti sabato?", "m1")])
    with business_session(factory, BUSINESS) as session:
        session.execute(
            update(ChannelMessage)
            .where(ChannelMessage.external_id == "m1")
            .values(status=InStatus.READ, text=None)
        )
    assert inbox.accept([_text("Siete aperti sabato?", "m1")]).work == ()


@pytest.mark.db
def test_live_mode_stores_everyone(factory: sessionmaker[Session], clock: Clock) -> None:
    live = Inbox(
        factory,
        business_id=BUSINESS,
        channel=CHANNEL,
        account_id=ACCOUNT,
        testers_only=False,
        pairing_code=None,
        now=clock,
    )
    work = live.accept([_text("Siete aperti?", "m1", user=BOB)]).work
    assert work == (conversation_key(BUSINESS, CHANNEL, BOB),)


def _staff_inbox(factory: sessionmaker[Session], clock: Clock) -> Inbox:
    """WhatsApp's inbox (W1): only the numbers of the staff list are let in."""
    return Inbox(
        factory,
        business_id=BUSINESS,
        channel="whatsapp",
        account_id=ACCOUNT,
        testers_only=False,
        pairing_code=None,
        now=clock,
        allowed_users=frozenset({ALICE}),
    )


@pytest.mark.db
def test_a_number_off_the_list_is_dropped_before_storage(
    factory: sessionmaker[Session], clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    tap = Inbound(ACCOUNT, BOB, InboundKind.BUTTON, "m2", "request:x:confirm", NOW)
    with caplog.at_level(logging.INFO, logger=INBOX_LOGGER):
        assert _staff_inbox(factory, clock).accept([_text("Ciao", "m1", user=BOB), tap]).work == ()
    assert _messages(factory) == []
    with business_session(factory, BUSINESS) as session:
        assert list(session.scalars(select(ChannelConversation))) == []
    assert _fields(caplog, "channel.not_allowed") == [{"channel": "whatsapp", "count": 2}]


@pytest.mark.db
def test_a_staff_member_s_message_is_stored(factory: sessionmaker[Session], clock: Clock) -> None:
    work = _staff_inbox(factory, clock).accept([_text("Ciao", "m1")]).work
    assert work == (conversation_key(BUSINESS, "whatsapp", ALICE),)
    assert [message.text for message in _messages(factory)] == ["Ciao"]


@pytest.mark.db
def test_an_echo_of_the_bots_reply_does_not_pause(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    _sent(factory, "Siamo aperti sabato.", status=OutStatus.SENT, external_id="out-9", at=NOW)
    inbox.accept([_echo("Siamo aperti sabato.", "out-9")])
    assert _paused_until(factory) is None


@pytest.mark.db
def test_an_echo_with_other_whitespace_is_the_bots(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    _sent(factory, "Siamo aperti\nsabato.", status=OutStatus.SENDING, external_id=None, at=NOW)
    inbox.accept([_echo(" Siamo aperti sabato. ", "out-10")])  # before the send returned
    assert _paused_until(factory) is None


@pytest.mark.db
def test_an_echo_of_the_staff_pauses_the_conversation(
    inbox: Inbox, factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    _paired(inbox)
    with caplog.at_level(logging.INFO, logger=INBOX_LOGGER):
        inbox.accept([_echo("Buongiorno, sono Anna della reception", "staff-1")])
    assert _paused_until(factory) == NOW + PAUSE
    assert _fields(caplog, "channel.staff_reply") == [{"channel": CHANNEL, "conversation_id": KEY}]


@pytest.mark.db
def test_an_old_text_of_the_bot_does_not_hide_the_staff(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    _sent(
        factory, "Ciao", status=OutStatus.SENT, external_id="out-1", at=NOW - timedelta(minutes=5)
    )
    inbox.accept([_echo("Ciao", "staff-2")])
    assert _paused_until(factory) == NOW + PAUSE


@pytest.mark.db
def test_messages_during_the_pause_are_not_stored_and_extend_it(
    inbox: Inbox, factory: sessionmaker[Session], clock: Clock
) -> None:
    _paired(inbox)
    inbox.accept([_echo("Buongiorno, sono Anna", "staff-1")])
    clock.at = NOW + timedelta(hours=1)
    assert inbox.accept([_text("Grazie Anna", "m2")]).work == ()
    assert [m.external_id for m in _messages(factory) if m.direction == "in"] == ["pair"]
    assert _paused_until(factory) == NOW + timedelta(hours=1) + PAUSE


@pytest.mark.db
def test_an_echo_in_a_thread_the_bot_is_not_in_is_ignored(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    inbox.accept([_echo("Buongiorno", "staff-3", user=BOB)])
    assert _messages(factory) == []


@pytest.mark.db
def test_a_message_unsent_before_its_turn_is_discarded(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    inbox.accept([_text("Numero sbagliato", "m3")])
    inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.DELETED, "m3", None, NOW)])
    (unsent,) = [m for m in _messages(factory) if m.external_id == "m3"]
    assert (unsent.status, unsent.text) == (InStatus.DISCARDED, None)


@pytest.mark.db
def test_events_for_another_account_are_logged_with_its_id(
    inbox: Inbox, factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=INBOX_LOGGER):
        work = inbox.accept([Inbound("999", ALICE, InboundKind.TEXT, "m1", CODE, NOW)]).work
    assert work == ()
    assert _messages(factory) == []
    assert _fields(caplog, "channel.other_account") == [{"channel": CHANNEL, "account_id": "999"}]
    # Final review of M4, M4: with one account per app, another account is a misconfiguration.
    assert all(
        r.levelno == logging.ERROR for r in caplog.records if r.msg == "channel.other_account"
    )


@pytest.mark.db
def test_reactions_and_reads_are_ignored(inbox: Inbox, factory: sessionmaker[Session]) -> None:
    _paired(inbox)
    assert inbox.accept([Inbound(ACCOUNT, ALICE, InboundKind.IGNORED, "m1", None, NOW)]).work == ()
    assert len(_messages(factory)) == 2  # the pairing code and the welcome


@pytest.mark.db
def test_testers_can_be_counted_and_cleared(inbox: Inbox) -> None:
    _paired(inbox)
    assert inbox.count_testers() == 1
    assert inbox.clear_testers() == 1
    assert inbox.count_testers() == 0


@pytest.mark.db
def test_the_echo_of_an_uncertain_send_is_the_bots(
    inbox: Inbox, factory: sessionmaker[Session]
) -> None:
    _paired(inbox)
    # The Send API timed out, but Meta delivered the message: its echo brings an ID never seen.
    _sent(factory, "Siamo aperti sabato.", status=OutStatus.UNCERTAIN, external_id=None, at=NOW)
    inbox.accept([_echo("Siamo aperti sabato.", "late-mid")])
    assert _paused_until(factory) is None
