"""The channel tables (migration 0007): a business sees and writes only its own rows, a blank
business is refused, a channel's message ID is stored once, and messages go with their
conversation."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import CursorResult, create_engine, func, select, text, update
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import BookingHold, ChannelConversation, ChannelMessage, ChannelTester

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 14, 9, tzinfo=UTC)
KEY = "instagram-" + "a" * 40


def _conversation(business: str) -> ChannelConversation:
    return ChannelConversation(
        business_id=business,
        conversation_id=KEY,
        channel="instagram",
        user_id="u1",
        last_message_at=NOW,
    )


def _message(business: str, external_id: str | None, direction: str = "in") -> ChannelMessage:
    return ChannelMessage(
        id=uuid.uuid4(),
        business_id=business,
        conversation_id=KEY,
        channel="instagram",
        direction=direction,
        kind="text",
        external_id=external_id,
        text="ciao",
        status="pending",
        attempts=0,
        created_at=NOW,
        updated_at=NOW,
    )


def _write(factory: sessionmaker[Session], scope: str, *rows: object) -> None:
    """Rows written in one transaction scoped to `scope`, each flushed at once."""
    with business_session(factory, scope) as session:
        for row in rows:
            session.add(row)
            session.flush()


def test_a_business_sees_only_its_own_channel_rows(app_url: str) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, "alpha") as session:
        session.add(_conversation("alpha"))
        session.flush()
        session.add(_message("alpha", "m1"))
        session.add(
            ChannelTester(business_id="alpha", channel="instagram", user_id="u1", created_at=NOW)
        )
    with business_session(factory, "beta") as session:
        assert session.scalars(select(ChannelConversation)).all() == []
        assert session.scalars(select(ChannelMessage)).all() == []
        assert session.scalars(select(ChannelTester)).all() == []


def test_the_new_columns_are_bound_by_row_level_security(app_url: str) -> None:
    # Migration 0008: a booking request's state (WhatsApp W1) lives on booking_holds, under its
    # policy: another business neither reads it nor decides it.
    factory = make_session_factory(make_engine(app_url))
    request = BookingHold(
        id=uuid.uuid4(),
        business_id="alpha",
        conversation_id="c1",
        doctor_id="dr-a",
        service_id="osteopathy",
        during=Range(NOW, NOW + timedelta(hours=1)),
        status="booked",
        expires_at=NOW + timedelta(hours=1),
        name="Mario Rossi",
        request_status="pending",
        language="it",
        created_at=NOW,
    )
    _write(factory, "alpha", request)
    with business_session(factory, "beta") as session:
        assert session.scalars(select(BookingHold.request_status)).all() == []
        decided = session.execute(
            update(BookingHold).values(request_status="confirmed", decided_by="Ana")
        )
        assert cast(CursorResult[Any], decided).rowcount == 0
    with business_session(factory, "alpha") as session:
        assert session.scalars(select(BookingHold.request_status)).all() == ["pending"]


def test_a_business_cannot_write_rows_of_another(app_url: str) -> None:
    factory = make_session_factory(make_engine(app_url))
    with pytest.raises(DBAPIError):
        _write(factory, "beta", _conversation("alpha"))


def test_a_message_id_is_stored_once(app_url: str) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, "alpha") as session:
        session.add(_conversation("alpha"))
        session.flush()
        session.add(_message("alpha", "m1"))
    with pytest.raises(IntegrityError):
        _write(factory, "alpha", _message("alpha", "m1"))


def test_replies_without_an_id_yet_do_not_collide(app_url: str) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, "alpha") as session:
        session.add(_conversation("alpha"))
        session.flush()
        session.add_all([_message("alpha", None, "out"), _message("alpha", None, "out")])
    with business_session(factory, "alpha") as session:
        assert session.scalar(select(func.count()).select_from(ChannelMessage)) == 2


def test_messages_go_with_their_conversation(app_url: str) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, "alpha") as session:
        session.add(_conversation("alpha"))
        session.flush()
        session.add(_message("alpha", "m1"))
    with business_session(factory, "alpha") as session:
        session.execute(text("DELETE FROM channel_conversations"))
    with business_session(factory, "alpha") as session:
        assert session.scalar(select(func.count()).select_from(ChannelMessage)) == 0


@pytest.mark.parametrize(
    ("table", "statement"),
    [
        (
            "channel_conversations",
            "INSERT INTO channel_conversations (business_id, conversation_id, channel, user_id, "
            "last_message_at) VALUES ('', 'k', 'instagram', 'u', now())",
        ),
        (
            "channel_messages",
            "INSERT INTO channel_messages (id, business_id, conversation_id, channel, direction, "
            "kind, status, created_at, updated_at) VALUES (gen_random_uuid(), '', 'k', "
            "'instagram', 'in', 'text', 'pending', now(), now())",
        ),
        (
            "channel_testers",
            "INSERT INTO channel_testers (business_id, channel, user_id, created_at) "
            "VALUES ('', 'instagram', 'u', now())",
        ),
    ],
)
def test_no_channel_table_stores_a_blank_business_id(
    migrated_url: str, table: str, statement: str
) -> None:
    engine = create_engine(migrated_url)  # the owner: a superuser, so only the constraint decides
    with (
        pytest.raises(IntegrityError, match=f"ck_{table}_business_id_not_blank"),
        engine.begin() as connection,
    ):
        connection.execute(text(statement))
    engine.dispose()
