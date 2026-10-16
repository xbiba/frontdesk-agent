"""Booking holds (migration 0005): no two patients on one time, personal data only while held."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.holds import (
    HOLD_LIFETIME,
    attach,
    forget,
    held,
    hold,
    mark_booked,
    purge,
    release,
)
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import BookingHold

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)
CLINIC = "clinica-a"
NINE = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
TEN = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 23, hour, minute, tzinfo=UTC)


def _hold(
    factory: sessionmaker[Session],
    *,
    business: str = CLINIC,
    conversation: str = "c1",
    doctor: str = "dr-a",
    start: datetime = NINE,
    end: datetime = TEN,
    now: datetime = NOW,
) -> uuid.UUID | None:
    with business_session(factory, business) as session:
        return hold(
            session,
            business_id=business,
            conversation_id=conversation,
            doctor_id=doctor,
            service_id="osteopathy",
            start=start,
            end=end,
            now=now,
        )


def _rows(factory: sessionmaker[Session], business: str = CLINIC) -> list[BookingHold]:
    with business_session(factory, business) as session:
        return list(session.scalars(select(BookingHold)))


def test_overlapping_holds_of_one_doctor_conflict_and_other_doctors_do_not(
    factory: sessionmaker[Session],
) -> None:
    assert _hold(factory) is not None
    assert _hold(factory, conversation="c2", start=at(9, 30), end=at(10, 30)) is None
    assert _hold(factory, conversation="c3", doctor="dr-b") is not None
    assert _hold(factory, conversation="c4", start=at(10), end=at(11)) is not None  # adjacent


def test_a_new_hold_of_a_conversation_replaces_its_earlier_one(
    factory: sessionmaker[Session],
) -> None:
    first = _hold(factory)
    second = _hold(factory, start=at(11), end=at(12))
    assert first is not None
    assert second is not None
    assert [row.id for row in _rows(factory)] == [second]


def test_an_expired_hold_frees_its_time_and_its_data_goes_with_it(
    factory: sessionmaker[Session],
) -> None:
    first = _hold(factory)
    assert first is not None
    with business_session(factory, CLINIC) as session:
        attach(session, first, name="Mario Rossi", phone="+355691234567")
        assert held(session, first, NOW + HOLD_LIFETIME) is not None
        assert held(session, first, NOW + HOLD_LIFETIME + timedelta(seconds=1)) is None
    later = NOW + HOLD_LIFETIME + timedelta(minutes=1)
    second = _hold(factory, conversation="c2", now=later)
    assert second is not None
    assert [(row.id, row.name, row.phone) for row in _rows(factory)] == [(second, None, None)]


def test_booking_keeps_the_time_and_drops_the_name_and_the_phone(
    factory: sessionmaker[Session],
) -> None:
    hold_id = _hold(factory)
    assert hold_id is not None
    with business_session(factory, CLINIC) as session:
        attach(session, hold_id, name="Mario Rossi")
        attach(session, hold_id, phone="+355691234567")
        current = held(session, hold_id, NOW)
        assert current is not None
        assert (current.name, current.phone, current.start, current.end) == (
            "Mario Rossi",
            "+355691234567",
            at(9),
            at(10),
        )
        mark_booked(session, hold_id, "event-1")
    (row,) = _rows(factory)
    assert (row.status, row.name, row.phone, row.event_id) == ("booked", None, None, "event-1")
    assert row.expires_at == at(10)  # kept until the slot has passed
    assert _hold(factory, conversation="c2") is None  # the booked time stays taken


def test_a_pending_booking_keeps_the_name_for_the_staff_and_never_the_phone(
    factory: sessionmaker[Session],
) -> None:
    # WhatsApp W1: the reception's request names the patient, never their phone (Albi, 2026-10-16).
    hold_id = _hold(factory)
    assert hold_id is not None
    with business_session(factory, CLINIC) as session:
        attach(session, hold_id, name="Mario Rossi", phone="+355691234567")
        mark_booked(session, hold_id, "event-1", pending=True, language="it")
    (row,) = _rows(factory)
    assert (row.status, row.request_status, row.name, row.phone, row.language) == (
        "booked",
        "pending",
        "Mario Rossi",
        None,
        "it",
    )
    refused = pytest.raises(IntegrityError, match="ck_booking_holds_no_personal_data_once_booked")
    with refused, business_session(factory, CLINIC) as session:
        session.execute(update(BookingHold).values(phone="+355691234567"))


def test_an_accepted_booking_keeps_neither_name_nor_phone(factory: sessionmaker[Session]) -> None:
    hold_id = _hold(factory)
    assert hold_id is not None
    with business_session(factory, CLINIC) as session:
        attach(session, hold_id, name="Mario Rossi", phone="+355691234567")
        mark_booked(session, hold_id, "event-1", language="it")
    (row,) = _rows(factory)
    assert (row.name, row.phone, row.request_status) == (None, None, None)


def test_a_booked_hold_cannot_take_personal_data_back(factory: sessionmaker[Session]) -> None:
    hold_id = _hold(factory)
    assert hold_id is not None
    with business_session(factory, CLINIC) as session:
        mark_booked(session, hold_id, "event-1")
    refused = pytest.raises(IntegrityError, match="ck_booking_holds_no_personal_data_once_booked")
    with refused, business_session(factory, CLINIC) as session:
        session.execute(update(BookingHold).values(name="Mario Rossi"))


def test_release_and_purge(factory: sessionmaker[Session]) -> None:
    released = _hold(factory)
    assert released is not None
    with business_session(factory, CLINIC) as session:
        release(session, released)
    assert _rows(factory) == []
    expired = _hold(factory, conversation="c1")
    booked = _hold(factory, conversation="c2", start=at(11), end=at(12))
    assert expired is not None
    assert booked is not None
    with business_session(factory, CLINIC) as session:
        mark_booked(session, booked, "event-2")
    with business_session(factory, CLINIC) as session:
        assert purge(session, NOW + HOLD_LIFETIME + timedelta(minutes=1)) == 1  # the expired hold
        assert purge(session, at(12, 1)) == 1  # the booking, once its slot has passed


def test_another_business_neither_sees_nor_blocks_the_holds(
    factory: sessionmaker[Session],
) -> None:
    hold_id = _hold(factory)
    assert hold_id is not None
    with business_session(factory, "clinica-b") as session:
        assert held(session, hold_id, NOW) is None
    assert _rows(factory, "clinica-b") == []
    assert _hold(factory, business="clinica-b") is not None  # its own doctor "dr-a"


def test_parallel_holds_on_one_time_leave_exactly_one(factory: sessionmaker[Session]) -> None:
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda n: _hold(factory, conversation=f"c{n}"), range(6)))
    assert sum(result is not None for result in results) == 1


def test_forget_deletes_every_hold_of_one_conversation(factory: sessionmaker[Session]) -> None:
    booked = _hold(factory, conversation="case-1")
    assert booked is not None
    with business_session(factory, CLINIC) as session:
        mark_booked(session, booked, "event-1")
    assert _hold(factory, conversation="case-2", start=at(11), end=at(12)) is not None
    with business_session(factory, CLINIC) as session:
        assert forget(session, "case-1") == 1
    assert [row.conversation_id for row in _rows(factory)] == ["case-2"]
