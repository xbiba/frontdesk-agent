"""Booking holds (M5, migration 0005): a slot taken while a patient books it, then booked.

Two patients never get the same time: the database refuses overlapping holds of one doctor, even in
parallel. Name and phone live on a hold only while it is held, at most HOLD_LIFETIME: booking or
cancelling drops them, and an expired hold is deleted by the next hold and by `purge`. A booked hold
keeps the doctor, the time and the calendar's event id, until its slot has passed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, cast

from sqlalchemy import CursorResult, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from frontdesk.db.models import BookingHold

HOLD_LIFETIME: Final = timedelta(minutes=15)
_EXCLUSION_VIOLATION: Final = "23P01"


@dataclass(frozen=True)
class Hold:
    id: uuid.UUID
    doctor_id: str
    service_id: str
    start: datetime
    end: datetime
    name: str | None
    phone: str | None


def hold(
    session: Session,
    *,
    business_id: str,
    conversation_id: str,
    doctor_id: str,
    service_id: str,
    start: datetime,
    end: datetime,
    now: datetime,
) -> uuid.UUID | None:
    """A new hold on the slot, or None when it overlaps another of the doctor's. The conversation's
    earlier hold and every expired hold are dropped first."""
    session.execute(
        delete(BookingHold).where(
            BookingHold.status == "held",
            or_(BookingHold.expires_at < now, BookingHold.conversation_id == conversation_id),
        )
    )
    hold_id = uuid.uuid4()
    row = BookingHold(
        id=hold_id,
        business_id=business_id,
        conversation_id=conversation_id,
        doctor_id=doctor_id,
        service_id=service_id,
        during=Range(start, end),
        status="held",
        expires_at=now + HOLD_LIFETIME,
        created_at=now,
    )
    try:
        with session.begin_nested():
            session.add(row)
    except IntegrityError as error:
        if getattr(error.orig, "sqlstate", None) != _EXCLUSION_VIOLATION:
            raise
        return None
    return hold_id


def attach(
    session: Session, hold_id: uuid.UUID, *, name: str | None = None, phone: str | None = None
) -> None:
    values = {key: value for key, value in (("name", name), ("phone", phone)) if value is not None}
    if values:
        session.execute(
            update(BookingHold)
            .where(BookingHold.id == hold_id, BookingHold.status == "held")
            .values(**values)
        )


def held(session: Session, hold_id: uuid.UUID, now: datetime) -> Hold | None:
    """The hold while it is held and not expired."""
    row = session.scalars(
        select(BookingHold).where(
            BookingHold.id == hold_id,
            BookingHold.status == "held",
            BookingHold.expires_at >= now,
        )
    ).one_or_none()
    if row is None or row.during.lower is None or row.during.upper is None:
        return None
    start, end = row.during.lower, row.during.upper
    return Hold(row.id, row.doctor_id, row.service_id, start, end, row.name, row.phone)


def mark_booked(session: Session, hold_id: uuid.UUID, event_id: str) -> None:
    """The appointment exists in the calendar: the hold keeps the time, drops name and phone."""
    session.execute(
        update(BookingHold)
        .where(BookingHold.id == hold_id, BookingHold.status == "held")
        .values(
            status="booked",
            name=None,
            phone=None,
            event_id=event_id,
            expires_at=func.upper(BookingHold.during),
        )
    )


def booked_events(session: Session, conversation_id: str, now: datetime) -> list[str]:
    """The calendar's references of the conversation's appointments still to come."""
    return [
        event_id
        for event_id in session.scalars(
            select(BookingHold.event_id).where(
                BookingHold.conversation_id == conversation_id,
                BookingHold.status == "booked",
                BookingHold.expires_at > now,
            )
        )
        if event_id is not None
    ]


def release(session: Session, hold_id: uuid.UUID) -> None:
    session.execute(
        delete(BookingHold).where(BookingHold.id == hold_id, BookingHold.status == "held")
    )


def purge(session: Session, now: datetime) -> int:
    """Deletes the expired holds and the booked ones whose slot has passed; how many."""
    result = session.execute(delete(BookingHold).where(BookingHold.expires_at < now))
    return cast(CursorResult[Any], result).rowcount


def forget(session: Session, conversation_id: str) -> int:
    """Deletes every hold of one conversation, booked or not: an evaluation case leaves nothing
    behind for the next one or the next run."""
    result = session.execute(
        delete(BookingHold).where(BookingHold.conversation_id == conversation_id)
    )
    return cast(CursorResult[Any], result).rowcount
