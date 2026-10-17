"""The booking operations the reception agent's tools call (M5; the agent since 2026-10-12).

Code decides every one: a hold on a slot that was shown and is still free; the booking of the
held slot, once the patient said yes to its summary, with the calendar read again first; the
release of the hold. A slot taken meanwhile or an expired hold is said so, and the agent offers
other times (frontdesk.conversation.reception_booking); never a booking that is not there. Name and phone live on the hold only (frontdesk.booking.holds).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.flow import phrase
from frontdesk.booking.holds import held, hold, mark_booked, release
from frontdesk.booking.reply import slot_time
from frontdesk.booking.scheduler import ACCEPTED, PENDING, Scheduler
from frontdesk.conversation.state import BookingStage, BookingStep
from frontdesk.db.engine import business_session
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, Language


@dataclass(frozen=True)
class StepReply:
    text: str
    outcome: str  # "asked" while the booking goes on; "replied" when booked or cancelled; "handoff"
    next: BookingStep | None
    doctor_id: str
    booked: bool = False  # the appointment is in the calendar


@dataclass(frozen=True)
class BookingContext:
    pack: BusinessPack
    catalog: Catalog
    scheduler: Scheduler
    session_factory: sessionmaker[Session]
    conversation_id: str
    language: Language
    now: datetime


def take(context: BookingContext, step: BookingStep, start: datetime) -> StepReply:
    """A hold on the slot picked, then the name; fresh slots when it was taken meanwhile.
    Raises CalendarUnavailableError."""
    end = start + timedelta(minutes=_duration(context, step.service_id))
    if not _free(context, step, start):
        return StepReply(_say(context, "taken"), "replied", None, step.doctor_id)
    with business_session(context.session_factory, context.pack.id) as session:
        hold_id = hold(
            session,
            business_id=context.pack.id,
            conversation_id=context.conversation_id,
            doctor_id=step.doctor_id,
            service_id=step.service_id,
            start=start,
            end=end,
            now=context.now,
        )
    if hold_id is None:
        return StepReply(_say(context, "taken"), "replied", None, step.doctor_id)
    after = step.model_copy(update={"stage": BookingStage.NAME, "hold_id": hold_id})
    return StepReply(_say(context, "ask_name"), "asked", after, step.doctor_id)


def confirm(context: BookingContext, step: BookingStep) -> StepReply:
    """The patient said yes: the held slot is booked, once the calendar shows it still free.
    Raises CalendarUnavailableError."""
    with business_session(context.session_factory, context.pack.id) as session:
        current = held(session, _hold_id(step), context.now)
    if current is None or current.name is None or current.phone is None:
        return StepReply(_say(context, "expired"), "replied", None, step.doctor_id)
    if not _free(context, step, current.start):
        _release(context, step)
        return StepReply(_say(context, "taken"), "replied", None, step.doctor_id)
    event_id = current.id.hex
    try:
        status = context.scheduler.book(
            context.pack,
            step.doctor_id,
            step.service_id,
            current.start,
            current.end,
            name=current.name,
            phone=current.phone,
            reference=event_id,
            language=context.language,
        )
    except CalendarUnavailableError:
        _release(context, step)
        return _handoff(context, step.doctor_id, _say(context, "failed"))
    with business_session(context.session_factory, context.pack.id) as session:
        mark_booked(
            session,
            current.id,
            event_id,
            pending=status == PENDING,
            language=context.language,
        )
    booked = phrase(
        "booked" if status == ACCEPTED else "requested",  # the clinic confirms a pending one
        context.language,
        when=slot_time(current.start.astimezone(ZoneInfo(context.pack.timezone)), context.language),
        service=_service(context, step.service_id),
        doctor=_doctor(context, step.doctor_id),
        contact=context.pack.location.phone,
    )
    privacy = context.pack.fixed_texts[context.language][FixedTextKey.BOOKING_PRIVACY]
    return StepReply(_joined(booked, privacy), "replied", None, step.doctor_id, booked=True)


def _free(context: BookingContext, step: BookingStep, start: datetime) -> bool:
    """The slot is still among the scheduler's free ones."""
    every = context.scheduler.free_slots(context.pack, step.doctor_id, step.service_id, context.now)
    return any(slot.start == start for slot in every)


def cancel(context: BookingContext, step: BookingStep) -> StepReply:
    _release(context, step)
    return StepReply(_say(context, "cancelled"), "replied", None, step.doctor_id)


def _release(context: BookingContext, step: BookingStep) -> None:
    if step.hold_id is not None:
        with business_session(context.session_factory, context.pack.id) as session:
            release(session, step.hold_id)


def _handoff(context: BookingContext, doctor_id: str, prefix: str | None = None) -> StepReply:
    """How to book through reception: the pack's next step, which gives the contact."""
    texts = context.pack.fixed_texts[context.language]
    how = texts.get(FixedTextKey.NEXT_STEP) or texts[FixedTextKey.HANDOFF]
    return StepReply(_joined(prefix, how), "handoff", None, doctor_id)


def _say(context: BookingContext, key: str) -> str:
    return phrase(key, context.language)


def _joined(*parts: str | None) -> str:
    return "\n\n".join(part for part in parts if part)


def _hold_id(step: BookingStep) -> uuid.UUID:
    if step.hold_id is None:  # the name, phone and confirm stages follow a hold
        raise ValueError(f"booking stage '{step.stage}' without a hold")
    return step.hold_id


def _doctor(context: BookingContext, doctor_id: str) -> str:
    return context.catalog.doctors.get(doctor_id, doctor_id)


def _service(context: BookingContext, service_id: str, language: str | None = None) -> str:
    names = context.catalog.services.get(service_id, {})
    return names.get(language or context.language, service_id)


def _duration(context: BookingContext, service_id: str) -> int:
    return next(item.duration_minutes for item in context.pack.services if item.id == service_id)
