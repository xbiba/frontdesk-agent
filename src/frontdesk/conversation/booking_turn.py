"""A booking from the chat, step by step (M5). Code reads every answer; the model is never asked.

The slots are offered numbered; a number (or a time) takes a hold on the slot; the name and the phone
go onto the hold; a summary asks for a yes; the yes writes the appointment in the doctor's calendar,
which is read again first. A slot taken meanwhile, an expired hold or a calendar that cannot be read
each end in fresh slots or in how to book through reception, never in a booking that is not there.
Name and phone live on the hold only (frontdesk.booking.holds); this module keeps neither.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import Calendar, CalendarUnavailableError
from frontdesk.booking.flow import (
    entry_description,
    parse_choice,
    phrase,
    still_free,
    valid_name,
    valid_phone,
    yes_or_no,
)
from frontdesk.booking.holds import attach, held, hold, mark_booked, release
from frontdesk.booking.reply import compose_slots, find_slots, read_calendar, slot_time
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


@dataclass(frozen=True)
class BookingContext:
    pack: BusinessPack
    catalog: Catalog
    calendar: Calendar
    session_factory: sessionmaker[Session]
    conversation_id: str
    language: Language
    now: datetime


def offer(
    context: BookingContext, doctor_id: str, service_id: str, *, prefix: str | None = None
) -> StepReply:
    """The first free slots, numbered, and how to pick one; how to book when none is free.
    Raises CalendarUnavailableError."""
    slots = find_slots(context.calendar, context.pack, doctor_id, service_id, context.now)
    texts = context.pack.fixed_texts[context.language]
    if not slots:
        none = texts.get(FixedTextKey.NO_SLOTS) or texts[FixedTextKey.NO_INFORMATION]
        return _handoff(context, doctor_id, _joined(prefix, none))
    listed = compose_slots(
        slots,
        doctor=_doctor(context, doctor_id),
        service=_service(context, service_id),
        minutes=_duration(context, service_id),
        language=context.language,
        numbered=True,
    )
    step = BookingStep(
        stage=BookingStage.CHOOSE,
        doctor_id=doctor_id,
        service_id=service_id,
        offered=tuple(slot.start for slot in slots),
    )
    text = _joined(prefix, listed, phrase("choose", context.language))
    return StepReply(text, "asked", step, doctor_id)


def answer(context: BookingContext, step: BookingStep, message: str) -> StepReply | None:
    """The next step, or None when the message is not an answer to the slots offered: the normal
    pipeline then takes it, and the booking ends."""
    try:
        if step.stage is BookingStage.CHOOSE:
            return _choose(context, step, message)
        if step.stage is BookingStage.NAME:
            return _name(context, step, message)
        if step.stage is BookingStage.PHONE:
            return _phone(context, step, message)
        return _confirm(context, step, message)
    except CalendarUnavailableError:
        _release(context, step)
        return _handoff(context, step.doctor_id)


def _choose(context: BookingContext, step: BookingStep, message: str) -> StepReply | None:
    start = parse_choice(message, step.offered, ZoneInfo(context.pack.timezone))
    if start is None:
        return None
    end = start + timedelta(minutes=_duration(context, step.service_id))
    if not _free(context, step, start):
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "taken"))
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
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "taken"))
    privacy = context.pack.fixed_texts[context.language][FixedTextKey.BOOKING_PRIVACY]
    after = step.model_copy(update={"stage": BookingStage.NAME, "hold_id": hold_id, "retries": 0})
    return StepReply(_joined(privacy, _say(context, "ask_name")), "asked", after, step.doctor_id)


def _name(context: BookingContext, step: BookingStep, message: str) -> StepReply:
    if yes_or_no(message) is False:
        return _cancel(context, step)
    name = valid_name(message)
    with business_session(context.session_factory, context.pack.id) as session:
        current = held(session, _hold_id(step), context.now)
        if current is not None and name is not None:
            attach(session, current.id, name=name)
    if current is None:
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "expired"))
    if name is None:
        return _again(context, step, "again_name")
    after = step.model_copy(update={"stage": BookingStage.PHONE, "retries": 0})
    return StepReply(_say(context, "ask_phone"), "asked", after, step.doctor_id)


def _phone(context: BookingContext, step: BookingStep, message: str) -> StepReply:
    if yes_or_no(message) is False:
        return _cancel(context, step)
    phone = valid_phone(message)
    with business_session(context.session_factory, context.pack.id) as session:
        current = held(session, _hold_id(step), context.now)
        if current is not None and phone is not None:
            attach(session, current.id, phone=phone)
    if current is None:
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "expired"))
    if phone is None:
        return _again(context, step, "again_phone")
    summary = phrase(
        "summary",
        context.language,
        when=slot_time(current.start.astimezone(ZoneInfo(context.pack.timezone)), context.language),
        service=_service(context, step.service_id),
        doctor=_doctor(context, step.doctor_id),
        name=current.name or "",
        phone=phone,
    )
    after = step.model_copy(update={"stage": BookingStage.CONFIRM, "retries": 0})
    return StepReply(summary, "asked", after, step.doctor_id)


def _confirm(context: BookingContext, step: BookingStep, message: str) -> StepReply:
    decision = yes_or_no(message)
    if decision is None:
        return _again(context, step, "again_yes_no")
    if decision is False:
        return _cancel(context, step)
    with business_session(context.session_factory, context.pack.id) as session:
        current = held(session, _hold_id(step), context.now)
    if current is None or current.name is None or current.phone is None:
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "expired"))
    if not _free(context, step, current.start):
        _release(context, step)
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "taken"))
    calendar = context.pack.integrations.calendar
    if calendar is None:  # a step exists only for a pack with a calendar
        return _handoff(context, step.doctor_id)
    default = context.pack.default_language
    event_id = current.id.hex
    try:
        context.calendar.book(
            calendar.calendars[step.doctor_id],
            current.start,
            current.end,
            event_id=event_id,
            title=f"{_service(context, step.service_id, default)} — {current.name}",
            description=entry_description(current.phone, default),
        )
    except CalendarUnavailableError:
        _release(context, step)
        return _handoff(context, step.doctor_id, _say(context, "failed"))
    with business_session(context.session_factory, context.pack.id) as session:
        mark_booked(session, current.id, event_id)
    booked = phrase(
        "booked",
        context.language,
        when=slot_time(current.start.astimezone(ZoneInfo(context.pack.timezone)), context.language),
        service=_service(context, step.service_id),
        doctor=_doctor(context, step.doctor_id),
        contact=context.pack.location.phone,
    )
    return StepReply(booked, "replied", None, step.doctor_id)


def _free(context: BookingContext, step: BookingStep, start: datetime) -> bool:
    calendar = context.pack.integrations.calendar
    if calendar is None:
        raise CalendarUnavailableError("no calendar")
    view = read_calendar(context.calendar, context.pack, step.doctor_id, context.now)
    return still_free(
        view,
        start,
        duration=timedelta(minutes=_duration(context, step.service_id)),
        now=context.now,
        horizon=timedelta(days=calendar.horizon_days),
        min_notice=timedelta(minutes=calendar.min_notice_minutes),
        buffer=timedelta(minutes=calendar.buffer_minutes),
        zone=ZoneInfo(context.pack.timezone),
    )


def _again(context: BookingContext, step: BookingStep, key: str) -> StepReply:
    """The question once more; a second wrong answer ends the booking with how to book."""
    if step.retries >= 1:
        _release(context, step)
        return _handoff(context, step.doctor_id)
    after = step.model_copy(update={"retries": step.retries + 1})
    return StepReply(_say(context, key), "asked", after, step.doctor_id)


def _cancel(context: BookingContext, step: BookingStep) -> StepReply:
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
