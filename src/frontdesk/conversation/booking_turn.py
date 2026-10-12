"""The booking operations the reception agent's tools call (M5; the agent since 2026-10-12).

Code decides every one: the free slots offered, numbered (the first ones, those that fit a
wish, or the ones not shown yet); a hold on a slot that was offered and is still free; the
booking of the held slot, once the patient said yes to its summary, with the calendar read
again first; the release of the hold. A slot taken meanwhile, an expired hold or a calendar
that cannot be read each end in fresh slots or in how to book through reception, never in a
booking that is not there. Name and phone live on the hold only (frontdesk.booking.holds).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.flow import phrase
from frontdesk.booking.holds import held, hold, mark_booked, release
from frontdesk.booking.reply import SlotWish, compose_slots, slot_time, wish_words
from frontdesk.booking.scheduler import Scheduler
from frontdesk.booking.slots import spread
from frontdesk.conversation.state import BookingStage, BookingStep
from frontdesk.db.engine import business_session
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, FixedTextKey, Language

NO_WISH: Final = SlotWish()


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


def offer(
    context: BookingContext,
    doctor_id: str,
    service_id: str,
    *,
    prefix: str | None = None,
    wish: SlotWish = NO_WISH,
    shown: tuple[datetime, ...] = (),
    others: bool = False,
) -> StepReply:
    """Free slots to pick from, numbered: the first ones, those that fit the patient's wish, or,
    with `others`, those not `shown` yet. When none fits, it says so and offers the first ones; how
    to book when none is free at all. Raises CalendarUnavailableError."""
    every = context.scheduler.free_slots(context.pack, doctor_id, service_id, context.now)
    zone = ZoneInfo(context.pack.timezone)
    language = context.language
    lead, candidates = "first", every
    if wish.narrows:
        fitting = tuple(slot for slot in every if wish.fits(slot.start.astimezone(zone)))
        if fitting:
            lead, candidates = "wished", fitting
        elif every:
            when = wish_words(wish, language)
            prefix = _joined(prefix, phrase("none_wished", language, when=when))
    elif others:
        rest = tuple(slot for slot in every if slot.start not in shown)
        if rest:
            lead, candidates = "more", rest
        elif every:
            days = str(_calendar(context).horizon_days)
            prefix = _joined(prefix, phrase("no_more", language, days=days))
    slots = spread(candidates, zone, _calendar(context).max_options)
    texts = context.pack.fixed_texts[language]
    if not slots:
        none = texts.get(FixedTextKey.NO_SLOTS) or texts[FixedTextKey.NO_INFORMATION]
        return _handoff(context, doctor_id, _joined(prefix, none))
    listed = compose_slots(
        slots,
        doctor=_doctor(context, doctor_id),
        service=_service(context, service_id),
        minutes=_duration(context, service_id),
        language=language,
        numbered=True,
        lead=lead,
        wish=wish,
    )
    offered = tuple(slot.start for slot in slots)
    earlier = shown if lead != "first" else ()
    step = BookingStep(
        stage=BookingStage.CHOOSE,
        doctor_id=doctor_id,
        service_id=service_id,
        offered=offered,
        shown=tuple(dict.fromkeys((*earlier, *offered))),
    )
    text = _joined(prefix, listed, phrase("choose", language))
    return StepReply(text, "replied", step, doctor_id)  # an answer (the slots) that invites a pick


def take(context: BookingContext, step: BookingStep, start: datetime) -> StepReply:
    """A hold on the slot picked, then the name; fresh slots when it was taken meanwhile.
    Raises CalendarUnavailableError."""
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
    after = step.model_copy(update={"stage": BookingStage.NAME, "hold_id": hold_id})
    return StepReply(_say(context, "ask_name"), "asked", after, step.doctor_id)


def confirm(context: BookingContext, step: BookingStep) -> StepReply:
    """The patient said yes: the held slot is booked, once the calendar shows it still free.
    Raises CalendarUnavailableError."""
    with business_session(context.session_factory, context.pack.id) as session:
        current = held(session, _hold_id(step), context.now)
    if current is None or current.name is None or current.phone is None:
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "expired"))
    if not _free(context, step, current.start):
        _release(context, step)
        return offer(context, step.doctor_id, step.service_id, prefix=_say(context, "taken"))
    event_id = current.id.hex
    try:
        context.scheduler.book(
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
        mark_booked(session, current.id, event_id)
    booked = phrase(
        "booked",
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


def country_code(pack: BusinessPack) -> str:
    """The clinic's own country code ("+355" from "+355 69 443 9339"), for a national number."""
    found = re.match(r"\+\d{1,3}(?=[\s\-.(]|$)", pack.location.phone.strip())
    return found.group(0) if found else ""


def _calendar(context: BookingContext) -> CalendarIntegration:
    if context.pack.integrations.calendar is None:
        raise ValueError(f"business '{context.pack.id}' has no calendar")
    return context.pack.integrations.calendar
