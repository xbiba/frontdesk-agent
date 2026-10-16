"""The reception agent's booking tools (Albi, 2026-10-12). Code decides each step; the model writes
the words.

`free_times` gives every free time of each day it lists, and says whether other days have free
times: "Solo questi?" got "sì, solo questi" on 3 times of a day that had 7, when the list was cut to
3 without the model knowing (live chat, 2026-10-12). It also says until which day the calendar is
read: a list the reach cut was presented as complete (review of agent-4, 2026-10-13). `hold_time` holds a time the patient was
shown, by its day and time, once the calendar shows it still free. `prepare_booking` checks name and
phone and writes the summary; `confirm_booking` books only on a yes, in the patient's own message,
to the summary of the reply just before. Every time the agent names is checked against the times
these tools gave (frontdesk.agent.reply_check).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Final

import phonenumbers

from frontdesk.agent.vocab import ClarifyField, SlotPart
from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.flow import affirms, format_phone, phrase, read_phone, valid_name
from frontdesk.booking.holds import attach, booked_events, held
from frontdesk.booking.reply import SlotWish, booking_target, day_words, slot_time
from frontdesk.booking.scheduler import Appointment
from frontdesk.booking.slots import Slot
from frontdesk.conversation.booking_turn import StepReply, cancel, confirm, take
from frontdesk.conversation.desk import Desk
from frontdesk.conversation.state import BookingStage, BookingStep
from frontdesk.db.engine import business_session

MAX_TIMES_PER_DAY: Final = 16  # a day's free times the tool lists at most
_PARTS: Final = {SlotPart.MORNING: "in the morning", SlotPart.AFTERNOON: "in the afternoon"}


def free_times(
    desk: Desk,
    day: str | None,
    from_day: str | None,
    part: SlotPart | None,
    more: bool,
    doctor_id: str | None,
    service_id: str | None,
) -> str:
    calendar = desk.pack.integrations.calendar
    if desk.scheduler is None or calendar is None or not calendar.chat_booking:
        return desk.said("Bookings are made by reception: give its phone number.")
    target = booking_target(
        desk.pack.bookable_doctors(),
        [doctor_id] if doctor_id else [],
        [service_id] if service_id else [],
    )
    if isinstance(target, ClarifyField):
        return desk.said(f"Ask the patient which {target.value} they mean.")
    if target is None:
        return desk.said("That cannot be booked in the chat: give reception's phone number.")
    doctor, service = target
    desk.sources.append(f"calendar:{doctor}")
    try:
        existing = _existing(desk, phone=None)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone number.")
    if existing is not None:
        return _has_one(desk, existing)  # this chat's own appointment
    reach = calendar_reach(desk)
    for asked in (parsed_day(day), parsed_day(from_day)):
        if asked is not None and asked > reach:
            return desk.said(
                f"{asked.isoformat()} is after {reach_words(desk, reach)}: say its times are not "
                "known here, and give reception's phone for later dates."
            )
    try:
        every = desk.scheduler.free_slots(desk.pack, doctor, service, desk.now)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone number.")
    earlier = desk.booking
    same = earlier is not None and (earlier.doctor_id, earlier.service_id) == target
    shown = earlier.shown if same and earlier is not None else ()
    wish = SlotWish(day=_day(desk, day), start_from=_day(desk, from_day), part=part)
    days = _by_day(desk, every)
    if more and shown:
        last = max(start.astimezone(desk.zone).date() for start in shown)
        days = {when: slots for when, slots in days.items() if when > last}
    lines: list[str] = []
    fitting = {
        when: kept
        for when, slots in days.items()
        if (kept := [slot for slot in slots if wish.fits(slot.start.astimezone(desk.zone))])
    }
    if wish.narrows and not fitting and days:
        lines.append(f"No free time {_wish_text(desk, wish)}. The first free days instead:")
        fitting = days
    if not fitting:
        return desk.said(
            f"No free time{' after those shown' if more else ''} until "
            f"{reach_words(desk, reach)}: say so and give reception's phone number (kind handoff)."
        )
    fell_back = bool(lines)  # nothing fitted the wish: the first free days instead
    listed = list(fitting)[: 1 if wish.day and not fell_back else calendar.max_options]
    offered = tuple(slot.start for when in listed for slot in fitting[when][:MAX_TIMES_PER_DAY])
    desk.booking = BookingStep(
        stage=BookingStage.CHOOSE,
        doctor_id=doctor,
        service_id=service,
        offered=offered,
        shown=tuple(dict.fromkeys((*shown, *offered))),
    )
    lines.append(_listing(desk, doctor, service, listed, fitting))
    later = [when for when in fitting if when not in listed]
    others = [when for when in days if when not in listed and when not in fitting]
    lines.append(_other_days(desk, later or others))
    return desk.said("\n".join(lines))


def _by_day(desk: Desk, slots: tuple[Slot, ...]) -> dict[date, list[Slot]]:
    days: dict[date, list[Slot]] = defaultdict(list)
    for slot in sorted(slots, key=lambda item: item.start):
        days[slot.start.astimezone(desk.zone).date()].append(slot)
    return dict(days)


def _listing(
    desk: Desk,
    doctor: str,
    service: str,
    listed: list[date],
    days: dict[date, list[Slot]],
) -> str:
    names = desk.catalog.services.get(service, {})
    minutes = next(item.duration_minutes for item in desk.pack.services if item.id == service)
    head = (
        f"Free times with {desk.catalog.doctors.get(doctor, doctor)} for "
        f"{names.get(desk.language, service)} ({minutes} minutes). Every free time of each day "
        "below is listed; times a day does not list are not free:"
    )
    rows = [
        f"- {when.isoformat()} ({day_words(when, desk.language)}): "
        + ", ".join(
            f"{slot.start.astimezone(desk.zone):%H:%M}" for slot in days[when][:MAX_TIMES_PER_DAY]
        )
        for when in listed
    ]
    return "\n".join([head, *rows])


def _other_days(desk: Desk, later: list[date]) -> str:
    if not later:
        return (
            f"Other days with free times: none until {reach_words(desk, calendar_reach(desk))}. Later "
            "days are not known here: never say there are no times after it; reception knows them."
        )
    return (
        f"Other days with free times: yes, from {later[0].isoformat()} "
        f"({day_words(later[0], desk.language)}). Call free_times with a day, or with more, for "
        "them."
    )


def _wish_text(desk: Desk, wish: SlotWish) -> str:
    parts = []
    if wish.day is not None:
        parts.append(f"on {wish.day.isoformat()} ({day_words(wish.day, desk.language)})")
    elif wish.start_from is not None:
        parts.append(f"from {wish.start_from.isoformat()}")
    if wish.part is not None:
        parts.append(_PARTS[wish.part])
    return " ".join(parts)


def _day(desk: Desk, text: str | None) -> date | None:
    """A day within the calendar's reach, or None."""
    day = parsed_day(text)
    return day if day is not None and desk.today <= day <= calendar_reach(desk) else None


def parsed_day(text: str | None) -> date | None:
    try:
        return date.fromisoformat(text.strip()) if text else None
    except ValueError:
        return None


def calendar_reach(desk: Desk) -> date:
    """The last whole day the calendar is read for (the pack's horizon from now)."""
    calendar = desk.pack.integrations.calendar
    end = desk.now + timedelta(days=calendar.horizon_days if calendar else 0)
    return end.astimezone(desk.zone).date() - timedelta(days=1)


def reach_words(desk: Desk, reach: date) -> str:
    return (
        f"{reach.isoformat()} ({day_words(reach, desk.language)}), the last day the calendar is "
        "read here"
    )


def hold_time(desk: Desk, day: str, time: str) -> str:
    step = desk.booking
    _calendar_source(desk, step)
    if step is None or not step.shown:
        return desk.said("No times were shown yet: call free_times first.")
    start = _shown_start(desk, step, day, time)
    if start is None:
        return desk.said(
            f"{day} {time} is not among the times shown: call free_times for that day first."
        )
    choosing = step.model_copy(update={"stage": BookingStage.CHOOSE, "summary_turn": None})
    try:
        reply = take(desk.context(), choosing, start)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone number.")
    if reply.next is None or reply.next.stage is not BookingStage.NAME:
        desk.booking = choosing
        return desk.said(
            "That time is no longer free: say so, and call free_times again for other times."
        )
    desk.booking = reply.next
    when = slot_time(start.astimezone(desk.zone), desk.language)
    return desk.said(
        f"Held for 15 minutes: {when}. Now ask the patient for their full name and phone number."
    )


def _shown_start(desk: Desk, step: BookingStep, day: str, time: str) -> datetime | None:
    for start in step.shown:
        local = start.astimezone(desk.zone)
        if local.date().isoformat() == day.strip() and f"{local:%H:%M}" == time.strip()[:5]:
            return start
    return None


def shown_times(desk: Desk) -> frozenset[str]:
    """The times of day the patient was shown ("10:00"), which a reply may name."""
    step = desk.booking
    if step is None:
        return frozenset()
    return frozenset(f"{start.astimezone(desk.zone):%H:%M}" for start in step.shown)


def prepare_booking(desk: Desk, name: str, phone: str, country: str | None = None) -> str:
    step = desk.booking
    _calendar_source(desk, step)
    if step is None or step.hold_id is None:
        return desk.said("No time is held: call hold_time first.")
    full_name = valid_name(name)
    named = (country or "").strip().upper()
    reading = read_phone(phone, [named]) if len(named) == 2 else read_phone("", [])
    if reading.number is None:  # the country the patient named first, if the number fits it
        reading = read_phone(phone, _phone_regions(desk, named))
    if full_name is None:
        return desk.said("That is not a first and last name: ask for it again.")
    if reading.options:
        return desk.said(
            f"That phone number could be {' or '.join(reading.options)}: ask which one is theirs."
        )
    number = reading.number
    if number is None:
        return desk.said(
            "That phone number fits no country: ask for it again, with its prefix if it is not "
            "Albanian or Italian."
        )
    try:
        existing = _existing(desk, phone=number)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone number.")
    if existing is not None:
        cancel(desk.context(), step)  # the time goes back to the others
        desk.booking = None
        if existing.reference in _mine(desk):
            return _has_one(desk, existing)
        # Whoever types a number may not own it: when and with whom stay with reception (the
        # security review of 2026-10-13: an appointment's time, read with someone else's number).
        return desk.said(
            "There is already an appointment under this phone number. Do not say when, or with "
            "whom: say only that this number has an appointment, as in \"Per questo numero c'è "
            "già un appuntamento; per cambiarlo o aggiungerne un altro, la reception la aiuta al "
            "…\", with reception's phone (kind handoff). Book no other."
        )
    with business_session(desk.session_factory, desk.pack.id) as session:
        current = held(session, step.hold_id, desk.now)
        if current is not None:
            attach(session, current.id, name=full_name, phone=number)
    if current is None:
        desk.booking = None
        return desk.said("The time is no longer held: call free_times again.")
    desk.booking = step.model_copy(
        update={"stage": BookingStage.CONFIRM, "summary_turn": desk.turn}
    )
    calendar = desk.pack.integrations.calendar
    summary = phrase(
        "summary_request" if calendar and calendar.requires_confirmation else "summary",
        desk.language,
        when=slot_time(current.start.astimezone(desk.zone), desk.language),
        service=desk.catalog.services.get(step.service_id, {}).get(desk.language, step.service_id),
        doctor=desk.catalog.doctors.get(step.doctor_id, step.doctor_id),
        name=full_name,
        phone=format_phone(number),
    )
    desk.summary = summary
    return desk.said(summary)


def confirm_booking(desk: Desk) -> str:
    step = desk.booking
    _calendar_source(desk, step)
    if step is None or step.stage is not BookingStage.CONFIRM:
        return desk.said("There is no summary to confirm: call prepare_booking first.")
    if step.summary_turn != desk.turn - 1:
        # A yes counts only as the answer to the summary in the reply just before it.
        return desk.said("The summary was not your last message: call prepare_booking again.")
    if not affirms(desk.message):
        return desk.said("The patient has not said yes in their last message: ask them to confirm.")
    try:
        reply: StepReply = confirm(desk.context(), step)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone number.")
    desk.booking = reply.next
    desk.booked = reply.booked
    desk.booked_text = reply.text if reply.booked else None
    if not reply.booked:
        return desk.said(f"Not booked: {reply.text} Say so, and call free_times for other times.")
    return desk.said(reply.text)


def _existing(desk: Desk, phone: str | None) -> Appointment | None:
    """The patient's appointment still to come, if any: one this conversation made, or one under
    the same phone, made in the chat or by reception (Albi, 2026-10-13: one per person at a time).
    Raises CalendarUnavailableError."""
    mine = _mine(desk)
    if desk.scheduler is None or not (mine or phone):
        return None
    digits = _digits(phone)
    for appointment in desk.scheduler.upcoming(desk.pack, desk.now):
        if appointment.reference in mine or (digits and _digits(appointment.phone) == digits):
            return appointment
    return None


# The country a conversation's language suggests for a phone without its prefix.
_LANGUAGE_REGIONS: Final = {"sq": "AL", "it": "IT"}


def _phone_regions(desk: Desk, named: str) -> list[str]:
    """Where a phone without its prefix may be from, the likeliest first: the country the patient
    named, their language's, the clinic's."""
    found = [named if len(named) == 2 else "", _LANGUAGE_REGIONS.get(desk.language, "")]
    try:  # the clinic's country from its phone's prefix, a valid number or not
        clinic = phonenumbers.parse(desk.pack.location.phone, None)
        found.append(phonenumbers.region_code_for_country_code(clinic.country_code or 0))
    except phonenumbers.NumberParseException:
        pass
    return list(dict.fromkeys(region for region in found if region and region != "ZZ"))


def made_earlier(desk: Desk) -> bool:
    """Whether this conversation has an appointment still to come, which a reply may tell again."""
    return bool(_mine(desk))


def _mine(desk: Desk) -> set[str]:
    """The references of this chat's appointments still to come."""
    with business_session(desk.session_factory, desk.pack.id) as session:
        return set(booked_events(session, desk.conversation_id, desk.now))


def _digits(phone: str | None) -> str:
    return "".join(char for char in phone or "" if char.isdigit())


def _has_one(desk: Desk, appointment: Appointment) -> str:
    when = slot_time(appointment.start.astimezone(desk.zone), desk.language)
    return desk.said(
        f'The patient already has an appointment: {when}. Book no other: say so, as in "Ha già '
        "un appuntamento il … alle …; per cambiarlo o aggiungerne un altro, la reception la aiuta "
        "al …\", with reception's phone (kind handoff)."
    )


def cancel_booking(desk: Desk) -> str:
    _calendar_source(desk, desk.booking)
    if desk.booking is not None:
        cancel(desk.context(), desk.booking)
        desk.booking = None
    return desk.said("Nothing is booked; the time is free again.")


def _calendar_source(desk: Desk, step: BookingStep | None) -> None:
    if step is not None:
        desk.sources.append(f"calendar:{step.doctor_id}")
