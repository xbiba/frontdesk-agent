"""When the clinic receives patients (Albi, 2026-10-13): it is open when a doctor has availability in
the calendar, weekends included, and visits are by appointment. The hours are the availability set
in the calendar, not the free times: a full day is no closed day, and a full morning does not move
the start. The free times only say whether a place is left."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import NamedTuple

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.reply import day_words
from frontdesk.booking.scheduler import Scheduler
from frontdesk.booking.slots import Interval
from frontdesk.conversation.desk import Desk
from frontdesk.conversation.reception_booking import calendar_reach, parsed_day, reach_words


class _Doctor(NamedTuple):
    name: str
    hours: tuple[Interval, ...]
    free: bool  # a time is still free that day


def clinic_hours(desk: Desk, day: str | None) -> str:
    calendar, scheduler = desk.pack.integrations.calendar, desk.scheduler
    if scheduler is None or calendar is None or not desk.pack.bookable_doctors():
        return desk.said("The clinic's days are in no calendar here: give reception's phone.")
    reach = calendar_reach(desk)
    asked = parsed_day(day)
    if day and asked is None:
        return desk.said("Call clinic_hours again with the day as YYYY-MM-DD.")
    if asked is not None and asked > reach:
        return desk.said(
            f"{asked.isoformat()} is after {reach_words(desk, reach)}: say its hours are not known "
            "here, and give reception's phone for later dates."
        )
    try:
        days = _days(desk, scheduler)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone.")
    if asked is not None:
        return desk.said(_on(desk, asked, days.get(asked, [])))
    coming = [when for when in sorted(days) if desk.today <= when <= reach]
    listed = coming[: calendar.max_options]
    if not listed:
        return desk.said(
            f"No doctor receives patients until {reach_words(desk, reach)}: say so, and give "
            "reception's phone and WhatsApp."
        )
    rows = [f"- {_day(desk, when)}: {_doctors(desk, days[when])}" for when in listed]
    later = coming[len(listed) :]
    last = (
        f"More days after these: yes, from {_day(desk, later[0])}."
        if later
        else f"No other days until {reach_words(desk, reach)}."
    )
    head = (
        'Visits are by appointment ("si riceve su appuntamento"). The next days a doctor is in the '
        "clinic:"
    )
    return desk.said("\n".join([head, *rows, last]))


def _days(desk: Desk, scheduler: Scheduler) -> dict[date, list[_Doctor]]:
    """The doctors in the clinic each day, with the hours set in their calendar."""
    days: dict[date, list[_Doctor]] = defaultdict(list)
    for doctor, services in desk.pack.bookable_doctors().items():
        desk.sources.append(f"calendar:{doctor}")
        hours = scheduler.receiving(desk.pack, doctor, desk.now)
        slots = scheduler.free_slots(desk.pack, doctor, services[0], desk.now)
        free = {slot.start.astimezone(desk.zone).date() for slot in slots}
        name = desk.catalog.doctors.get(doctor, doctor)
        for when, spans in hours.items():
            days[when].append(_Doctor(name, spans, when in free))
    return dict(days)


def _on(desk: Desk, asked: date, doctors: list[_Doctor]) -> str:
    if not doctors:
        return (
            f"No doctor receives patients on {_day(desk, asked)}: say no visits are received that "
            "day, and give reception's phone and WhatsApp."
        )
    if not any(doctor.free for doctor in doctors):
        return (
            f"On {_day(desk, asked)} the doctor receives patients, but no free time is left: "
            f'{_doctors(desk, doctors)}. Say so ("riceve, ma non ci sono più posti liberi"), and '
            "give reception's phone and WhatsApp."
        )
    return (
        f"On {_day(desk, asked)} the clinic receives patients: {_doctors(desk, doctors)}. Offer a "
        "time (free_times)."
    )


def _doctors(desk: Desk, doctors: list[_Doctor]) -> str:
    return "; ".join(
        f"{doctor.name}, "
        + " and ".join(
            f"from {span.start.astimezone(desk.zone):%H:%M} to {span.end.astimezone(desk.zone):%H:%M}"
            for span in doctor.hours
        )
        + ("" if doctor.free else " (no free time left)")
        for doctor in doctors
    )


def _day(desk: Desk, when: date) -> str:
    return f"{when.isoformat()} ({day_words(when, desk.language)})"
