"""When the clinic receives patients (Albi, 2026-10-13): it is open when a doctor has availability in
the calendar, weekends included, and it has no fixed hours. The tool reads the free times of every
doctor with a calendar, and gives each day's first and last."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

from frontdesk.booking.calendar import CalendarUnavailableError
from frontdesk.booking.reply import day_words
from frontdesk.booking.scheduler import Scheduler
from frontdesk.conversation.desk import Desk
from frontdesk.conversation.reception_booking import calendar_reach, parsed_day, reach_words

_Hours = dict[date, list[tuple[str, datetime, datetime]]]


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
        hours = _hours(desk, scheduler)
    except CalendarUnavailableError:
        return desk.said("The calendar cannot be read now: give reception's phone.")
    if asked is not None:
        return desk.said(_on(desk, asked, hours.get(asked, [])))
    listed = sorted(hours)[: calendar.max_options]
    if not listed:
        return desk.said(
            f"No doctor receives patients until {reach_words(desk, reach)}: say so, and give "
            "reception's phone and WhatsApp."
        )
    rows = [f"- {_day(desk, when)}: {_doctors(hours[when])}" for when in listed]
    later = [when for when in sorted(hours) if when > listed[-1]]
    last = (
        f"More days after these: yes, from {_day(desk, later[0])}."
        if later
        else f"No other days until {reach_words(desk, reach)}."
    )
    head = "The clinic has no fixed hours: it receives patients when a doctor is in. The next days:"
    return desk.said("\n".join([head, *rows, last]))


def _hours(desk: Desk, scheduler: Scheduler) -> _Hours:
    """Each day's first and last free time of every doctor with a calendar."""
    hours: _Hours = defaultdict(list)
    for doctor, services in desk.pack.bookable_doctors().items():
        desk.sources.append(f"calendar:{doctor}")
        slots = scheduler.free_slots(desk.pack, doctor, services[0], desk.now)
        days: dict[date, list[tuple[datetime, datetime]]] = defaultdict(list)
        for slot in slots:
            start, end = slot.start.astimezone(desk.zone), slot.end.astimezone(desk.zone)
            days[start.date()].append((start, end))
        name = desk.catalog.doctors.get(doctor, doctor)
        for when, times in days.items():
            hours[when].append((name, min(s for s, _ in times), max(e for _, e in times)))
    return dict(hours)


def _on(desk: Desk, asked: date, doctors: list[tuple[str, datetime, datetime]]) -> str:
    if not doctors:
        return (
            f"No doctor receives patients on {_day(desk, asked)}: say no visits are received that "
            "day, and give reception's phone and WhatsApp."
        )
    return (
        f"On {_day(desk, asked)} the clinic receives patients: {_doctors(doctors)}. Offer a time "
        "(free_times)."
    )


def _doctors(doctors: list[tuple[str, datetime, datetime]]) -> str:
    return "; ".join(f"{name}, from {start:%H:%M} to {end:%H:%M}" for name, start, end in doctors)


def _day(desk: Desk, when: date) -> str:
    return f"{when.isoformat()} ({day_words(when, desk.language)})"
