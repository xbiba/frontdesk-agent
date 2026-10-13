"""Where the free slots come from and where an appointment goes (ADR-0003): a scheduler.

CalendarScheduler computes the slots from a doctor's calendar (the "Disponibile" blocks and the busy
time) and writes the event itself: Google Calendar, or the fixture of the evaluations. CalcomScheduler
(frontdesk.booking.calcom) asks cal.com, which keeps each doctor's availability and writes to the
clinic's Google Calendar (Albi, 2026-10-12). The conversation sees only this protocol.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from frontdesk.booking.calendar import Calendar, FixtureCalendar
from frontdesk.booking.flow import entry_description
from frontdesk.booking.reply import read_calendar
from frontdesk.booking.slots import Interval, Slot, free_slots
from frontdesk.tenancy.pack import BusinessPack, Language


class Scheduler(Protocol):
    def free_slots(
        self, pack: BusinessPack, doctor_id: str, service_id: str, now: datetime
    ) -> tuple[Slot, ...]:
        """Every free slot of the doctor for the service, from now to the pack's horizon. Raises
        CalendarUnavailableError."""
        ...

    def receiving(
        self, pack: BusinessPack, doctor_id: str, now: datetime
    ) -> dict[date, tuple[Interval, ...]]:
        """The hours the doctor receives patients, as set in the calendar, by day from today to the
        pack's horizon: booked or not (Albi, 2026-10-13: a full day is no closed day). Raises
        CalendarUnavailableError."""
        ...

    def book(
        self,
        pack: BusinessPack,
        doctor_id: str,
        service_id: str,
        start: datetime,
        end: datetime,
        *,
        name: str,
        phone: str,
        reference: str,
        language: Language,
    ) -> None:
        """Books the slot once: the same `reference` never makes a second appointment. Raises
        CalendarUnavailableError when it cannot say that the appointment exists."""
        ...


def by_day(
    intervals: tuple[Interval, ...] | list[Interval], zone: ZoneInfo
) -> dict[date, tuple[Interval, ...]]:
    """Intervals grouped by the local day they start on, in time order."""
    days: dict[date, list[Interval]] = defaultdict(list)
    for item in sorted(intervals, key=lambda each: each.start):
        days[item.start.astimezone(zone).date()].append(item)
    return {day: tuple(items) for day, items in days.items()}


class CalendarScheduler:
    """Slots computed from the doctor's calendar, and the event written in it."""

    def __init__(self, calendar: Calendar) -> None:
        self.calendar = calendar

    def fresh(self) -> CalendarScheduler:
        """A fixture without the bookings made since; any other calendar as it is."""
        if isinstance(self.calendar, FixtureCalendar):
            return CalendarScheduler(self.calendar.fresh())
        return self

    def free_slots(
        self, pack: BusinessPack, doctor_id: str, service_id: str, now: datetime
    ) -> tuple[Slot, ...]:
        calendar = pack.integrations.calendar
        if calendar is None:
            raise ValueError(f"business '{pack.id}' has no calendar")
        view = read_calendar(self.calendar, pack, doctor_id, now)
        service = next(item for item in pack.services if item.id == service_id)
        return free_slots(
            view.blocks,
            view.busy,
            duration=timedelta(minutes=service.duration_minutes),
            now=now,
            horizon=timedelta(days=calendar.horizon_days),
            min_notice=timedelta(minutes=calendar.min_notice_minutes),
            buffer=timedelta(minutes=calendar.buffer_minutes),
            zone=ZoneInfo(pack.timezone),
            limit=10_000,
        )

    def receiving(
        self, pack: BusinessPack, doctor_id: str, now: datetime
    ) -> dict[date, tuple[Interval, ...]]:
        """The doctor's "Disponibile" blocks, by day, from the start of today."""
        zone = ZoneInfo(pack.timezone)
        today = datetime.combine(now.astimezone(zone).date(), time(), tzinfo=zone)
        return by_day(read_calendar(self.calendar, pack, doctor_id, today).blocks, zone)

    def book(
        self,
        pack: BusinessPack,
        doctor_id: str,
        service_id: str,
        start: datetime,
        end: datetime,
        *,
        name: str,
        phone: str,
        reference: str,
        language: Language,
    ) -> None:
        calendar = pack.integrations.calendar
        if calendar is None:
            raise ValueError(f"business '{pack.id}' has no calendar")
        default = pack.default_language
        service = next(item for item in pack.services if item.id == service_id)
        self.calendar.book(
            calendar.calendars[doctor_id],
            start,
            end,
            event_id=reference,
            title=f"{service.names.get(default, service_id)} — {name}",
            description=entry_description(phone, default),
        )
