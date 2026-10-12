"""Where the free slots come from and where an appointment goes (ADR-0003): a scheduler.

CalendarScheduler computes the slots from a doctor's calendar (the "Disponibile" blocks and the busy
time) and writes the event itself: Google Calendar, or the fixture of the evaluations. CalcomScheduler
(frontdesk.booking.calcom) asks cal.com, which keeps each doctor's availability and writes to the
clinic's Google Calendar (Albi, 2026-10-12). The conversation sees only this protocol.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from frontdesk.booking.calendar import Calendar, FixtureCalendar
from frontdesk.booking.flow import entry_description
from frontdesk.booking.reply import read_calendar
from frontdesk.booking.slots import Slot, free_slots
from frontdesk.tenancy.pack import BusinessPack, Language


class Scheduler(Protocol):
    def free_slots(
        self, pack: BusinessPack, doctor_id: str, service_id: str, now: datetime
    ) -> tuple[Slot, ...]:
        """Every free slot of the doctor for the service, from now to the pack's horizon. Raises
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
