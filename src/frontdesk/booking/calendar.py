"""Reading a doctor's calendar (ADR-0003, amended for M4).

One calendar per doctor. The doctor's hours are the events titled with the pack's marker word; an
all-day one means the clinic's opening hours of that day. Every other event is an appointment, and
only its busy time is read. A reader returns no event content, and an error carries a status only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import yaml

from frontdesk.booking.slots import Interval
from frontdesk.tenancy.pack import OpeningHours

Opening = Callable[[date], Sequence[Interval]]
_WEEK = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class CalendarView:
    blocks: tuple[Interval, ...]
    busy: tuple[Interval, ...]


class CalendarUnavailableError(RuntimeError):
    """The calendar could not be read. The message is a status, never an event's content."""


class CalendarReader(Protocol):
    def read(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        marker: str,
        zone: ZoneInfo,
        opening: Opening,
    ) -> CalendarView: ...


class CalendarWriter(Protocol):
    def book(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        event_id: str,
        title: str,
        description: str,
    ) -> None:
        """Writes the appointment, once: the same event id never makes a second one. Raises
        CalendarUnavailableError when it cannot say that the appointment exists."""
        ...


class Calendar(CalendarReader, CalendarWriter, Protocol):
    """A doctor's calendar, read for free slots and written with appointments (M5)."""


def opening_intervals(hours: Sequence[OpeningHours], day: date, zone: ZoneInfo) -> list[Interval]:
    """The clinic's opening hours on `day`, from the pack's structured hours."""
    weekday = day.weekday()
    return [
        Interval(_local(day, item.opens, zone), _local(day, item.closes, zone))
        for item in hours
        if weekday in _days(item.days)
    ]


def _days(days: str) -> set[int]:
    first, _, last = days.partition("-")
    start, end = _WEEK.index(first), _WEEK.index(last or first)
    if start <= end:
        return set(range(start, end + 1))
    return set(range(start, 7)) | set(range(end + 1))


def _local(day: date, clock: str, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time.fromisoformat(clock), tzinfo=zone)


class FixtureCalendar:
    """Calendars from a YAML file, for tests and evaluations: per calendar id, `blocks` (a local
    `start` and `end`, or an all-day `date`) and `busy` (a local `start` and `end`)."""

    def __init__(self, path: Path) -> None:
        self._calendars: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.booked: dict[str, dict[str, Interval]] = {}  # calendar -> event id -> its time

    def fresh(self) -> FixtureCalendar:
        """The same calendars without the bookings made since: each evaluation case starts here."""
        copy = object.__new__(FixtureCalendar)
        copy._calendars = self._calendars
        copy.booked = {}
        return copy

    def read(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        marker: str,
        zone: ZoneInfo,
        opening: Opening,
    ) -> CalendarView:
        entry = self._calendars.get(calendar_id)
        if entry is None:
            raise CalendarUnavailableError("unknown calendar")
        blocks = [
            interval
            for item in entry.get("blocks") or []
            for interval in (
                opening(date.fromisoformat(item["date"]))
                if "date" in item
                else [_timed(item, zone)]
            )
        ]
        busy = [_timed(item, zone) for item in entry.get("busy") or []]
        busy += [interval for interval, _ in self.appointments(calendar_id, zone)]
        busy += list(self.booked.get(calendar_id, {}).values())
        return CalendarView(_within(blocks, start, end), _within(busy, start, end))

    def appointments(self, calendar_id: str, zone: ZoneInfo) -> list[tuple[Interval, str | None]]:
        """The appointments the fixture declares that are not cancelled, with their phone: those
        reception made, which a patient may already have (one per person at a time)."""
        entry = self._calendars.get(calendar_id) or {}
        return [
            (_timed(item, zone), item.get("phone"))
            for item in entry.get("appointments") or []
            if item.get("status", "accepted") != "cancelled"
        ]

    def book(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        event_id: str,
        title: str,
        description: str,
    ) -> None:
        """Keeps the appointment in memory, as busy time for the next read."""
        if calendar_id not in self._calendars:
            raise CalendarUnavailableError("unknown calendar")
        self.booked.setdefault(calendar_id, {})[event_id] = Interval(start, end)


def _timed(item: dict[str, str], zone: ZoneInfo) -> Interval:
    return Interval(
        datetime.fromisoformat(item["start"]).replace(tzinfo=zone),
        datetime.fromisoformat(item["end"]).replace(tzinfo=zone),
    )


def _within(items: Sequence[Interval], start: datetime, end: datetime) -> tuple[Interval, ...]:
    return tuple(item for item in items if item.start < end and start < item.end)
