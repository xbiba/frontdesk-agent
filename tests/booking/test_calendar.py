"""The fixture calendar and the clinic's opening hours (ADR-0003, amended for M4)."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from frontdesk.booking.calendar import CalendarUnavailableError, FixtureCalendar, opening_intervals
from frontdesk.booking.slots import Interval
from frontdesk.tenancy.pack import OpeningHours

TIRANE = ZoneInfo("Europe/Tirane")
HOURS = [OpeningHours(days="mon-fri", opens="09:00", closes="18:00")]


def at(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=TIRANE)


def opening(day: date) -> list[Interval]:
    return opening_intervals(HOURS, day, TIRANE)


def test_opening_hours_give_the_day_s_interval_on_open_days_only() -> None:
    assert opening(date(2026, 10, 26)) == [Interval(at(26, 9), at(26, 18))]  # a Monday
    assert opening(date(2026, 10, 24)) == []  # a Saturday


def test_a_day_range_may_wrap_around_the_week() -> None:
    weekend = [OpeningHours(days="sat-mon", opens="10:00", closes="12:00")]
    assert opening_intervals(weekend, date(2026, 10, 25), TIRANE) == [
        Interval(at(25, 10), at(25, 12))
    ]
    assert opening_intervals(weekend, date(2026, 10, 27), TIRANE) == []


def test_the_fixture_gives_the_blocks_and_busy_time_inside_the_window(tmp_path: Path) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text(
        """
cal-1:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 13:00"}
    - {date: "2026-10-26"}
    - {start: "2026-11-20 09:00", end: "2026-11-20 13:00"}
  busy:
    - {start: "2026-10-23 10:00", end: "2026-10-23 11:00"}
""",
        encoding="utf-8",
    )
    view = FixtureCalendar(path).read(
        "cal-1", at(20, 0), at(31, 0), marker="Disponibile", zone=TIRANE, opening=opening
    )
    assert view.blocks == (Interval(at(23, 9), at(23, 13)), Interval(at(26, 9), at(26, 18)))
    assert view.busy == (Interval(at(23, 10), at(23, 11)),)


def test_an_unknown_calendar_cannot_be_read(tmp_path: Path) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text("cal-1: {blocks: [], busy: []}\n", encoding="utf-8")
    with pytest.raises(CalendarUnavailableError, match="unknown calendar"):
        FixtureCalendar(path).read(
            "cal-2", at(20, 0), at(31, 0), marker="Disponibile", zone=TIRANE, opening=opening
        )


def test_the_fixture_keeps_its_bookings_as_busy_time(tmp_path: Path) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text("cal-1: {blocks: [], busy: []}\n", encoding="utf-8")
    calendar = FixtureCalendar(path)
    calendar.book("cal-1", at(23, 10), at(23, 11), event_id="e1", title="t", description="d")
    calendar.book("cal-1", at(23, 10), at(23, 11), event_id="e1", title="t", description="d")
    view = calendar.read(
        "cal-1", at(20, 0), at(31, 0), marker="Disponibile", zone=TIRANE, opening=opening
    )
    assert view.busy == (Interval(at(23, 10), at(23, 11)),)
    with pytest.raises(CalendarUnavailableError, match="unknown calendar"):
        calendar.book("cal-2", at(23, 10), at(23, 11), event_id="e2", title="t", description="d")


def test_a_fresh_fixture_forgets_the_bookings(tmp_path: Path) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text("cal-1: {blocks: [], busy: []}\n", encoding="utf-8")
    calendar = FixtureCalendar(path)
    calendar.book("cal-1", at(23, 10), at(23, 11), event_id="e1", title="t", description="d")
    fresh = calendar.fresh()
    view = fresh.read(
        "cal-1", at(20, 0), at(31, 0), marker="Disponibile", zone=TIRANE, opening=opening
    )
    assert (view.busy, fresh.booked) == ((), {})
    assert calendar.booked  # the original keeps its own
