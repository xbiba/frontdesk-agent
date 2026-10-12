"""Free slots computed by code (the bookings design of 2026-10-12, §3; product spec §8)."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from frontdesk.booking.slots import Interval, Slot, free_slots

TIRANE = ZoneInfo("Europe/Tirane")
HOUR = timedelta(hours=1)
WEEK_BEFORE = datetime(2026, 10, 19, 8, tzinfo=TIRANE)
FORTNIGHT = timedelta(days=14)
TWO_HOURS = timedelta(minutes=120)
NONE = timedelta(0)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TIRANE)


def block(day: int, start: int, end: int) -> Interval:
    return Interval(at(day, start), at(day, end))


def slots(
    blocks: Sequence[Interval],
    busy: Sequence[Interval] = (),
    *,
    now: datetime = WEEK_BEFORE,
    duration: timedelta = HOUR,
    horizon: timedelta = FORTNIGHT,
    min_notice: timedelta = TWO_HOURS,
    buffer: timedelta = NONE,
    limit: int = 3,
) -> list[str]:
    found = free_slots(
        blocks,
        busy,
        duration=duration,
        now=now,
        horizon=horizon,
        min_notice=min_notice,
        buffer=buffer,
        zone=TIRANE,
        limit=limit,
    )
    return [f"{slot.start:%d %H:%M}-{slot.end:%H:%M}" for slot in found]


def test_a_block_gives_whole_durations_from_its_start() -> None:
    assert slots([block(23, 9, 13)], limit=5) == [
        "23 09:00-10:00",
        "23 10:00-11:00",
        "23 11:00-12:00",
        "23 12:00-13:00",
    ]


def test_a_busy_interval_inside_a_block_splits_it() -> None:
    busy = [Interval(at(23, 10, 30), at(23, 11))]
    assert slots([block(23, 9, 13)], busy, limit=5) == [
        "23 09:00-10:00",
        "23 11:00-12:00",
        "23 12:00-13:00",
    ]


def test_the_buffer_widens_a_busy_interval_on_both_sides() -> None:
    busy = [Interval(at(23, 11), at(23, 12))]
    found = slots([block(23, 9, 14)], busy, buffer=timedelta(minutes=15), limit=5)
    assert found == ["23 09:00-10:00", "23 13:00-14:00"]


def test_the_minimum_notice_drops_the_slots_that_start_too_soon() -> None:
    assert slots([block(23, 9, 13)], now=at(23, 8), limit=5) == [
        "23 10:00-11:00",
        "23 11:00-12:00",
        "23 12:00-13:00",
    ]


def test_the_horizon_cuts_the_list() -> None:
    blocks = [block(23, 9, 10), block(30, 9, 10)]
    assert slots(blocks, now=at(20, 8), horizon=timedelta(days=7)) == ["23 09:00-10:00"]


def test_overlapping_blocks_are_merged() -> None:
    assert slots([block(23, 9, 11), block(23, 10, 12)], limit=5) == [
        "23 09:00-10:00",
        "23 10:00-11:00",
        "23 11:00-12:00",
    ]


def test_an_empty_calendar_or_a_block_shorter_than_the_service_gives_no_slot() -> None:
    assert slots([]) == []
    assert slots([Interval(at(23, 9), at(23, 9, 45))]) == []


def test_the_first_slot_of_each_day_comes_first() -> None:
    blocks = [block(23, 9, 13), block(24, 9, 13), block(26, 9, 13), block(27, 9, 13)]
    assert slots(blocks) == ["23 09:00-10:00", "24 09:00-10:00", "26 09:00-10:00"]


def test_one_long_day_alone_gives_its_first_slots() -> None:
    assert slots([block(23, 9, 18)]) == ["23 09:00-10:00", "23 10:00-11:00", "23 11:00-12:00"]


def test_two_days_fill_the_last_place_with_the_next_slot_in_time() -> None:
    found = slots([block(23, 9, 11), block(24, 9, 11)])
    assert found == ["23 09:00-10:00", "23 10:00-11:00", "24 09:00-10:00"]


def test_a_block_left_as_busy_still_counts() -> None:
    # Reception forgot "Show as: free": freeBusy returns the block itself as busy time.
    hours = block(23, 9, 11)
    assert slots([hours], [Interval(hours.start, hours.end)]) == [
        "23 09:00-10:00",
        "23 10:00-11:00",
    ]


def test_slots_keep_local_hours_across_the_daylight_saving_change() -> None:
    # Europe/Tirane moves back one hour in the night of 25 October 2026.
    found = free_slots(
        [block(24, 9, 10), block(26, 9, 10)],
        [],
        duration=HOUR,
        now=at(19, 8),
        horizon=timedelta(days=14),
        min_notice=timedelta(0),
        buffer=timedelta(0),
        zone=TIRANE,
        limit=3,
    )
    assert [(slot.start.hour, slot.start.utcoffset()) for slot in found] == [
        (9, timedelta(hours=2)),
        (9, timedelta(hours=1)),
    ]
    assert all(slot.end - slot.start == HOUR for slot in found)
    assert all(isinstance(slot, Slot) for slot in found)


def test_a_block_across_the_change_counts_real_hours() -> None:
    # 01:00 to 04:00 local in the night of the change is four real hours.
    night = Interval(
        datetime(2026, 10, 25, 1, tzinfo=TIRANE), datetime(2026, 10, 25, 4, tzinfo=TIRANE)
    )
    found = free_slots(
        [night],
        [],
        duration=HOUR,
        now=at(19, 8),
        horizon=timedelta(days=14),
        min_notice=timedelta(0),
        buffer=timedelta(0),
        zone=TIRANE,
        limit=5,
    )
    assert len(found) == 4


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (datetime(2026, 10, 23, 9), at(23, 10), "aware"),
        (at(23, 10), at(23, 9), "ends after it starts"),
    ],
)
def test_an_interval_needs_aware_times_and_a_positive_length(
    start: datetime, end: datetime, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Interval(start, end)
