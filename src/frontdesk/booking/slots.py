"""Free slots, computed by code (ADR-0003; product spec §8). The model never writes a time.

The doctor's blocks, minus the busy intervals widened by the buffer. A slot starts at its block's
start plus whole durations. The arithmetic runs on instants, so a block across a daylight-saving
change counts its real hours; slots are shown in the business's time zone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("an interval needs aware times")
        if self.end <= self.start:
            raise ValueError("an interval ends after it starts")


@dataclass(frozen=True)
class Slot:
    start: datetime  # in the business's time zone
    end: datetime


def free_slots(
    blocks: Sequence[Interval],
    busy: Sequence[Interval],
    *,
    duration: timedelta,
    now: datetime,
    horizon: timedelta,
    min_notice: timedelta,
    buffer: timedelta,
    zone: ZoneInfo,
    limit: int,
) -> tuple[Slot, ...]:
    """Up to `limit` free slots between now + min_notice and now + horizon (by start).

    The first slot of each day comes first, for up to `limit` days; the places left take the next
    slots in time order. A busy interval that equals a block is the block itself, left as busy in
    the calendar ("Show as" not set to free), and is dropped.
    """
    hours = [_utc(item) for item in blocks]
    own = {(item.start, item.end) for item in hours}
    windows = _merged(hours)
    taken = [
        (item.start - buffer, item.end + buffer)
        for item in map(_utc, busy)
        if (item.start, item.end) not in own
    ]
    earliest, latest = now.astimezone(UTC) + min_notice, now.astimezone(UTC) + horizon
    candidates = [
        (start, start + duration)
        for window in windows
        for start in _starts(window, duration)
        if earliest <= start < latest
        and not any(start < end and begin < start + duration for begin, end in taken)
    ]
    chosen = _spread(candidates, zone, limit)
    return tuple(Slot(start.astimezone(zone), end.astimezone(zone)) for start, end in chosen)


def _utc(item: Interval) -> Interval:
    return Interval(item.start.astimezone(UTC), item.end.astimezone(UTC))


def _merged(items: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for item in sorted(items, key=lambda each: each.start):
        if merged and item.start <= merged[-1].end:
            last = merged.pop()
            merged.append(Interval(last.start, max(last.end, item.end)))
        else:
            merged.append(item)
    return merged


def _starts(window: Interval, duration: timedelta) -> list[datetime]:
    count = (window.end - window.start) // duration
    return [window.start + duration * index for index in range(count)]


def _spread(
    candidates: list[tuple[datetime, datetime]], zone: ZoneInfo, limit: int
) -> list[tuple[datetime, datetime]]:
    ordered = sorted(candidates)
    seen: set[date] = set()
    firsts = []
    for candidate in ordered:
        day = candidate[0].astimezone(zone).date()
        if day not in seen:
            seen.add(day)
            firsts.append(candidate)
    chosen = firsts[:limit]
    chosen += [item for item in ordered if item not in chosen][: limit - len(chosen)]
    return sorted(chosen)
