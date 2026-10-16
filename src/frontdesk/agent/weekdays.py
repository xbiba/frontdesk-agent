"""A weekday next to a date must be that date's (run agent-13, 2026-10-16: "giovedì 23 ottobre" where
clinic_hours said "venerdì 23 ottobre").

A reply names the days to come: a day of the month is the first such date from yesterday on, in the
month it names when it names one. Nearest in either direction read "lunedì 2", said on 16 October of
2 November, as 2 October (final review of #48, C3). These are no dated weekday (I3): a weekday before
a clock time ("sabato 9:00-14:00", "Friday 10:00") or before a range, and one in a list of weekdays
("sabato e domenica 24 e 25 ottobre").
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Final

from frontdesk.booking.reply import MONTHS

# Albanian weekdays by stem: "e premte", "të premten".
_WEEKDAY_NAMES: Final = (
    r"luned[iì]|monday|h[eë]n[eë]n?",
    r"marted[iì]|tuesday|mart[eë]n?",
    r"mercoled[iì]|wednesday|m[eë]rkur[eë]n?",
    r"gioved[iì]|thursday|enjten?",
    r"venerd[iì]|friday|premten?",
    r"sabato|saturday|shtun[eë]n?",
    r"domenica|sunday|diel(?:[eë]n)?",
)
_ANY_WEEKDAY: Final = "|".join(_WEEKDAY_NAMES)
_MONTH_NUMBER: Final = {
    name.lower(): number for names in MONTHS.values() for number, name in enumerate(names, 1)
}
_MONTH: Final = "|".join(_MONTH_NUMBER)
# A day of the month: not part of a clock time ("9:00", "9.30") or of a range ("9-14").
_DAY: Final = r"(\d{1,2})(?!\d)(?:st|nd|rd|th)?[°º]?(?![:.]\d)(?!\s*[-\u2013]\s*\d)"
_DATED_WEEKDAY: Final = re.compile(
    r"\b(?:("
    + ")|(".join(_WEEKDAY_NAMES)
    + r"))\b,?\s+(?:the\s+)?(?:më\s+)?"
    + _DAY
    + r"(?:\s+("
    + _MONTH
    + r")\w*)?",
    re.IGNORECASE,
)
_DAY_OF_MONTH: Final = re.compile(r"\b" + _DAY + r"\s+(" + _MONTH + r")\w*", re.IGNORECASE)
# The weekday before it, in a list: "sabato e domenica 24 e 25".
_LISTED: Final = re.compile(rf"\b(?:{_ANY_WEEKDAY})\s*(?:,|e|ed|and|dhe)\s*$", re.IGNORECASE)
_REACH: Final = 13  # the months a coming date is looked for in


def coming(day: int, month: int | None, today: date) -> date | None:
    """The first date from yesterday on with this day of the month, in `month` when it is named."""
    start = today - timedelta(days=1)
    for shift in range(_REACH):
        years, index = divmod(start.month - 1 + shift, 12)
        if month is not None and index + 1 != month:
            continue
        try:
            found = date(start.year + years, index + 1, day)
        except ValueError:
            continue
        if found >= start:
            return found
    return None


def wrong_weekdays(reply: str, today: date) -> list[str]:
    """Each weekday the reply puts next to a date that is not that date's."""
    wrong = []
    for match in _DATED_WEEKDAY.finditer(reply):
        if _LISTED.search(reply[: match.start()]):
            continue
        weekday = next(index for index in range(7) if match[index + 1])
        month = _MONTH_NUMBER.get((match[9] or "").lower())
        found = coming(int(match[8]), month, today)
        if found is not None and found.weekday() != weekday:
            wrong.append(f"{match[0]} ({found.isoformat()} is a {found:%A})")
    return wrong


def days_named(text: str) -> set[int]:
    """The days of the month the text names, next to a weekday or a month."""
    days = {int(match[8]) for match in _DATED_WEEKDAY.finditer(text)}
    return days | {int(match[1]) for match in _DAY_OF_MONTH.finditer(text)}
