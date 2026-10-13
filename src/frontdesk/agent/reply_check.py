"""Checks on a reply the reception agent wrote (Albi, 2026-10-12: the model leads the conversation,
code keeps the facts).

Every number and link in the reply must come from what the agent was given this turn: the clinic's
facts, its tools' results, the conversation so far. It may not use diagnostic language, promise an
action nobody takes, or say a booking is confirmed when no booking was made. A reply that fails is
sent back to the model once with the reasons; a second failure ends the turn with a handoff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from frontdesk.agent.answer_check import links_in, numbers_in
from frontdesk.booking.reply import MONTHS
from frontdesk.safety.rules import contains_diagnostic_language, promise_phrase

# Saying a booking is done: only the booking tool may lead to these words.
_BOOKED: Final = re.compile(
    r"prenotazione (?:è )?confermata|ho prenotato|è stat[oa] prenotat"
    r"|booking (?:is )?confirmed|(?:i have|i've) booked|you are booked|you're booked"
    r"|rezervimi (?:u )?konfirmua|kam rezervuar|u rezervua",
    re.IGNORECASE,
)


# Saying the clinic offers only some services, or not one: the agent knows the services it prices,
# and the clinic has more ("Klinika jonë ofron vetëm trajtime osteopatike", "Non disponiamo delle
# analisi del sangue", run of 2026-10-13). What the chat books, or what is free, may be "only" that.
_SENTENCE: Final = re.compile(r"(?<=[.!?;:])\s+|\n+")
_ONLY: Final = re.compile(r"\b(?:solo|soltanto|unicamente|esclusivamente|vetëm|only)\b", re.I)
_OFFERS: Final = re.compile(
    r"\b(?:offr|forni|esegu|effettu|facciamo|dispon|serviz|trattament|prestazion"
    r"|ofr|kryej|bëjmë|shërbim|trajtim|offer|provid|perform|service|treatment)",
    re.IGNORECASE,
)
_NOT_OFFERED: Final = re.compile(
    r"\bnon\s+(?:\w+\s+){0,2}?(?:offriamo|forniamo|eseguiamo|effettuiamo|facciamo|disponiamo)\b"
    r"|\bnon\s+(?:è|sono)\s+tra\s+i\s+(?:nostri\s+)?servizi"
    r"|\bnuk\s+(?:\w+\s+){0,2}?(?:ofrojmë|kryejmë|bëjmë|disponojmë)\b"
    r"|\b(?:do|does)\s*n[o']t\s+(?:offer|provide|perform|do)\b"
    r"|\bnot\s+(?:among|one\s+of)\s+our\s+services\b",
    re.IGNORECASE,
)
_BOOKS_OR_FREE: Final = re.compile(
    r"prenot|book|rezerv|\bchat\b|orar|\btimes?\b|or[ae]r|\bposti?\b|slot|availab|disponibilit"
    r"|\bfree\b|liber|consig|advice|këshill|diagnos",
    re.IGNORECASE,
)


def _denies_a_service(reply: str) -> bool:
    for sentence in _SENTENCE.split(reply):
        if _BOOKS_OR_FREE.search(sentence):
            continue
        if _NOT_OFFERED.search(sentence) or (_ONLY.search(sentence) and _OFFERS.search(sentence)):
            return True
    return False


# The times a reply names (Albi, 2026-10-12: the model writes the times, code checks them): a clock
# time ("10:00", "9.30"), and the hours after a word that brings them, alone or in a list ("alle 9,
# 10, 12 e 16", "at 9 and 10", "në orën 9").
_HOUR: Final = r"(?:2[0-3]|[01]?\d)(?:[:.][0-5]\d)?"
_CLOCK: Final = re.compile(r"(?<![\d:.])([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)")
_HOURS: Final = re.compile(
    rf"\b(?:alle|ore|dalle|per le|at|from|until|or[eë]n|ora)\s+"
    rf"({_HOUR}(?:\s*(?:,|e|ed|and|o|or|dhe|ose|/)\s*{_HOUR})*)\b",
    re.IGNORECASE,
)

# A day of the month is no hour: "from 23 October until 2 November" was read as 23:00 and 02:00 in
# the run of 2026-10-13, and a right reply was refused twice.
_MONTH_NEXT: Final = re.compile(
    r"\s*(?:" + "|".join(month for names in MONTHS.values() for month in names) + ")",
    re.IGNORECASE,
)


# The 12-hour clock of English replies ("2 PM", "2:00 p.m."): read as the 24-hour time it is. The
# run of 2026-10-13 read "2:00 PM" as 02:00 and refused a right reply twice.
_MERIDIEM: Final = re.compile(
    r"(?<![\d:.])(1[0-2]|0?[1-9])(?:[:.]([0-5]\d))?\s*([ap])\.?\s?m\b\.?", re.IGNORECASE
)


def times_named(reply: str) -> list[tuple[tuple[int, int], str]]:
    """Each time the reply names, as "HH:MM", with where it stands in the reply."""
    found: dict[tuple[int, int], str] = {}
    for match in _MERIDIEM.finditer(reply):
        hours = int(match[1]) % 12 + (12 if match[3].lower() == "p" else 0)
        found[match.span()] = f"{hours:02d}:{match[2] or '00'}"
    for match in _CLOCK.finditer(reply):
        if any(start <= match.start() < end for start, end in found):
            continue  # part of a 12-hour time, already read
        found[match.span()] = f"{int(match[1]):02d}:{match[2]}"
    for match in _HOURS.finditer(reply):
        offset = match.start(1)
        for item in re.finditer(_HOUR, match[1]):
            span = (offset + item.start(), offset + item.end())
            if any(start <= span[0] < end for start, end in found):
                continue  # a clock time, already read
            if _MONTH_NEXT.match(reply, span[1]):
                continue  # a day of the month
            hour, _, minute = item[0].replace(".", ":").partition(":")
            found[span] = f"{int(hour):02d}:{minute or '00'}"
    return sorted(found.items())


# The numbers of a numbered list ("1) venerdì 23: ...") count nothing: they are no fact.
_LIST_MARK: Final = re.compile(r"(?m)^[ \t]*\d{1,2}[).][ \t]")


def _without(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


@dataclass(frozen=True)
class Problem:
    """Why a reply is sent back. `code` names the kind and goes to traces and logs, never a value of
    the reply; `detail` tells the model what to change and goes to the model only."""

    code: str
    detail: str


def reply_problems(
    reply: str, known: str, *, booked: bool, times: frozenset[str] = frozenset()
) -> tuple[Problem, ...]:
    """Why `reply` may not be sent; empty when it may. `known`: everything the agent was given,
    which a number or a link must come from. `booked`: the booking tool made a booking. `times`:
    the free times the patient was shown ("10:00"); a time the reply names must be one of them or a
    time of `known` (an opening hour), and is then no number to check."""
    problems = []
    if not reply.strip():
        problems.append(Problem("empty", "the reply is empty"))
    named = times_named(reply)
    allowed = set(times) | {number for number in numbers_in(known) if ":" in number}
    wrong = sorted({time for _, time in named if time not in allowed})
    if wrong:
        detail = f"times that no tool gave and no fact holds: {', '.join(wrong)}"
        problems.append(Problem("times", detail))
    rest = _LIST_MARK.sub(" ", _without(reply, [span for span, _ in named]))
    new_numbers = sorted(numbers_in(rest) - numbers_in(known))
    if new_numbers:
        detail = f"numbers that are in no fact and no tool result: {', '.join(new_numbers)}"
        problems.append(Problem("numbers", detail))
    new_links = sorted(links_in(reply) - links_in(known))
    if new_links:
        problems.append(Problem("links", f"links that are in no fact: {', '.join(new_links)}"))
    if contains_diagnostic_language(reply):
        problems.append(Problem("diagnosis", "it reads like a diagnosis or medical advice"))
    promise = promise_phrase(reply)
    if promise is not None:
        detail = f"it promises an action nobody will take: {promise}"
        problems.append(Problem("promise", detail))
    if not booked and _BOOKED.search(reply):
        detail = "it says a booking is confirmed, but confirm_booking made none"
        problems.append(Problem("booking", detail))
    if _denies_a_service(reply):
        detail = (
            "it says the clinic offers only some services, or not one; the clinic has more than "
            "the facts list: say you have no details on it here and give reception's phone"
        )
        problems.append(Problem("denies", detail))
    return tuple(problems)
