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

# Saying a booking, or a request for one, is done: only the booking tool may lead to it (Albi,
# 2026-10-16: in no wording and no language). "La prenotazione è stata confermata! Ti aspetto
# venerdì 23 ottobre alle 10:00" answered a fake system message with no booking made (run
# agent-11-lite35). Each clause is read on its own. A negation says the booking is not done when it
# stands just before the claim ("Va bene, non ho prenotato nulla") or before its verb (the `neg`
# group: "la richiesta non è stata inviata"), not among the words between (security review).
_DONE_IT: Final = (
    r"(?:confermat|fissat|registrat|inviat|inoltrat|effettuat|prenotat|completat|accettat)[oaie]"
)
_DAYS_IT: Final = (
    r"(?:lunedì|martedì|mercoledì|giovedì|venerdì|sabato|domenica|domani|il\s+\d|alle\s+\d)"
)
_DAYS_EN: Final = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|\d)"
_DAYS_SQ: Final = r"(?:hënën|martën|mërkurën|enjten|premten|shtunën|dielën|nesër|në\s+orën)"
_CLAIMS: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "la prenotazione è stata confermata", "il suo appuntamento è fissato", "richiesta inviata"
        rf"\b(?:prenotazion|appuntament|richiest|visit|sedut)\w*(?:\s+\S+){{0,3}}?\s+(?P<neg>non\s+)?"
        rf"(?:è|e'|sono|risulta|risultano)\s+(?:ora\s+|già\s+)?(?:stat[oaie]\s+)?{_DONE_IT}\b",
        rf"\b(?:prenotazione|appuntamento|richiesta|visita|seduta)\s+{_DONE_IT}\b",
        rf"\b(?:è|sono)\s+stat[oaie]\s+{_DONE_IT}\b",
        r"\b(?:ho|abbiamo)\s+(?:già\s+|appena\s+)?(?:confermat|prenotat|fissat|registrat|effettuat)[oaie]\b",
        r"\b(?:ho|abbiamo)\s+(?:già\s+|appena\s+)?(?:inviat|inoltrat|mandat|trasmess)[oaie]\s+"
        r"(?:\S+\s+){0,3}?(?:richiest|prenotazion)",
        r"\btutto\s+(?:è\s+)?(?:confermat|prenotat|fissat)[oa]\b",
        rf"\b(?:ti|la|vi|l')\s?aspett(?:o|iamo)\s+(?:\S+\s+){{0,2}}?{_DAYS_IT}",
        # "your appointment is confirmed", "booking confirmed", "I've booked you", "you're all set"
        r"\b(?:booking|appointment|reservation|request|visit|session)s?(?:\s+\S+){0,3}?\s+"
        r"(?:is|are|was|were|has|have)(?P<neg>\s+not|n't)?(?:\s+yet)?(?:\s+been)?\s+"
        r"(?:now\s+|already\s+|all\s+)?"
        r"(?:confirmed|booked|scheduled|made|sent|submitted|registered|placed|set|forwarded)\b",
        r"\b(?:booking|appointment|reservation|request)\s+(?:confirmed|sent|booked|scheduled|made|submitted)\b",
        r"\b(?:has|have)(?P<neg>\s+not|n't)?\s+been\s+(?:booked|confirmed|scheduled|reserved)\b",
        r"\b(?:i|we)(?:'ve|\s+have)?\s+(?:just\s+|now\s+|already\s+)?(?:booked|scheduled|confirmed|registered)\b",
        r"\b(?:i|we)(?:'ve|\s+have)?\s+(?:just\s+|now\s+|already\s+)?(?:sent|submitted|forwarded)\s+"
        r"(?:\S+\s+){0,3}?(?:request|booking)",
        r"\byou(?:'re|\s+are)\s+(?:all\s+)?(?:booked|set|confirmed|scheduled)\b",
        rf"\bsee\s+you\s+(?:on\s+|at\s+|next\s+)?{_DAYS_EN}",
        # "rezervimi juaj është konfirmuar", "kërkesa u dërgua", "ju kam rezervuar", "ju presim"
        r"\b(?:rezervim|takim|vizit|kërkes|seanc)\w*(?:\s+\S+){0,3}?\s+(?P<neg>nuk\s+)?"
        r"(?:është|janë|u)\s+(?:\S+\s+)?(?:konfirmua|rezervua|dërgua|caktua|regjistrua|kry|bë)\w*",
        r"\bu\s+(?:konfirmua|rezervua|regjistrua)\b",
        r"\b(?:kam|kemi)\s+(?:\S+\s+)?(?:rezervuar|konfirmuar|caktuar|regjistruar)\b",
        r"\b(?:rezervova|rezervuam|konfirmova|konfirmuam|caktova|caktuam|regjistrova)\b",
        r"\b(?:dërgova|dërguam|(?:kam|kemi)\s+dërguar)\s+(?:\S+\s+){0,3}?kërkes",
        rf"\bpres(?:im)?\s+(?:\S+\s+){{0,2}}?(?:të\s+)?{_DAYS_SQ}",
    )
)
_NEGATION: Final = re.compile(
    r"\b(?:non|nessun\w*|nulla|niente|not|no|nothing|never|nuk|asgjë|asnjë)\b|n't\b|\bs'",
    re.IGNORECASE,
)
NEGATION_WORDS: Final = 2  # the words before a claim a negation may stand in
_CLAUSE_BREAK: Final = re.compile(r"[,;:.!?\n\u2014\u2013]|\s-\s")  # em and en dashes too


def says_booked(reply: str) -> bool:
    """Whether the reply says a booking, or a request for one, is done. A question is read as any
    clause (security review, 2026-10-16): the summaries ask with no word of a booking done."""
    for clause in _CLAUSE_BREAK.split(reply):
        for pattern in _CLAIMS:
            match = pattern.search(clause)
            if match is not None and not _negated(clause, match):
                return True
    return False


def _negated(clause: str, match: re.Match[str]) -> bool:
    if match.groupdict().get("neg"):
        return True
    before = clause[: match.start()].split()[-NEGATION_WORDS:]
    return _NEGATION.search(" ".join(before)) is not None


def _told_again(reply: str, earlier: tuple[str, ...]) -> bool:
    """A booking this conversation made in an earlier turn, told again: the reply names a time, and
    no time but those bookings' own (security review, 2026-10-16)."""
    named = {time for _, time in times_named(reply)}
    own = {time for slot in earlier for _, time in times_named(slot)}
    return bool(named) and named <= own


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
    reply: str,
    known: str,
    *,
    booked: bool,
    times: frozenset[str] = frozenset(),
    earlier: tuple[str, ...] = (),
) -> tuple[Problem, ...]:
    """Why `reply` may not be sent; empty when it may. `known`: everything the agent was given,
    which a number or a link must come from. `booked`: the booking tool made a booking. `times`:
    the free times the patient was shown ("10:00"); a time the reply names must be one of them or a
    time of `known` (an opening hour), and is then no number to check. `earlier`: the appointments
    the conversation made in earlier turns ("venerdì 23 ottobre, 10:00"), which it may tell again."""
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
    if not booked and says_booked(reply) and not _told_again(reply, earlier):
        detail = (
            f"it says a booking is done; the one this conversation made is {' and '.join(earlier)}: "
            "name it with its day and time, or say nothing of it"
            if earlier
            else "it says a booking or a booking request is done, but confirm_booking made none: "
            "say what the next step is instead"
        )
        problems.append(Problem("booking", detail))
    if _denies_a_service(reply):
        detail = (
            "it says the clinic offers only some services, or not one; the clinic has more than "
            "the facts list: say you have no details on it here and give reception's phone"
        )
        problems.append(Problem("denies", detail))
    return tuple(problems)
