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
from frontdesk.safety.rules import contains_diagnostic_language, promise_phrase

# Saying a booking is done: only the booking tool may lead to these words.
_BOOKED: Final = re.compile(
    r"prenotazione (?:è )?confermata|ho prenotato|è stat[oa] prenotat"
    r"|booking (?:is )?confirmed|(?:i have|i've) booked|you are booked|you're booked"
    r"|rezervimi (?:u )?konfirmua|kam rezervuar|u rezervua",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Problem:
    """Why a reply is sent back. `code` names the kind and goes to traces and logs, never a value of
    the reply; `detail` tells the model what to change and goes to the model only."""

    code: str
    detail: str


def reply_problems(reply: str, known: str, *, booked: bool) -> tuple[Problem, ...]:
    """Why `reply` may not be sent; empty when it may. `known`: everything the agent was given,
    which a number or a link must come from. `booked`: the booking tool made a booking."""
    problems = []
    if not reply.strip():
        problems.append(Problem("empty", "the reply is empty"))
    new_numbers = sorted(numbers_in(reply) - numbers_in(known))
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
    return tuple(problems)
