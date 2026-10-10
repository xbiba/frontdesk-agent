"""What a reply does for the person, in the five kinds the M3 expected answers name."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final

from frontdesk.agent.vocab import Category
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnTrace
from frontdesk.tenancy.pack import FixedTextKey

# Fallbacks, not decisions: a handoff after one of these says nothing about the message.
FAILURE_RULES: Final = frozenset({"error", "timeout"})
FAILURE_REJECTIONS: Final = frozenset({"classifier_error", "responder_error"})


class Behaviour(StrEnum):
    ANSWER = "answer"  # the bot answers by itself: a fact, a quoted answer, a greeting, a notice
    CLARIFY = "clarify"  # it asks what the person means
    HANDOFF = "handoff"  # it gives the contact of a person
    CLINICAL = "clinical"  # it gives the clinical notice, then the contact
    EMERGENCY = "emergency"  # it gives the emergency text, then the contact


def observed_behaviour(result: TurnResult, texts: Mapping[FixedTextKey, str]) -> Behaviour:
    """The kind of reply `result` is; `texts` are the pack's fixed texts in the reply's language.

    An emergency is read from the trace (the rules or the classifier decided it); the clinical
    notice and the clarification from their fixed texts, which only the pipeline writes.
    """
    if result.trace.category is Category.EMERGENCY:
        return Behaviour.EMERGENCY
    if texts[FixedTextKey.CLINICAL] in result.text:
        return Behaviour.CLINICAL
    if result.outcome is TurnOutcome.HANDOFF:
        return Behaviour.HANDOFF
    if result.outcome is TurnOutcome.ASKED or texts[FixedTextKey.CLARIFICATION] in result.text:
        return Behaviour.CLARIFY
    return Behaviour.ANSWER


def pipeline_failure(trace: TurnTrace) -> str | None:
    """The failure a turn fell back from (an error, a timeout, a model that answered off its
    contract), or None when the pipeline decided the reply itself (final review P86)."""
    if trace.rule in FAILURE_RULES:
        return trace.rule
    if trace.rejection in FAILURE_REJECTIONS:
        return trace.rejection
    return None
