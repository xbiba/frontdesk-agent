"""What a reply does for the person, in the five kinds the M3 expected answers name."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final

from frontdesk.agent.vocab import Category
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnTrace
from frontdesk.tenancy.pack import FixedTextKey

# Fallbacks, not decisions: a handoff after one of these says nothing about the message. A rule is
# a turn that crashed or ran out of time. A rejection is a model call that failed, or a classifier
# answer outside its contract that no message explains: an id the catalogue does not hold, a
# question it had to write and left out or left blank.
FAILURE_RULES: Final = frozenset({"error", "timeout"})
FAILURE_REJECTIONS: Final = frozenset(
    {
        "classifier_error",
        "responder_error",
        "unknown_service_id",
        "unknown_doctor_id",
        "unknown_problem_category",
        "missing_general_question",
        "empty_text",
    }
)
# Kept out on purpose: the handoff is the designed answer to the message. `intent_not_allowed` (the
# business does not offer it: an order at the clinic, a price at the shop), `orientation_disabled`
# and `order_status_disabled` (a feature it has switched off), `order_lookups_exhausted` (the limit
# on order lookups), the `text_*` filters on the question text (privacy), and `no_active_knowledge`
# (a deployment state: the preflight of a run refuses it).


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
    """The failure a turn fell back from, or None when the pipeline decided the reply itself.

    A failure is a crash or a timeout, a model call that failed, or a classifier answer outside its
    contract that the message does not explain (an unknown id, a missing or blank question). A
    handoff the business designed (a feature it lacks, the order-lookup limit, a filter on the
    question text) is a decision, not a failure (final review P86).
    """
    if trace.rule in FAILURE_RULES:
        return trace.rule
    if trace.rejection in FAILURE_REJECTIONS:
        return trace.rejection
    return None
