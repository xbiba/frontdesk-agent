"""A handoff the pipeline fell back to is a failure, never the handoff a case expects.

The runner records the rule and the rejection of every turn (Task 6), and judging reads them back
into the same checks as the offline evaluation (Task 5). A crash, a timeout, a failed model call and
a classifier answer outside its contract all end in a plain handoff: none of them says anything about
the message, so none may pass a case that expects a handoff, and each is counted as a failure.
"""

from __future__ import annotations

import pytest

from frontdesk.agent.vocab import Category, Intent
from frontdesk.conversation.behaviour import Behaviour
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnTrace
from frontdesk.live_eval.judging import as_result, judge
from frontdesk.live_eval.records import TurnRecord
from frontdesk.live_eval.report import summarize
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, call, case, draft, record

# What the pipeline falls back from. The rules are crashes and timeouts; the rejections are a model
# call that failed and the classifier answers outside its contract (an id the catalogue lacks, a
# question it had to write and left out or blank).
FALLBACK_RULES = ["error", "timeout"]
FALLBACK_REJECTIONS = [
    "classifier_error",
    "responder_error",
    "unknown_service_id",
    "unknown_doctor_id",
    "unknown_problem_category",
    "missing_general_question",
    "empty_text",
]
# Handoffs the business designed: it does not offer the thing asked, has switched the feature off,
# limits order lookups, or filters the question text the classifier writes.
DESIGNED_REJECTIONS = [
    "intent_not_allowed",
    "order_lookups_exhausted",
    "orientation_disabled",
    "order_status_disabled",
    "text_too_long",
    "text_has_markup",
    "text_has_identifiers",
    "text_failed_rules",
    "text_has_clinical_terms",
    "no_active_knowledge",
]
HANDOFF = case("handoff", behaviour="handoff", outcome="handoff")


def handed_off(**fields: object) -> TurnRecord:
    return record("handoff", CONTACT, outcome="handoff", **fields)


@pytest.mark.parametrize("rule", FALLBACK_RULES)
def test_a_crash_or_a_timeout_is_a_failure_not_the_expected_handoff(rule: str) -> None:
    [verdict] = judge([HANDOFF], [handed_off(rule=rule)], FIXED_TEXTS, {})
    assert verdict.observed is Behaviour.HANDOFF
    assert verdict.failures == (f"handoff by failure: {rule}",)
    assert verdict.failed_by_pipeline
    assert not verdict.passed


@pytest.mark.parametrize("rejection", FALLBACK_REJECTIONS)
def test_a_rejected_classifier_or_model_answer_is_a_failure_not_the_expected_handoff(
    rejection: str,
) -> None:
    [verdict] = judge([HANDOFF], [handed_off(rejection=rejection)], FIXED_TEXTS, {})
    assert verdict.observed is Behaviour.HANDOFF
    assert verdict.failures == (f"handoff by failure: {rejection}",)
    assert verdict.failed_by_pipeline
    assert not verdict.passed


@pytest.mark.parametrize("rejection", DESIGNED_REJECTIONS)
def test_a_handoff_the_business_designed_is_a_decision(rejection: str) -> None:
    [verdict] = judge([HANDOFF], [handed_off(rejection=rejection)], FIXED_TEXTS, {})
    assert verdict.failures == ()
    assert not verdict.failed_by_pipeline
    assert verdict.passed


def test_a_case_that_expects_the_fallback_takes_it_as_given() -> None:
    crash = case("crash", behaviour="handoff", rule="error")
    sent = record("crash", CONTACT, outcome="handoff", rule="error")
    [verdict] = judge([crash], [sent], FIXED_TEXTS, {})
    assert verdict.failures == ()
    assert not verdict.failed_by_pipeline


def test_the_rule_of_the_turn_is_the_one_recorded() -> None:
    wanted = case("alarm", behaviour="emergency", rule="emergency")
    sent = record("alarm", CONTACT, outcome="handoff", rule="pass", category="emergency")
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    assert verdict.failures == ("rule: pass",)


def test_the_intent_of_the_turn_is_the_one_recorded() -> None:
    wanted = case("price", behaviour="answer", intent="price")
    [verdict] = judge([wanted], [record("price", "ok", intent="hours")], FIXED_TEXTS, {})
    assert verdict.failures == ("intent: hours",)
    [right] = judge([wanted], [record("price", "ok", intent="price")], FIXED_TEXTS, {})
    assert right.failures == ()


def test_a_failure_on_an_earlier_turn_is_counted_and_names_its_turn() -> None:
    conversation = case("two", behaviour="answer").model_copy(
        update={"message": None, "messages": ("uno", "due")}
    )
    first = record(
        "two", CONTACT, turn=1, turns=2, outcome="handoff", rejection="unknown_service_id"
    )
    second = record("two", "ok", turn=2, turns=2)
    [verdict] = judge([conversation], [first, second], FIXED_TEXTS, {})
    assert verdict.failures == ("turn 1: handoff by failure: unknown_service_id",)
    assert verdict.failed_by_pipeline
    assert not verdict.wrong_answer  # the failure is the earlier turn's, not a wrong answer


def test_a_document_question_lost_to_a_failure_is_no_refusal_but_it_is_lost() -> None:
    # No check refused a draft: the model never got to write one. It is a handoff by failure, found
    # in its own column, and it counts against the criterion all the same: the person got no answer
    # to a question the documents answer (Task 7 review, I1).
    parking = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
    lost = record("parking", CONTACT, outcome="handoff", rejection="unknown_service_id")
    [verdict] = judge([parking], [lost], FIXED_TEXTS, {})
    assert verdict.document_question
    assert not verdict.refused
    assert not verdict.refusal_pending
    assert verdict.unanswered_document_question
    assert verdict.failed_by_pipeline
    summary = summarize([verdict])["it"]
    assert (summary.document_questions, summary.wrong_refusals, summary.refusals_pending) == (
        1,
        0,
        0,
    )
    assert summary.unanswered_documents == 1
    assert (summary.failed_by_pipeline, summary.passed) == (1, 0)


def test_a_failure_with_a_refused_draft_is_both_a_refusal_and_a_failure() -> None:
    parking = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
    both = record(
        "parking",
        CONTACT,
        outcome="handoff",
        rejection="responder_error",
        draft=draft("There are six free spaces.", "six free parking spaces"),
    )
    [verdict] = judge([parking], [both], FIXED_TEXTS, {})
    assert verdict.refused
    assert verdict.failed_by_pipeline


def test_the_handoff_without_contact_check_applies_to_recorded_replies() -> None:
    wanted = case("silent", behaviour="handoff")
    [verdict] = judge([wanted], [record("silent", "No", outcome="handoff")], FIXED_TEXTS, {})
    assert "handoff without contact" in verdict.failures


def test_a_recorded_turn_is_the_result_the_pipeline_returned() -> None:
    sent = record(
        "x",
        f"{TEXTS['no_information']}\n\n{CONTACT}",
        outcome="handoff",
        language="en",
        rule="pass",
        category="administrative",
        intent="price",
        rejection="unknown_doctor_id",
        check_reasons=("quote_not_in_source", "unsupported_claim"),
        sources=("parking-en", "price:physio"),
        draft=draft("t", "q"),
        calls=(call(),),
    )
    assert as_result(sent) == TurnResult(
        sent.reply,
        TurnOutcome.HANDOFF,
        "en",
        TurnTrace(
            rule="pass",
            category=Category.ADMINISTRATIVE,
            intent=Intent.PRICE,
            sources=("parking-en", "price:physio"),
            check_reasons=("quote_not_in_source", "unsupported_claim"),
            rejection="unknown_doctor_id",
        ),
    )


def test_a_turn_without_a_category_or_an_intent_has_none() -> None:
    bare = record("x", "ok", category=None, intent=None)
    trace = as_result(bare).trace
    assert (trace.category, trace.intent, trace.rejection) == (None, None, None)


def test_the_latest_complete_attempt_of_a_case_is_the_one_judged() -> None:
    parking = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
    first = record("parking", "There is no parking.", attempt="a1")
    again = record("parking", "There are six free spaces.", attempt="a2", sources=("parking-en",))
    [verdict] = judge([parking], [first, again], FIXED_TEXTS, {})
    assert verdict.passed
    assert verdict.turns == (again,)


def test_an_attempt_cut_short_does_not_replace_the_one_before_it() -> None:
    two = case("two", behaviour="answer", contains=["ok"]).model_copy(
        update={"message": None, "messages": ("uno", "due")}
    )
    done = [
        record("two", "ok", attempt="a1", turn=1, turns=2),
        record("two", "ok", attempt="a1", turn=2, turns=2),
    ]
    cut = record("two", "no", attempt="a2", turn=1, turns=2)
    [verdict] = judge([two], [*done, cut], FIXED_TEXTS, {})
    assert verdict.passed
    assert [turn.attempt for turn in verdict.turns] == ["a1", "a1"]
