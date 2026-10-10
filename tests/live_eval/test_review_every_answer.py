"""Every answer the model wrote goes to the reviewer in M3's first round, and the reviewer's "wrong"
fails the case (external review of 2026-10-11, points 1 and 2).

The number, link and negation filter only says where to look first: an added sentence with none of
them ("The clinic is open every Sunday") is as invented as a wrong price.
"""

from __future__ import annotations

from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.report import ALL, criteria, summarize
from frontdesk.live_eval.review import render_review_page, review_cases
from tests.live_eval.helpers import FIXED_TEXTS, case, manifest, record

INVENTED = "There are free parking spaces. The clinic is open every Sunday."
PARKING = case(
    "parking-en",
    "Is there parking?",
    behaviour="answer",
    contains=["free parking spaces"],
    sources=["parking-en"],
)
CRITICAL_PARKING = case(
    "parking-en",
    "Is there parking?",
    behaviour="answer",
    contains=["free parking spaces"],
    sources=["parking-en"],
    critical=True,
)
PRICE = case("price", "Quanto costa?", behaviour="answer", contains=["30 EUR"], sources=["price:p"])


def _answered(reply: str = INVENTED) -> object:
    return record("parking-en", reply, language="en", sources=("parking-en",))


def _status(verdicts: list[CaseVerdict]) -> dict[str, str]:
    found = criteria(summarize(verdicts), spent_eur=None, cap_eur=None)
    return {criterion.name: criterion.status for criterion in found}


def test_an_added_sentence_without_a_number_link_or_negation_goes_to_the_reviewer() -> None:
    [verdict] = judge([PARKING], [_answered()], FIXED_TEXTS, {})
    assert verdict.failures == ()  # the automatic checks find nothing wrong with it
    assert not verdict.dubious  # and the filter has nothing to point at
    assert verdict.answer_pending
    assert review_cases([verdict]) == (verdict,)
    assert _status([verdict])["Invented facts that passed the checks"] == "pending"


def test_an_answer_that_adds_a_number_is_marked_where_to_look_first() -> None:
    [verdict] = judge([PARKING], [_answered("There are 20 free parking spaces.")], FIXED_TEXTS, {})
    assert verdict.dubious
    assert verdict.answer_pending
    page = render_review_page(manifest(), [verdict], {}, run_dir="run")
    assert "adds a number, a link or a negation the expected answer lacks" in page


def test_a_record_answer_is_code_s_text_and_goes_to_no_reviewer() -> None:
    priced = record("price", "Fisioterapia: 30 EUR", sources=("price:p",))
    [verdict] = judge([PRICE], [priced], FIXED_TEXTS, {})
    assert not verdict.answer_pending
    assert review_cases([verdict]) == ()


def test_the_reviewer_s_wrong_fails_the_case_and_the_critical_criterion() -> None:
    labels = {"parking-en": Label(answer="wrong", invented_fact=True, note="no Sunday hours")}
    [verdict] = judge([CRITICAL_PARKING], [_answered()], FIXED_TEXTS, labels)
    assert not verdict.passed
    assert verdict.reasons == ("reviewer: wrong answer", "reviewer: invented fact")
    every = summarize([verdict])[ALL]
    assert (every.passed, every.critical_failures) == (0, 1)
    status = _status([verdict])
    assert status["Critical cases failed"] == "fail"
    assert status["Invented facts that passed the checks"] == "fail"


def test_a_critical_case_still_waiting_for_its_review_keeps_the_critical_criterion_pending() -> (
    None
):
    [verdict] = judge([CRITICAL_PARKING], [_answered()], FIXED_TEXTS, {})
    assert verdict.passed  # nothing failed yet
    assert _status([verdict])["Critical cases failed"] == "pending"


def test_the_reviewer_s_right_keeps_the_case_passed() -> None:
    labels = {"parking-en": Label(answer="right", note="the parking text says so")}
    [verdict] = judge([CRITICAL_PARKING], [_answered()], FIXED_TEXTS, labels)
    assert verdict.passed
    assert verdict.reasons == ()
    status = _status([verdict])
    assert status["Critical cases failed"] == "pass"
    assert status["Invented facts that passed the checks"] == "pass"


def test_an_automatic_failure_and_the_reviewer_s_wrong_are_both_given() -> None:
    labels = {"parking-en": Label(answer="wrong", note="it says nothing of parking")}
    [verdict] = judge([PARKING], [_answered("We have a garden.")], FIXED_TEXTS, labels)
    assert verdict.reasons == ("missing text: free parking spaces", "reviewer: wrong answer")
