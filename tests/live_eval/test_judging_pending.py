"""What keeps a case pending, and what is refused at all (Task 7 review, M1, M2).

A criterion that waits for the review waits for a label of the right kind: a note, or a label for
the other question, settles nothing. And a draft the checks refused is a refusal only on a case
that expects an answer: on a case that expects a handoff it is nothing to review.
"""

from __future__ import annotations

import pytest

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.records import TurnRecord
from frontdesk.live_eval.report import criteria, summarize
from frontdesk.live_eval.review import review_cases
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, draft, record

PARKING = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
DUBIOUS = record("parking", "There are six free spaces, open 24 hours.", sources=("parking-en",))
REFUSED = record(
    "parking",
    f"{TEXTS['no_information']}\n\n{CONTACT}",
    outcome="handoff",
    check_reasons=("quote_not_in_source",),
    draft=draft("There are six free spaces.", "six free parking spaces"),
)


def verdict_with(label: Label | None, sent: TurnRecord) -> CaseVerdict:
    [verdict] = judge([PARKING], [sent], FIXED_TEXTS, {"parking": label} if label else {})
    return verdict


def invented(verdict: CaseVerdict) -> tuple[str, str]:
    found = criteria(summarize([verdict]), spent_eur=None, cap_eur=None)
    return next((c.value, c.status) for c in found if c.name.startswith("Invented"))


# --- M1: a label that does not answer the question leaves the case pending ----------------------


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(None, id="no-label"),
        pytest.param(Label(note="to check with the clinic"), id="a-note-only"),
        pytest.param(Label(refusal="right"), id="a-label-for-the-other-question"),
        pytest.param(Label(refusal="wrong", note="n"), id="the-other-question-and-a-note"),
    ],
)
def test_a_dubious_answer_waits_for_a_label_that_says_whether_it_is_right(
    label: Label | None,
) -> None:
    verdict = verdict_with(label, DUBIOUS)
    assert verdict.dubious
    assert verdict.answer_pending
    assert invented(verdict) == ("0", "pending")


@pytest.mark.parametrize("label", [Label(answer="right"), Label(answer="wrong")])
def test_an_answer_label_settles_it(label: Label) -> None:
    verdict = verdict_with(label, DUBIOUS)
    assert not verdict.answer_pending
    assert invented(verdict)[1] == "pass"


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(None, id="no-label"),
        pytest.param(Label(note="to check with the clinic"), id="a-note-only"),
        pytest.param(Label(answer="right"), id="a-label-for-the-other-question"),
    ],
)
def test_a_refused_answer_waits_for_a_label_that_says_whether_the_refusal_was_wrong(
    label: Label | None,
) -> None:
    verdict = verdict_with(label, REFUSED)
    assert verdict.refused
    assert verdict.refusal_pending
    found = criteria(summarize([verdict]), spent_eur=None, cap_eur=None)
    assert next(c.status for c in found if c.name == "Right answers refused (it)") == "pending"


@pytest.mark.parametrize("label", [Label(refusal="right"), Label(refusal="wrong")])
def test_a_refusal_label_settles_it(label: Label) -> None:
    assert not verdict_with(label, REFUSED).refusal_pending


# --- M2: a refusal is a refusal only where an answer was expected -------------------------------


@pytest.mark.parametrize(
    "wanted",
    [
        pytest.param(case("kept", behaviour="handoff"), id="expects-a-handoff"),
        pytest.param(case("kept"), id="expects-no-behaviour"),
        pytest.param(case("kept", behaviour="clarify"), id="expects-a-question"),
        pytest.param(case("kept", behaviour="emergency"), id="expects-an-emergency"),
        pytest.param(case("kept", behaviour="clinical"), id="expects-the-clinical-notice"),
    ],
)
def test_a_draft_on_a_case_that_expects_no_answer_is_nothing_to_review(wanted: TurnCase) -> None:
    sent = REFUSED.model_copy(update={"case_id": "kept"})
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    assert not verdict.refused
    assert not verdict.refusal_pending
    assert not verdict.document_question
    assert review_cases([verdict]) == ()
