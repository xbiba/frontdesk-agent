"""The review page shows only the refused and the dubious cases, escaped, with their labels."""

from __future__ import annotations

from frontdesk.live_eval.judging import Label, judge
from frontdesk.live_eval.review import render_review_page, review_cases
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, draft, manifest, record

DOC = case(
    "doc", "Is there <b>parking</b>?", behaviour="answer", contains=["six"], sources=["parking-en"]
)
FINE = case("fine", behaviour="answer", contains=["six"], sources=["parking-en"])
REFUSED = record(
    "doc",
    f"{TEXTS['no_information']}\n\n{CONTACT}",
    message="Is there <b>parking</b>?",
    outcome="handoff",
    check_reasons=("quote_not_in_source",),
    draft=draft("<script>alert(1)</script> six spaces", "six free parking spaces"),
)
ANSWERED = record("fine", "There are six free spaces.", sources=("parking-en",))


def test_only_refused_and_dubious_cases_are_reviewed() -> None:
    verdicts = judge([DOC, FINE], [REFUSED, ANSWERED], FIXED_TEXTS, {})
    assert [v.case.id for v in review_cases(verdicts)] == ["doc"]


def test_the_page_escapes_every_text_and_shows_why_a_draft_was_refused() -> None:
    verdicts = judge([DOC, FINE], [REFUSED, ANSWERED], FIXED_TEXTS, {})
    page = render_review_page(manifest(), verdicts, {}, run_dir="runs/run-1")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt; six spaces" in page
    assert "Is there &lt;b&gt;parking&lt;/b&gt;?" in page
    assert "Why: quote_not_in_source" in page
    assert 'name="refusal-doc" value="wrong"' in page
    assert 'data-case="fine"' not in page
    assert "frontdesk eval report --run-dir runs/run-1" in page


def test_existing_labels_are_shown_as_chosen() -> None:
    verdicts = judge([DOC], [REFUSED], FIXED_TEXTS, {})
    labels = {"doc": Label(refusal="wrong", note='the quote "six" is there')}
    page = render_review_page(manifest(), verdicts, labels, run_dir="runs/run-1")
    assert 'name="refusal-doc" value="wrong" checked' in page
    assert "the quote &quot;six&quot; is there" in page
