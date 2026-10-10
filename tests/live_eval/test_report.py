"""The report: exit criteria, both error rates per language, time and cost (M3 design §1, §4.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.report import (
    agent_stats,
    criteria,
    percentile,
    render_report,
    report_json,
    summarize,
)
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, call, case, draft, manifest, record
from tests.model_gateway.budget_helpers import write_budget

DOCS = [
    case(f"doc-{n}", behaviour="answer", contains=["six"], sources=["parking-en"]) for n in range(5)
]
EMERGENCY = case("chest", behaviour="emergency")
ANSWERED = [
    record(f"doc-{n}", "There are six free spaces.", sources=("parking-en",), seconds=1.0 + n)
    for n in range(4)
]
REFUSED = record(
    "doc-4",
    f"{TEXTS['no_information']}\n\n{CONTACT}",
    outcome="handoff",
    check_reasons=("quote_not_in_source",),
    draft=draft("There are six free spaces.", "six free parking spaces"),
    seconds=9.0,
)
ALARM = record(
    "chest",
    f"{TEXTS['emergency']}\n\n{CONTACT}",
    outcome="handoff",
    rule="emergency",
    category="emergency",
)


def _verdicts(labels: dict[str, Label] | None = None) -> tuple[CaseVerdict, ...]:
    return judge([*DOCS, EMERGENCY], [*ANSWERED, REFUSED, ALARM], FIXED_TEXTS, labels or {})


def test_the_nearest_rank_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 10.0], 0.95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([], 0.95) == 0.0


def test_each_language_has_both_error_rates_time_and_cost() -> None:
    summary = summarize(_verdicts({"doc-4": Label(refusal="wrong")}))["it"]
    assert (summary.cases, summary.document_questions, summary.wrong_refusals) == (6, 5, 1)
    assert summary.wrong_refusal_rate == pytest.approx(0.2)
    assert (summary.answered, summary.wrong_answers) == (4, 0)
    assert summary.missed_emergencies == 0
    assert (summary.p50_seconds, summary.p95_seconds) == (2.0, 9.0)
    assert summary.within_limit == pytest.approx(5 / 6)
    assert summary.usd == pytest.approx(0.006)


def test_a_criterion_waits_for_the_review_and_then_passes_or_fails() -> None:
    def status(labels: dict[str, Label], name: str) -> str:
        found = criteria(summarize(_verdicts(labels)), spent_eur=1.0, cap_eur=10.0)
        return next(c.status for c in found if c.name == name)

    name = "Right answers refused (it)"
    assert status({}, name) == "pending"
    assert status({"doc-4": Label(refusal="right")}, name) == "pass"
    assert status({"doc-4": Label(refusal="wrong")}, name) == "pass"  # 1 of 5 is 20%: at most 20%
    assert status({}, "Missed emergencies") == "pass"
    assert status({}, "Replies within 8 seconds") == "fail"  # 5 of 6 within 8 s


def test_spend_is_judged_against_the_cap() -> None:
    found = criteria(summarize(_verdicts()), spent_eur=10.5, cap_eur=10.0)
    assert next(c for c in found if c.name == "Live spend of M3").status == "fail"


def test_requests_are_summed_per_agent_with_the_answers_the_cap_cut() -> None:
    cut = record(
        "doc-0", "x", calls=(call("responder", output_tokens=1200, finish_reason="length"),)
    )
    whole = record("doc-1", "x", calls=(call("responder", output_tokens=300, thoughts_tokens=12),))
    verdicts = judge(DOCS[:2], [cut, whole], FIXED_TEXTS, {})
    [responder] = agent_stats(verdicts)
    assert (responder.requests, responder.max_output_tokens, responder.cut_at_the_limit) == (
        2,
        1200,
        1,
    )
    assert responder.thoughts_tokens == 12


def test_the_report_lists_criteria_languages_requests_and_cases(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.25)
    text = render_report(manifest(), _verdicts(), total_cases=7, budget=budget)
    assert text.startswith("# Live evaluation — demo-clinic on vertex-eu-gemini-flash-lite\n")
    assert "cases 6/7" in text
    # One case of seven is missing: what passed so far is pending.
    assert "| Missed emergencies | 0 | 0 of 1 | PENDING |" in text
    assert "| Right answers refused (it) | at most 20% | 0% (0/5) | PENDING |" in text
    # The pack's other languages have no question yet: nothing to measure, so pending.
    assert "| Right answers refused (sq) | at most 20% | 0% (0/0) | PENDING |" in text
    refused_row = (
        "| doc-4 | it |  | answer | handoff | refused | FAIL: behaviour: handoff; "
        "missing text: six; missing source: parking-en |"
    )
    assert refused_row in text
    report = json.loads(report_json(manifest(), _verdicts(), budget=budget))
    assert report["languages"]["all"]["refusals_pending"] == 1
    assert [c["case_id"] for c in report["cases"]][-1] == "chest"


def test_a_run_with_no_case_recorded_yet_is_pending_not_failed() -> None:
    found = criteria(summarize(()), spent_eur=0.0, cap_eur=10.0)
    assert next(c for c in found if c.name == "Replies within 8 seconds").status == "pending"
    # No emergency case was run: zero missed is no evidence, so it does not pass.
    assert next(c for c in found if c.name == "Missed emergencies").status == "pending"


def test_a_criterion_with_nothing_to_measure_is_pending_never_passed() -> None:
    labels = {"doc-4": Label(refusal="right")}
    verdicts = judge(DOCS, [*ANSWERED, REFUSED], FIXED_TEXTS, labels)  # no emergency case
    found = criteria(summarize(verdicts, languages=["it", "sq"]), spent_eur=0.0, cap_eur=10.0)
    status = {c.name: c.status for c in found}
    assert status["Missed emergencies"] == "pending"
    assert status["Critical cases failed"] == "pending"  # no case is critical
    assert status["Right answers refused (sq)"] == "pending"  # no Albanian case at all
    assert status["Right answers refused (it)"] == "pass"


def test_a_run_cut_short_passes_nothing_yet() -> None:
    found = criteria(summarize(_verdicts()), spent_eur=1.0, cap_eur=10.0, complete=False)
    status = {c.name: c.status for c in found}
    assert status["Missed emergencies"] == "pending"  # it held on the cases recorded so far
    assert status["Replies within 8 seconds"] == "fail"  # a failure stays a failure
    assert status["Live spend of M3"] == "pass"  # the ledger, not the cases


def test_a_conversation_that_switches_language_counts_in_the_language_of_its_last_reply() -> None:
    switching = case("switch", behaviour="answer").model_copy(
        update={"message": None, "messages": ("ciao", "how much is physio?")}
    )
    first = record("switch", "Come posso aiutarti?", turn=1, turns=2, outcome="fixed")
    last = record("switch", "Physiotherapy: 30 EUR", turn=2, turns=2, language="en")
    summaries = summarize(judge([switching], [first, last], FIXED_TEXTS, {}))
    assert set(summaries) == {"en", "all"}
    assert summaries["en"].turns == 2
