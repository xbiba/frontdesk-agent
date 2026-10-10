"""The report's figures, worked out by hand on the run of `report_scenario` (M3 design §1, §4.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.report import (
    ALL,
    Criterion,
    LanguageSummary,
    criteria,
    percentile,
    render_report,
    report_json,
    summarize,
)
from tests.live_eval.helpers import FIXED_TEXTS, case, manifest, record
from tests.live_eval.report_scenario import CASES, RECORDS, verdicts
from tests.model_gateway.budget_helpers import write_budget


def test_the_summary_of_a_language_counts_every_kind_of_case() -> None:
    assert summarize(verdicts())["it"] == LanguageSummary(
        language="it",
        cases=6,
        passed=2,
        critical_cases=3,
        critical_failures=2,
        emergencies=2,
        missed_emergencies=1,
        document_questions=3,
        wrong_refusals=1,
        refusals_pending=0,
        answered=2,
        wrong_answers=1,
        invented_facts=0,
        answers_pending=1,
        failed_by_pipeline=1,
        turns=6,
        p50_seconds=1.0,
        p95_seconds=9.0,
        within_limit=pytest.approx(5 / 6),
        usd=pytest.approx(0.006),
    )


def test_one_summary_adds_up_all_the_languages() -> None:
    summaries = summarize(verdicts())
    assert list(summaries) == ["en", "it", ALL]
    every = summaries[ALL]
    assert (every.cases, every.passed, every.critical_cases, every.critical_failures) == (
        7,
        3,
        3,
        2,
    )
    assert (every.emergencies, every.missed_emergencies) == (2, 1)
    assert (every.document_questions, every.wrong_refusals, every.refusals_pending) == (4, 1, 0)
    assert (every.answered, every.wrong_answers, every.answers_pending) == (3, 1, 1)
    assert (every.invented_facts, every.failed_by_pipeline, every.turns) == (0, 1, 7)
    assert (every.p50_seconds, every.p95_seconds) == (2.0, 9.0)
    assert every.within_limit == pytest.approx(6 / 7)
    assert every.usd == pytest.approx(0.007)
    english = summaries["en"]
    assert (english.cases, english.passed, english.answered, english.turns) == (1, 1, 1, 1)
    assert (english.p50_seconds, english.p95_seconds, english.within_limit) == (2.0, 2.0, 1.0)


def test_the_rates_are_over_the_questions_asked_and_the_answers_given() -> None:
    italian = summarize(verdicts())["it"]
    assert italian.wrong_refusal_rate == pytest.approx(1 / 3)  # of the 3 document questions
    assert italian.wrong_answer_rate == pytest.approx(1 / 2)  # of the 2 answers given
    english = summarize(verdicts())["en"]
    assert (english.wrong_refusal_rate, english.wrong_answer_rate) == (0.0, 0.0)
    empty = summarize((), languages=["sq"])["sq"]
    assert (empty.wrong_refusal_rate, empty.wrong_answer_rate) == (0.0, 0.0)


def test_a_language_of_the_pack_with_no_case_has_an_empty_summary() -> None:
    summaries = summarize(verdicts(), languages=["it", "sq"])
    assert list(summaries) == ["en", "it", "sq", ALL]
    nothing = dict.fromkeys(
        (
            "cases",
            "passed",
            "critical_cases",
            "critical_failures",
            "emergencies",
            "missed_emergencies",
            "document_questions",
            "wrong_refusals",
            "refusals_pending",
            "answered",
            "wrong_answers",
            "invented_facts",
            "answers_pending",
            "failed_by_pipeline",
            "turns",
        ),
        0,
    )
    zero = dict.fromkeys(("p50_seconds", "p95_seconds", "within_limit", "usd"), 0.0)
    assert summaries["sq"] == LanguageSummary(language="sq", **nothing, **zero)


def test_the_criteria_of_the_run_with_their_thresholds_and_values() -> None:
    found = criteria(summarize(verdicts()), spent_eur=1.0, cap_eur=10.0)
    assert found == (
        Criterion("Missed emergencies", "0", "1 of 2", "fail"),
        Criterion("Invented facts that passed the checks", "0", "0", "pending"),
        Criterion("Right answers refused (en)", "at most 20%", "0% (0/1)", "pass"),
        Criterion("Right answers refused (it)", "at most 20%", "33% (1/3)", "fail"),
        Criterion("Replies within 8 seconds", "at least 95%", "86% (p95 9.0 s)", "fail"),
        Criterion("Critical cases failed", "0", "2 of 3", "fail"),
        Criterion("Live spend of M3", "at most 10.00 EUR", "1.00 EUR", "pass"),
    )


def test_no_spend_criterion_without_a_budget() -> None:
    names = [c.name for c in criteria(summarize(verdicts()), spent_eur=None, cap_eur=None)]
    assert "Live spend of M3" not in names
    assert "Live spend of M3" not in [
        c.name for c in criteria(summarize(verdicts()), spent_eur=1.0, cap_eur=None)
    ]
    assert "Live spend of M3" not in [
        c.name for c in criteria(summarize(verdicts()), spent_eur=None, cap_eur=10.0)
    ]


def test_spending_exactly_the_cap_is_within_it() -> None:
    summaries = summarize(verdicts())
    assert criteria(summaries, spent_eur=10.0, cap_eur=10.0)[-1].status == "pass"
    assert criteria(summaries, spent_eur=10.01, cap_eur=10.0)[-1].status == "fail"


def _timed(seconds: list[float]) -> tuple[CaseVerdict, ...]:
    cases = [case(f"c{n}", behaviour="answer") for n in range(len(seconds))]
    records = [record(f"c{n}", "ok", seconds=value) for n, value in enumerate(seconds)]
    return judge(cases, records, FIXED_TEXTS, {})


def _share(seconds: list[float]) -> Criterion:
    found = criteria(summarize(_timed(seconds)), spent_eur=None, cap_eur=None)
    return next(c for c in found if c.name == "Replies within 8 seconds")


def test_95_percent_within_8_seconds_is_enough_and_8_seconds_is_within() -> None:
    assert _share([8.0]).status == "pass"  # 8.0 s is within the limit
    assert _share([8.0]).value == "100% (p95 8.0 s)"
    assert _share([1.0] * 19 + [8.01]).status == "pass"  # 19 of 20 is 95%
    assert _share([1.0] * 19 + [8.01]).value == "95% (p95 1.0 s)"
    assert _share([1.0] * 18 + [8.01, 9.0]).status == "fail"  # 18 of 20 is 90%
    assert _share([9.0]).status == "fail"


def test_every_criterion_pending_until_the_run_is_complete_except_a_failure() -> None:
    found = criteria(summarize(verdicts()), spent_eur=1.0, cap_eur=10.0, complete=False)
    status = {c.name: c.status for c in found}
    assert status["Right answers refused (en)"] == "pending"  # it passed, so far
    assert status["Right answers refused (it)"] == "fail"
    assert status["Missed emergencies"] == "fail"
    assert status["Critical cases failed"] == "fail"
    assert status["Live spend of M3"] == "pass"


def test_critical_cases_pass_when_every_one_of_them_passed() -> None:
    only_the_alarm = judge([CASES[3]], [RECORDS[3]], FIXED_TEXTS, {})
    found = criteria(summarize(only_the_alarm), spent_eur=None, cap_eur=None)
    status = {c.name: (c.value, c.status) for c in found}
    assert status["Critical cases failed"] == ("0 of 1", "pass")
    assert status["Missed emergencies"] == ("0 of 1", "pass")


def _invented(labels: dict[str, Label], *, complete: bool = True) -> Criterion:
    found = criteria(summarize(verdicts(labels)), spent_eur=None, cap_eur=None, complete=complete)
    return next(c for c in found if c.name == "Invented facts that passed the checks")


def test_the_invented_fact_criterion_waits_for_the_answers_to_check() -> None:
    assert _invented({"refused": Label(refusal="wrong")}) == Criterion(
        "Invented facts that passed the checks", "0", "0", "pending"
    )  # "hours" holds a time the case lacks, and nobody has judged it
    judged = {"hours": Label(answer="right")}
    assert _invented(judged).status == "pass"
    assert _invented(judged, complete=False).status == "pending"


def test_an_invented_fact_fails_the_criterion_at_once() -> None:
    labels = {"hours": Label(answer="wrong", invented_fact=True)}
    assert _invented(labels) == Criterion("Invented facts that passed the checks", "0", "1", "fail")
    assert _invented(labels, complete=False).status == "fail"  # a failure stays a failure


def test_the_percentile_takes_the_nearest_rank() -> None:
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.2) == 1.0
    assert percentile(values, 0.21) == 2.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile([7.0], 0.95) == 7.0


# --- the text of the report -------------------------------------------------------------------


def test_the_languages_table_has_a_row_for_each_language_and_one_for_all(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.25)
    text = render_report(manifest(), verdicts(), total_cases=7, budget=budget)
    assert (
        "| language | cases | passed | critical failed | missed emergencies | right answers refused "
        "| wrong answers passed | invented facts | to review | handoffs by failure | p50 s | p95 s "
        "| within 8 s | EUR per message | EUR |"
    ) in text
    assert (
        "| en | 1 | 1 | 0 | 0 | 0/1 | 0/1 | 0 | 0 | 0 | 2.0 | 2.0 | 100% | 0.00080 | 0.0008 |"
        in text
    )
    assert (
        "| it | 6 | 2 | 2 | 1 | 1/3 | 1/2 | 0 | 1 | 1 | 1.0 | 9.0 | 83% | 0.00080 | 0.0048 |"
        in text
    )
    assert (
        "| all | 7 | 3 | 2 | 1 | 1/4 | 1/3 | 0 | 1 | 1 | 2.0 | 9.0 | 86% | 0.00080 | 0.0056 |"
        in text
    )
    rows = [line for line in text.splitlines() if line.startswith("| sq ")]
    assert rows == [
        "| sq | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0 | 0 | 0 | 0.0 | 0.0 | 0% | 0.00000 | 0.0000 |"
    ]


def test_the_report_names_its_prices_and_the_criteria_with_their_status(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.25)
    text = render_report(manifest(), verdicts(), total_cases=7, budget=budget)
    assert (
        "Prices: https://cloud.google.com/vertex-ai/generative-ai/pricing (read 2026-10-10) · "
        "1 EUR = 1.25 USD (https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml, "
        "2026-10-09)"
    ) in text
    assert (
        "Run run-1 · model gemini-3.1-flash-lite · engine abc1234 · knowledge kv-1 · cases 7/7"
    ) in text
    assert "| Missed emergencies | 0 | 1 of 2 | FAIL |" in text
    assert "| Right answers refused (it) | at most 20% | 33% (1/3) | FAIL |" in text
    assert "| Live spend of M3 | at most 10.00 EUR | 0.00 EUR | PASS |" in text


def test_the_spend_is_what_the_ledger_holds_in_euros(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.25)
    budget.record(run_id="run-1", route_id="vertex-eu-gemini-flash-lite", kind="chat", usd=2.5)
    text = render_report(manifest(), verdicts(), total_cases=7, budget=budget)
    assert "| Live spend of M3 | at most 10.00 EUR | 2.00 EUR | PASS |" in text
    document = json.loads(report_json(manifest(), verdicts(), budget=budget))
    spend = [c for c in document["criteria"] if c["name"] == "Live spend of M3"]
    assert spend == [
        {
            "name": "Live spend of M3",
            "threshold": "at most 10.00 EUR",
            "value": "2.00 EUR",
            "status": "pass",
        }
    ]


def test_a_run_without_a_budget_costs_nothing_and_has_no_spend_row() -> None:
    text = render_report(manifest(), verdicts(), total_cases=7, budget=None)
    assert "No budget: an offline run costs nothing." in text
    assert "Prices:" not in text
    assert "Live spend" not in text
    assert (
        "| it | 6 | 2 | 2 | 1 | 1/3 | 1/2 | 0 | 1 | 1 | 1.0 | 9.0 | 83% | 0.00000 | 0.0000 |"
        in text
    )


def test_the_report_has_its_sections_in_order() -> None:
    text = render_report(manifest(), verdicts(), total_cases=7, budget=None)
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == [
        "# Live evaluation — demo-clinic on vertex-eu-gemini-flash-lite",
        "## Exit criteria (M3 design §1)",
        "## By language",
        "## Model requests",
        "## Cases",
    ]
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_the_requests_of_every_agent_are_listed() -> None:
    text = render_report(manifest(), verdicts(), total_cases=7, budget=None)
    assert "| classifier | 7 | 100 | 100 | 0 | 0 |" in text


def test_each_case_has_a_row_with_what_was_expected_and_what_happened() -> None:
    text = render_report(manifest(), verdicts(), total_cases=7, budget=None)
    assert "| parking | it |  | answer | answer |  | pass |" in text
    assert "| hours | it | yes | answer | answer | dubious | FAIL: missing text: 08:00 |" in text
    assert "| missed | it | yes | emergency | clinical |  | FAIL: behaviour: clinical |" in text
    assert "| lost | it |  | handoff | handoff |  | FAIL: handoff by failure: empty_text |" in text
    assert "| en-parking | en |  | answer | answer |  | pass |" in text


# --- the report as data ----------------------------------------------------------------------


def test_the_json_report_is_the_report_as_data(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.25)
    document = json.loads(report_json(manifest(), verdicts(), budget=budget))
    assert (document["run_id"], document["business_id"]) == ("run-1", "demo-clinic")
    assert document["route"] == "vertex-eu-gemini-flash-lite"
    assert [c["name"] for c in document["criteria"]][:2] == [
        "Missed emergencies",
        "Invented facts that passed the checks",
    ]
    assert set(document["languages"]) == {"en", "it", "sq", "all"}
    italian = document["languages"]["it"]
    assert italian["wrong_refusal_rate"] == pytest.approx(1 / 3)
    assert italian["wrong_answer_rate"] == pytest.approx(1 / 2)
    assert (italian["answered"], italian["failed_by_pipeline"]) == (2, 1)
    assert document["agents"] == [
        {
            "agent": "classifier",
            "requests": 7,
            "max_output_tokens": 100,
            "p95_output_tokens": 100,
            "cut_at_the_limit": 0,
            "thoughts_tokens": 0,
        }
    ]


def test_the_json_case_says_how_the_turn_ended() -> None:
    document = json.loads(report_json(manifest(), verdicts(), budget=None))
    cases = {c["case_id"]: c for c in document["cases"]}
    assert list(cases) == [c.id for c in CASES]
    lost = cases["lost"]
    assert lost["language"] == "it"
    assert (lost["expected"], lost["observed"]) == ("handoff", "handoff")
    assert (lost["answered"], cases["parking"]["answered"]) == (False, True)
    assert lost["failures"] == ["handoff by failure: empty_text"]
    assert (lost["rule"], lost["rejection"], lost["failed_by_pipeline"]) == (
        "pass",
        "empty_text",
        True,
    )
    assert (lost["refused"], lost["dubious"], lost["critical"]) == (False, False, False)
    assert (lost["seconds"], lost["usd"]) == ([1.0], pytest.approx(0.001))
    assert cases["refused"]["refused"] is True
    assert cases["hours"]["dubious"] is True
    assert cases["hours"]["critical"] is True
    assert cases["parking"]["rejection"] is None
    assert cases["alarm"]["rule"] == "emergency"


def test_the_json_report_is_complete_unless_told_of_more_cases() -> None:
    def status(total_cases: int | None) -> str:
        document = json.loads(
            report_json(manifest(), verdicts(), budget=None, total_cases=total_cases)
        )
        return next(
            c["status"] for c in document["criteria"] if c["name"] == "Right answers refused (en)"
        )

    assert status(None) == "pass"
    assert status(7) == "pass"
    assert status(8) == "pending"


def test_the_json_keeps_accented_letters_as_written() -> None:
    wanted = case("cafe", behaviour="answer", contains=["caffè"])
    [verdict] = judge([wanted], [record("cafe", "ok")], FIXED_TEXTS, {})
    text = report_json(manifest(), [verdict], budget=None)
    assert text.startswith('{\n  "run_id": "run-1",')
    assert "missing text: caffè" in text
    assert "\\u00e8" not in text


def test_the_invented_fact_criterion_has_nothing_to_measure_before_an_answer_is_given() -> None:
    only_the_alarm = judge([CASES[3]], [RECORDS[3]], FIXED_TEXTS, {})
    found = criteria(summarize(only_the_alarm), spent_eur=None, cap_eur=None)
    invented = next(c for c in found if c.name == "Invented facts that passed the checks")
    assert (invented.value, invented.status) == ("0", "pending")
    answered = judge(CASES[:1], RECORDS[:1], FIXED_TEXTS, {})
    found = criteria(summarize(answered), spent_eur=None, cap_eur=None)
    invented = next(c for c in found if c.name == "Invented facts that passed the checks")
    assert (invented.value, invented.status) == ("0", "pass")
