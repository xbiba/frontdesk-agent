"""A conversation of several turns is one case and several turns (Task 7 review, M3, M4).

Cases count cases; time and cost count turns; the verdict reads its last reply and sums every turn.
Every other test of the report is single-turn, so there a turn and a case could be taken for one
another without a test noticing. The quantile of the displayed 95th percentile needs enough values
to tell the 90th, the 95th and the largest apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frontdesk.conversation.behaviour import Behaviour
from frontdesk.live_eval.judging import CaseVerdict, judge
from frontdesk.live_eval.report import agent_stats, render_report, summarize
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, call, case, manifest, record
from tests.model_gateway.budget_helpers import write_budget

TWO = case("two", behaviour="answer", contains=["ok"]).model_copy(
    update={"message": None, "messages": ("uno", "due")}
)
# Turn 1 is a handoff, turn 2 an answer: the case is judged on the last reply.
TURNS = [
    record(
        "two",
        CONTACT,
        turn=1,
        turns=2,
        outcome="handoff",
        seconds=1.0,
        calls=(call(usd=0.01),),
    ),
    record("two", "ok", turn=2, turns=2, seconds=9.0, calls=(call(usd=0.02),)),
]


@pytest.fixture
def verdict() -> CaseVerdict:
    [only] = judge([TWO], TURNS, FIXED_TEXTS, {})
    return only


def test_the_verdict_reads_its_last_reply_and_sums_every_turn(verdict: CaseVerdict) -> None:
    assert verdict.observed is Behaviour.ANSWER  # not the handoff of turn 1
    assert verdict.language == "it"
    assert verdict.seconds == (1.0, 9.0)
    assert verdict.usd == pytest.approx(0.03)  # both turns, not the last


def test_time_counts_turns_and_cases_count_cases(verdict: CaseVerdict) -> None:
    summary = summarize([verdict])["it"]
    assert (summary.cases, summary.turns) == (1, 2)
    assert summary.within_limit == pytest.approx(0.5)  # 1 of the 2 turns is within 8 seconds
    assert (summary.p50_seconds, summary.p95_seconds) == (1.0, 9.0)
    assert summary.usd == pytest.approx(0.03)


def test_the_cost_per_message_is_over_turns_not_cases(verdict: CaseVerdict, tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.25)
    text = render_report(manifest(), [verdict], total_cases=1, budget=budget)
    [row] = [line for line in text.splitlines() if line.startswith("| it ")]
    cells = row.split(" | ")
    assert cells[1] == "1"  # one case
    assert cells[13] == "0.01200"  # 0.03 USD = 0.024 EUR over 2 turns
    assert cells[14] == "0.0240 |"


# --- the quantile of the displayed 95th percentile ---------------------------------------------


def test_the_displayed_percentiles_are_the_95th_and_the_median() -> None:
    # 20 turns of 1 to 20 seconds: the 95th percentile is the 19th, not the 18th (90th) or the 20th.
    cases = [case(f"c{n}", behaviour="answer") for n in range(20)]
    records = [record(f"c{n}", "ok", seconds=float(n + 1)) for n in range(20)]
    summary = summarize(judge(cases, records, FIXED_TEXTS, {}))["it"]
    assert summary.p95_seconds == 19.0
    assert summary.p50_seconds == 10.0


def test_the_output_tokens_of_an_agent_are_summed_at_the_95th_percentile() -> None:
    cases = [case(f"c{n}", behaviour="answer") for n in range(20)]
    records = [
        record(f"c{n}", "ok", calls=(call("responder", output_tokens=n + 1),)) for n in range(20)
    ]
    [responder] = agent_stats(judge(cases, records, FIXED_TEXTS, {}))
    assert responder.requests == 20
    assert responder.max_output_tokens == 20
    assert responder.p95_output_tokens == 19
