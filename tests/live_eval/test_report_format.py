"""The report's percentages and table cells (Task 7 review, M8).

A percentage that decides a verdict must not be rounded across its threshold: 17 of 19 replies within
10 seconds is 89.5%, below the 90% that is asked, and must not read 90%. A whole percentage keeps its
short form ("20%", "0%"). A text from a dataset may hold a pipe, which must not split a table row.
"""

from __future__ import annotations

import re

import pytest

from frontdesk.live_eval.judging import CaseVerdict, judge
from frontdesk.live_eval.report import criteria, render_report, summarize
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, manifest, record


def within_ten_seconds(slow: int, total: int) -> tuple[CaseVerdict, ...]:
    cases = [case(f"c{n}", behaviour="answer") for n in range(total)]
    records = [record(f"c{n}", "ok", seconds=(12.0 if n < slow else 1.0)) for n in range(total)]
    return judge(cases, records, FIXED_TEXTS, {})


def test_17_of_19_replies_within_10_seconds_is_not_shown_as_90_percent() -> None:
    verdicts = within_ten_seconds(slow=2, total=19)
    found = criteria(summarize(verdicts), spent_eur=None, cap_eur=None)
    [reply_time] = [c for c in found if c.name == "Replies within 10 seconds"]
    assert (reply_time.value, reply_time.status) == ("89.5% (p95 12.0 s)", "fail")
    assert reply_time.threshold == "at least 90%"


def test_the_language_table_shows_the_share_within_10_seconds_to_one_decimal() -> None:
    text = render_report(
        manifest(), within_ten_seconds(slow=2, total=19), total_cases=19, budget=None
    )
    [row] = [line for line in text.splitlines() if line.startswith("| it ")]
    assert row.split(" | ")[12] == "89.5%"


@pytest.mark.parametrize(
    ("lost", "total", "shown"),
    [
        pytest.param(0, 5, "0%", id="none"),
        pytest.param(1, 5, "20%", id="a-whole-percentage-keeps-its-short-form"),
        pytest.param(1, 3, "33.3%", id="a-third"),
        pytest.param(1, 15, "6.7%", id="a-fifteenth"),
        pytest.param(5, 5, "100%", id="all"),
    ],
)
def test_the_rate_of_right_answers_lost_shows_one_decimal_when_it_has_one(
    lost: int, total: int, shown: str
) -> None:
    cases = [
        case(f"d{n}", behaviour="answer", contains=["six"], sources=["parking-en"])
        for n in range(total)
    ]
    records = [
        record(f"d{n}", "There are six free spaces.", sources=("parking-en",))
        for n in range(total - lost)
    ] + [
        record(f"d{n}", f"{TEXTS['no_information']}\n\n{CONTACT}", outcome="handoff")
        for n in range(total - lost, total)
    ]
    verdicts = judge(cases, records, FIXED_TEXTS, {})
    found = criteria(summarize(verdicts), spent_eur=None, cap_eur=None)
    [refused] = [c for c in found if c.name == "Right answers refused (it)"]
    assert refused.value.startswith(f"{shown} (")


# --- a pipe in a text --------------------------------------------------------------------------


def test_a_pipe_in_a_text_of_the_dataset_does_not_split_the_case_row() -> None:
    wanted = [case("pipe", behaviour="answer", contains=["Mon|Fri"])]
    text = render_report(
        manifest(),
        judge(wanted, [record("pipe", "ok")], FIXED_TEXTS, {}),
        total_cases=1,
        budget=None,
    )
    [row] = [line for line in text.splitlines() if line.startswith("| pipe ")]
    assert row.endswith("| FAIL: missing text: Mon\\|Fri |")
    assert len(re.split(r"(?<!\\)\|", row)) == 9  # seven cells between the outer pipes


def test_a_row_without_a_pipe_in_its_text_is_unchanged() -> None:
    wanted = [case("plain", behaviour="answer", contains=["Mon-Fri"])]
    text = render_report(
        manifest(),
        judge(wanted, [record("plain", "ok")], FIXED_TEXTS, {}),
        total_cases=1,
        budget=None,
    )
    assert "| plain | it |  | answer | answer |  | FAIL: missing text: Mon-Fri |" in text
