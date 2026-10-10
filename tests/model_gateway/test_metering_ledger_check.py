"""A live run proves its ledger before its first paid call (M3 Task 8, ruling 3; Task 6 re-review N3).

A ledger that cannot take a line would let the first paid request through uncounted. The check
reads the ledger, so that a damaged one is found, and appends a line of zero dollars to it, which is
what a paid request does later: it costs nothing and sends no request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from frontdesk.model_gateway.costs import Budget, BudgetError
from frontdesk.model_gateway.metering import Meter
from tests.model_gateway.attempt_helpers import ledger
from tests.model_gateway.budget_helpers import ROUTE, write_budget

NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root can write to any file")


def _meter(budget: Budget) -> Meter:
    return Meter(budget, route_id=ROUTE, run_id="run-1")


def test_a_ledger_that_does_not_exist_yet_is_created_with_a_line_of_zero_dollars(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    assert not budget.ledger_path.exists()
    _meter(budget).check_ledger()
    [line] = budget.ledger_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert set(entry) == {"at", "run", "route", "kind", "usd"}
    assert (entry["run"], entry["route"], entry["kind"], entry["usd"]) == (
        "run-1",
        ROUTE,
        "preflight",
        0.0,
    )


def test_the_line_is_added_after_what_the_ledger_holds_and_adds_nothing_to_it(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=2.5)
    _meter(budget).check_ledger()
    assert ledger(budget) == [("chat", 2.5), ("preflight", 0.0)]
    assert budget.spent_usd() == 2.5


def test_the_check_is_not_a_request_and_not_a_call_of_the_run(tmp_path: Path) -> None:
    meter = _meter(write_budget(tmp_path))
    meter.check_ledger()
    assert meter.drain() == ()  # no call to report
    assert meter.held_usd == 0.0
    assert meter.stop_reason is None
    assert meter.consecutive_failures == {}


def test_each_check_adds_its_own_line(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    for _ in range(2):
        _meter(budget).check_ledger()
    assert ledger(budget) == [("preflight", 0.0)] * 2


def test_a_damaged_ledger_is_named_before_anything_is_written(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text('{"usd": 1}\nnot json\n', encoding="utf-8")
    meter = _meter(budget)
    with pytest.raises(BudgetError, match=r"spend\.jsonl: line 2 is not a spend entry"):
        meter.check_ledger()
    assert budget.ledger_path.read_text(encoding="utf-8") == '{"usd": 1}\nnot json\n'
    assert meter.stop_reason == "ledger"  # a run that proves nothing does not start


def test_a_ledger_that_is_a_directory_is_named(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()
    meter = _meter(budget)
    with pytest.raises(BudgetError, match=r"spend\.jsonl: cannot be read \(Is a directory\)"):
        meter.check_ledger()
    assert meter.stop_reason == "ledger"


@NOT_ROOT
def test_a_ledger_nobody_may_read_is_named(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("", encoding="utf-8")
    budget.ledger_path.chmod(0o000)
    try:
        with pytest.raises(BudgetError, match=r"cannot be read \(Permission denied\)"):
            _meter(budget).check_ledger()
    finally:
        budget.ledger_path.chmod(0o600)


@NOT_ROOT
def test_a_ledger_that_may_be_read_but_not_appended_to_is_named_and_left_as_it_was(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=1.0)
    budget.ledger_path.chmod(0o444)
    meter = _meter(budget)
    try:
        with pytest.raises(BudgetError, match=r"cannot be appended to \(Permission denied\)"):
            meter.check_ledger()
    finally:
        budget.ledger_path.chmod(0o600)
    assert ledger(budget) == [("chat", 1.0)]
    assert meter.stop_reason == "ledger"


def test_a_ledger_whose_directory_is_missing_cannot_be_appended_to(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    nowhere = Budget(tmp_path / "gone" / "budget.yaml", budget.config)
    with pytest.raises(BudgetError, match=r"cannot be appended to \(No such file or directory\)"):
        _meter(nowhere).check_ledger()


def test_a_line_that_cannot_be_written_is_a_budget_error_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    budget = write_budget(tmp_path)
    monkeypatch.setattr(Budget, "record", disk_full)
    meter = _meter(budget)
    with pytest.raises(BudgetError, match=r"cannot be appended to \(No space left on device\)"):
        meter.check_ledger()
    assert meter.stop_reason == "ledger"
    assert (
        meter.stop_detail
        == f"{budget.ledger_path}: cannot be appended to (No space left on device)"
    )
