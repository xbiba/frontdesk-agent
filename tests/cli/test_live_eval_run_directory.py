"""A run directory that cannot take the run refuses it before the ledger is touched (M3 Task 8, fix
round 1, review m2, probes P10 and P11).

The run claims its directory (it writes `run.json` if the run is new, and opens `turns.jsonl` for
appending) before it proves the ledger with a line of zero dollars: a refused run leaves no ledger
line, and a run that could not record what it pays for does not start.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from frontdesk.model_gateway.metering import Meter
from tests.cli.conftest import AccessChecks
from tests.cli.live_chat_helpers import PRICE, Install, bootstrapped, ledger
from tests.cli.live_eval_helpers import eval_live, one_error_line, recorded_cases, write_cases
from tests.model_gateway.budget_helpers import write_budget

pytestmark = pytest.mark.db
NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")


def test_a_run_directory_with_a_file_in_the_way_leaves_no_ledger_line(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be", encoding="utf-8")
    budget = write_budget(tmp_path)

    result = eval_live(env, budget, blocker / "run", write_cases(tmp_path, 1))

    assert one_error_line(result).startswith(f"error: {blocker / 'run'}")
    assert sent == []
    assert not budget.ledger_path.exists()  # the proof was not appended: the run was refused first
    with budget.exclusive():
        pass


@NOT_ROOT
@pytest.mark.parametrize("exists", [False, True], ids=["to-create", "to-write-in"])
def test_a_run_directory_that_may_not_be_written_leaves_no_ledger_line(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    exists: bool,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    parent = tmp_path / "read-only"
    parent.mkdir()
    run_dir = parent / "run"
    if exists:
        run_dir.mkdir()
        run_dir.chmod(0o555)
    parent.chmod(0o555)
    budget = write_budget(tmp_path)
    try:
        result = eval_live(env, budget, run_dir, write_cases(tmp_path, 1))
    finally:
        parent.chmod(0o755)
        if exists:
            run_dir.chmod(0o755)

    line = one_error_line(result)
    assert line.startswith(f"error: {run_dir}")
    assert line.endswith("Permission denied")
    assert sent == []
    assert not budget.ledger_path.exists()
    with budget.exclusive():
        pass


@NOT_ROOT
def test_a_run_that_cannot_record_is_refused_before_it_pays_for_anything(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 3)
    run_dir = tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, cases, "--limit", "1").exit_code == 0
    requests, lines = len(sent), ledger(budget)
    turns = run_dir / "turns.jsonl"
    turns.chmod(0o444)  # the first case is recorded; nothing more can be
    try:
        result = eval_live(env, budget, run_dir, cases)
    finally:
        turns.chmod(0o644)

    assert one_error_line(result) == f"error: {turns}: Permission denied"
    assert len(sent) == requests  # no request was sent for a case that could not be written down
    assert ledger(budget) == lines  # and the ledger holds what it held, the proof included
    assert recorded_cases(run_dir) == ["price-1"]


def test_a_run_that_starts_has_claimed_its_directory_before_the_ledger_proof(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini(PRICE)
    seen: list[tuple[bool, bool]] = []
    real = Meter.check_ledger

    def check_ledger(self: Meter) -> None:
        seen.append(((run_dir / "run.json").exists(), (run_dir / "turns.jsonl").exists()))
        real(self)

    monkeypatch.setattr(Meter, "check_ledger", check_ledger)
    env = bootstrapped(tmp_path, businesses_dir, database)
    run_dir = tmp_path / "runs" / "live-1"

    result = eval_live(env, write_budget(tmp_path), run_dir, write_cases(tmp_path, 1))

    assert result.exit_code == 0, result.output
    assert seen == [(True, True)]  # the manifest was written and the record file was open


def test_a_resumed_run_keeps_the_manifest_it_was_started_with(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 2)
    run_dir = tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, cases, "--limit", "1").exit_code == 0
    manifest = (run_dir / "run.json").read_bytes()

    assert eval_live(env, budget, run_dir, cases).exit_code == 0

    assert (run_dir / "run.json").read_bytes() == manifest  # the same run, as it began
    assert recorded_cases(run_dir) == ["price-1", "price-2"]


@NOT_ROOT
def test_a_run_refused_by_its_ledger_is_resumed_by_the_same_command_once_it_is_mended(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 1)
    budget.record(run_id="earlier", route_id="vertex-eu-gemini-flash-lite", kind="chat", usd=1.0)
    budget.ledger_path.chmod(0o444)
    run_dir = tmp_path / "runs" / "live-1"
    try:
        refused = eval_live(env, budget, run_dir, cases)
    finally:
        budget.ledger_path.chmod(0o644)
    assert refused.exit_code == 1
    claimed = (run_dir / "run.json").read_bytes()  # the directory was claimed before the proof
    assert recorded_cases(run_dir) == []

    mended = eval_live(env, budget, run_dir, cases)

    assert mended.exit_code == 0, mended.output
    assert (run_dir / "run.json").read_bytes() == claimed  # the same run, not another
    assert [line["kind"] for line in ledger(budget)] == ["chat", "preflight", "chat"]
    assert recorded_cases(run_dir) == ["price-1"]
