"""An outage leaves no failed case in the evidence (M3 Task 8, fix round 1, review I-1, probes P3 and
P4): the cases that ended in a failure just before the run stopped are not recorded, the stop line
names them, and a resume runs them again, so that `report.md` counts only what the provider
answered. A failure the provider recovered from is the case's own, and stays.

The provider is a function that can be kept away: the turn limit (shortened here) then cuts the
request. Nothing reaches a model or the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx2
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from frontdesk.conversation.turn import TURN_TIMEOUT_SECONDS
from tests.cli.live_chat_helpers import PARKING, PRICE, Install, bootstrapped
from tests.cli.live_eval_helpers import (
    eval_live,
    failures_in,
    install_provider,
    quick_turns,
    recorded_cases,
    rules_of,
    write_cases,
    write_parking_cases,
)
from tests.model_gateway.budget_helpers import write_budget

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _no_waiting_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frontdesk.model_gateway.metering.RETRY_WAITS", (0.0, 0.0, 0.0))


async def _hang() -> None:
    await asyncio.sleep(30)


def test_a_provider_that_hangs_leaves_no_timeout_in_the_evidence_after_the_resume(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Probe P3: every request hangs and the turn limit cuts it. The first case ends as a timeout
    # after one failed attempt, the second after two, and the third request ends the run.
    sent = install_provider(monkeypatch, PRICE, lambda role, number: _hang())
    limit = quick_turns(monkeypatch)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path, usd_per_eur=1.0), write_cases(tmp_path, 5)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 5
    assert sent == ["classifier"] * 3
    assert recorded_cases(run_dir) == []  # not one timeout is in the evidence
    first, second = stopped.stderr.splitlines()
    assert first.startswith("stopped: the model keeps failing")
    assert second == (
        "2 cases ended in a failure just before the stop and are not recorded: "
        "the same command runs them again (price-1, price-2)"
    )

    sent = gemini(PRICE)  # the provider answers again
    limit.seconds = TURN_TIMEOUT_SECONDS
    resumed = eval_live(env, budget, run_dir, cases)

    assert resumed.exit_code == 0, resumed.output
    assert recorded_cases(run_dir) == [f"price-{number}" for number in range(1, 6)]
    assert sent == ["classifier"] * 5  # price-1 and price-2 ran again, in full
    assert set(rules_of(run_dir)) == {"pass"}
    assert failures_in(run_dir) == 0
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "FAIL" not in report
    assert "timeout" not in report


def test_resets_then_timeouts_leave_no_failed_case_in_the_evidence_after_the_resume(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Probe P4: two requests reset (not retried: two cases end as errors), then four timeouts.
    def behave(role: str, number: int) -> None:
        if number <= 2:
            raise httpx2.ReadError("connection reset by peer")
        raise httpx2.ReadTimeout("no answer after the request went out")

    sent = install_provider(monkeypatch, PRICE, behave)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path, usd_per_eur=1.0), write_cases(tmp_path, 4)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 5
    assert len(sent) == 2 + 4  # two resets, then one request and its three retries
    assert recorded_cases(run_dir) == []
    assert stopped.stderr.splitlines()[1] == (
        "2 cases ended in a failure just before the stop and are not recorded: "
        "the same command runs them again (price-1, price-2)"
    )

    gemini(PRICE)
    resumed = eval_live(env, budget, run_dir, cases)

    assert resumed.exit_code == 0, resumed.output
    assert recorded_cases(run_dir) == [f"price-{number}" for number in range(1, 5)]
    assert set(rules_of(run_dir)) == {"pass"}
    assert failures_in(run_dir) == 0


def test_a_responder_that_hangs_while_the_classifier_answers_ends_the_run(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Probe P5: the classifier answers every request, so a count shared by both agents never
    # reaches three. Each agent has its own, and the responder's failures in a row end the run.
    sent = install_provider(
        monkeypatch, PARKING, lambda role, number: _hang() if role == "responder" else None
    )
    limit = quick_turns(monkeypatch)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path, usd_per_eur=1.0), write_parking_cases(tmp_path, 6)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 5
    assert sent == ["classifier", "responder"] * 3
    assert recorded_cases(run_dir) == []
    assert stopped.stderr.splitlines() == [
        "stopped: the model keeps failing (3 attempts of the responder in a row failed; the last "
        "request ended with CancelledError): run the same command later to resume",
        "2 cases ended in a failure just before the stop and are not recorded: "
        "the same command runs them again (parking-1, parking-2)",
    ]

    sent = gemini(PARKING)
    limit.seconds = TURN_TIMEOUT_SECONDS
    resumed = eval_live(env, budget, run_dir, cases)

    assert resumed.exit_code == 0, resumed.output
    assert recorded_cases(run_dir) == [f"parking-{number}" for number in range(1, 7)]
    assert set(rules_of(run_dir)) == {"pass"}
    assert failures_in(run_dir) == 0


def test_a_failure_the_provider_recovered_from_is_the_cases_own_and_stays_in_the_evidence(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def behave(role: str, number: int) -> None:
        if number == 1:  # Google refuses the first request; the next ones are answered
            raise ModelHTTPError(403, "gemini-3.1-flash-lite", body={"error": {"code": 403}})

    install_provider(monkeypatch, PRICE, behave)
    env = bootstrapped(tmp_path, businesses_dir, database)
    run_dir = tmp_path / "runs" / "live-1"

    result = eval_live(env, write_budget(tmp_path), run_dir, write_cases(tmp_path, 3))

    assert result.exit_code == 0, result.output
    assert recorded_cases(run_dir) == ["price-1", "price-2", "price-3"]
    lines = (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["rejection"] == "classifier_error"
    assert failures_in(run_dir) == 1  # and the report counts it, as it should


def test_a_failure_at_the_end_of_a_run_that_nothing_stopped_is_written(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def behave(role: str, number: int) -> None:
        if number == 3:  # the last case
            raise httpx2.ReadError("connection reset by peer")

    install_provider(monkeypatch, PRICE, behave)
    env = bootstrapped(tmp_path, businesses_dir, database)
    run_dir = tmp_path / "runs" / "live-1"

    result = eval_live(env, write_budget(tmp_path), run_dir, write_cases(tmp_path, 3))

    assert result.exit_code == 0, result.output
    assert rules_of(run_dir) == ["pass", "pass", "error"]
    assert "cases recorded: 3/3" in result.stdout


def test_a_stop_that_dropped_nothing_says_nothing_of_it(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_provider(monkeypatch, PRICE, lambda role, number: _status_429())
    env = bootstrapped(tmp_path, businesses_dir, database)

    stopped = eval_live(env, write_budget(tmp_path), tmp_path / "live-1", write_cases(tmp_path, 2))

    assert stopped.exit_code == 3
    assert stopped.stderr.splitlines() == [
        "stopped by the rate limit: run the same command later to resume"
    ]


def _status_429() -> None:
    raise ModelHTTPError(429, "gemini-3.1-flash-lite", body={"error": {"code": 429}})
