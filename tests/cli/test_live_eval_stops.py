"""How a live run stops, and what each stop means to whoever launched it (M3 Task 8, rulings 1, 2, 3, 4).

A stop is never a success: every reason the meter can give maps to a non-zero exit code, an unknown
one included, and what the run recorded stays recorded, so that the same command resumes it. The
model here is a function that fails on request; nothing reaches the network.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from frontdesk.cli import live
from frontdesk.live_eval.runner import RunProgress
from frontdesk.model_gateway.costs import Budget, BudgetError
from frontdesk.model_gateway.routes import ROUTES, OutputLimits
from tests.cli.conftest import AccessChecks
from tests.cli.live_chat_helpers import (
    PRICE,
    Failure,
    Install,
    bootstrapped,
    env_file,
    ledger,
)
from tests.cli.live_eval_helpers import (
    eval_live,
    eval_report,
    one_error_line,
    recorded_cases,
    write_cases,
)
from tests.model_gateway.budget_helpers import ROUTE, write_budget

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _no_waiting_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frontdesk.model_gateway.metering.RETRY_WAITS", (0.0, 0.0, 0.0))


def _status(code: int) -> Failure:
    return lambda: ModelHTTPError(code, "gemini-3.1-flash-lite", body={"error": {"code": code}})


def _timeout() -> Failure:
    return lambda: httpx2.ReadTimeout("no answer after the request went out")


def _holds(budget: Budget) -> bool:
    """Whether another run holds the budget now: a second run would be refused."""
    try:
        with budget.exclusive():
            return False
    except BudgetError:
        return True


# --- Each stop has an exit code and says why -------------------------------------------------------


def test_a_rate_limit_stops_the_run_with_exit_3_and_the_same_command_resumes_it(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE, failing={"classifier": _status(429)})
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 2)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 3
    assert stopped.stderr.splitlines() == [
        "stopped by the rate limit: run the same command later to resume"
    ]
    assert "cases recorded: 0/2" in stopped.stdout
    assert sent == ["classifier"] * 4  # four attempts: the retries of design §6
    assert recorded_cases(run_dir) == []
    assert not _holds(budget)

    sent = gemini(PRICE)  # later, when Google answers
    resumed = eval_live(env, budget, run_dir, cases)

    assert resumed.exit_code == 0, resumed.output
    assert "2 cases, 2 turns to run" in resumed.stdout
    assert recorded_cases(run_dir) == ["price-1", "price-2"]
    assert sent == ["classifier"] * 2


def test_a_budget_with_no_room_stops_the_run_with_exit_4_and_says_what_is_left(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    # A request reserves 0.038 USD (32,000 tokens in, a cap of 600 out): the budget has 0.02.
    budget, cases = write_budget(tmp_path, cap_eur=0.02, usd_per_eur=1.0), write_cases(tmp_path, 2)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 4
    assert stopped.stderr.splitlines() == [
        "stopped: the budget of 0.02 EUR has 0.0200 EUR left, and the next call may cost 0.0380 EUR"
    ]
    assert (sent, recorded_cases(run_dir)) == ([], [])
    assert budget.spent_usd() == 0.0
    assert not _holds(budget)

    write_budget(
        tmp_path, cap_eur=10.0, usd_per_eur=1.0
    )  # the cap is raised: same file, same ledger
    resumed = eval_live(env, budget, run_dir, cases)
    assert resumed.exit_code == 0, resumed.output
    assert recorded_cases(run_dir) == ["price-1", "price-2"]


def test_a_request_without_an_output_cap_stops_the_run_and_the_line_names_the_cause(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = gemini(PRICE)
    capless = replace(ROUTES[ROUTE], output_limits=OutputLimits(classifier=0, responder=1200))
    monkeypatch.setitem(ROUTES, ROUTE, capless)
    env = bootstrapped(tmp_path, businesses_dir, database)

    stopped = eval_live(env, write_budget(tmp_path), tmp_path / "live-1", write_cases(tmp_path, 1))

    assert stopped.exit_code == 4
    assert stopped.stderr.splitlines() == [
        "stopped: the request has no output cap: its cost cannot be bounded, so it is not sent"
    ]
    assert sent == []  # a request that cannot be bounded is never sent


def test_a_request_longer_than_the_reserve_assumes_stops_the_run_and_the_line_says_how_long(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = gemini(PRICE)
    monkeypatch.setattr("frontdesk.model_gateway.metering.MAX_INPUT_TOKENS_PER_REQUEST", 100)
    env = bootstrapped(tmp_path, businesses_dir, database)

    stopped = eval_live(env, write_budget(tmp_path), tmp_path / "live-1", write_cases(tmp_path, 1))

    assert stopped.exit_code == 4
    [line] = stopped.stderr.splitlines()
    assert line.startswith("stopped: the request is ")
    assert "bytes long, more than the 100 tokens the reserve assumes" in line
    assert line.endswith("its cost cannot be bounded, so it is not sent")
    assert sent == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only file")
def test_a_ledger_that_stops_taking_lines_stops_the_run_with_exit_1_and_resumes_when_mended(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 2)
    run_dir = tmp_path / "runs" / "live-1"

    def read_only() -> None:
        budget.ledger_path.chmod(0o444)  # while the first request is away

    sent = gemini(PRICE, failing={"classifier": lambda: read_only()})
    try:
        stopped = eval_live(env, budget, run_dir, cases)
    finally:
        budget.ledger_path.chmod(0o644)

    assert stopped.exit_code == 1
    [line] = stopped.stderr.splitlines()
    assert line.startswith("stopped: the ledger can no longer count what is spent (")
    assert f"{budget.ledger_path}: cannot be appended to (Permission denied)" in line
    assert line.endswith("fix it, then run the same command again to resume")
    assert sent == ["classifier"]  # nothing was sent once the ledger could not be written
    assert recorded_cases(run_dir) == []  # the case the stop cut runs again
    assert not _holds(budget)

    sent = gemini(PRICE)
    resumed = eval_live(env, budget, run_dir, cases)
    assert resumed.exit_code == 0, resumed.output
    assert recorded_cases(run_dir) == ["price-1", "price-2"]


@pytest.mark.parametrize(
    ("reason", "code", "line"),
    [
        pytest.param(
            "rate_limited",
            3,
            "stopped by the rate limit: run the same command later to resume",
            id="rate-limited",
        ),
        pytest.param("budget", 4, "stopped: the budget has no room for the next call", id="budget"),
        pytest.param(
            "unavailable",
            5,
            "stopped: the model keeps failing: run the same command later to resume",
            id="unavailable",
        ),
        pytest.param(
            "ledger",
            1,
            "stopped: the ledger can no longer count what is spent: "
            "fix it, then run the same command again to resume",
            id="ledger",
        ),
        pytest.param("mystery", 1, "stopped: mystery", id="a-reason-nobody-knows"),
    ],
)
def test_every_reason_a_run_stops_is_a_non_zero_exit_and_the_cases_still_get_their_reports(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    code: int,
    line: str,
) -> None:
    async def stopped(*args: object, **kwargs: object) -> RunProgress:
        return RunProgress(done=1, total=2, stopped=reason)

    monkeypatch.setattr(live, "run_live_eval", stopped)
    env = bootstrapped(tmp_path, businesses_dir, database)
    run_dir = tmp_path / "runs" / "dry-1"

    result = eval_live(
        env, write_budget(tmp_path), run_dir, write_cases(tmp_path, 2), route="offline-demo"
    )

    assert result.exit_code == code
    assert result.stderr.splitlines() == [line]
    assert "cases recorded: 1/2" in result.stdout  # the run says where it got to
    assert (run_dir / "report.md").exists()  # and writes what it has


@pytest.mark.parametrize(
    ("dropped", "line"),
    [
        pytest.param(
            ("price-1",),
            "1 case ended in a failure just before the stop and is not recorded: "
            "the same command runs it again (price-1)",
            id="one",
        ),
        pytest.param(
            ("a", "b"),
            "2 cases ended in a failure just before the stop and are not recorded: "
            "the same command runs them again (a, b)",
            id="two",
        ),
        pytest.param(
            tuple("abcdefg"),
            "7 cases ended in a failure just before the stop and are not recorded: "
            "the same command runs them again (a, b, c, d, e and 2 more)",
            id="more-than-it-names",
        ),
    ],
)
@pytest.mark.parametrize("reason", ["rate_limited", "budget", "unavailable", "ledger", "mystery"])
def test_whatever_the_reason_the_stop_names_the_cases_it_dropped_on_a_line_of_their_own(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    dropped: tuple[str, ...],
    line: str,
) -> None:
    async def stopped(*args: object, **kwargs: object) -> RunProgress:
        return RunProgress(done=1, total=9, stopped=reason, dropped=dropped)

    monkeypatch.setattr(live, "run_live_eval", stopped)
    env = bootstrapped(tmp_path, businesses_dir, database)

    result = eval_live(
        env,
        write_budget(tmp_path),
        tmp_path / "dry-1",
        write_cases(tmp_path, 2),
        route="offline-demo",
    )

    assert result.exit_code != 0
    first, second = result.stderr.splitlines()
    assert first.startswith("stopped")
    assert second == line  # whatever the reason, and the first line is the stop's own


def test_a_run_that_is_done_exits_0(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def done(*args: object, **kwargs: object) -> RunProgress:
        return RunProgress(done=2, total=2, stopped=None)

    monkeypatch.setattr(live, "run_live_eval", done)
    env = bootstrapped(tmp_path, businesses_dir, database)
    args = (env, write_budget(tmp_path), tmp_path / "dry-1", write_cases(tmp_path, 2))
    assert eval_live(*args, route="offline-demo").exit_code == 0


def test_ctrl_c_ends_the_run_with_130_and_gives_the_budget_back(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini(PRICE)

    async def interrupted(*args: object, **kwargs: object) -> RunProgress:
        raise KeyboardInterrupt

    monkeypatch.setattr(live, "run_live_eval", interrupted)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 1)

    result = eval_live(env, budget, tmp_path / "live-1", cases)

    assert result.exit_code == 130
    assert result.stderr.splitlines() == ["stopped: run the same command again to resume"]
    assert not _holds(budget)


# --- The breaker -----------------------------------------------------------------------------------


def test_a_provider_that_keeps_answering_504_ends_the_run_with_exit_5_at_no_cost_and_it_resumes(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    # A 504 DEADLINE_EXCEEDED settles at zero, so only counting failures of any kind sees it.
    sent = gemini(PRICE, failing={"classifier": _status(504)})
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 3)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 5
    assert stopped.stderr.splitlines() == [
        "stopped: the model keeps failing (4 attempts in a row failed; the last request ended "
        "with HTTP 504): run the same command later to resume"
    ]
    assert "cases recorded: 0/3" in stopped.stdout
    assert sent == ["classifier"] * 4  # one request and its retries, not one case more
    assert recorded_cases(run_dir) == []  # the case that failed runs again
    assert [entry["kind"] for entry in ledger(budget)] == ["preflight"]
    assert budget.spent_usd() == 0.0
    assert not _holds(budget)

    sent = gemini(PRICE)
    resumed = eval_live(env, budget, run_dir, cases)
    assert resumed.exit_code == 0, resumed.output
    assert recorded_cases(run_dir) == ["price-1", "price-2", "price-3"]


def test_requests_that_time_out_are_counted_and_stop_the_run_before_the_cap_is_spent(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE, failing={"classifier": _timeout()})
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path, usd_per_eur=1.0)

    stopped = eval_live(env, budget, tmp_path / "runs" / "live-1", write_cases(tmp_path, 50))

    assert stopped.exit_code == 5
    assert "the last request ended with ReadTimeout" in stopped.stderr
    assert sent == ["classifier"] * 4
    # Each attempt that was sent and never answered is recorded at its reserve (0.038 USD): four
    # of them, not four for each of fifty cases.
    kinds = [entry["kind"] for entry in ledger(budget)]
    assert kinds == ["preflight", *["chat-unconfirmed"] * 4]
    assert budget.spent_usd() == pytest.approx(4 * 0.038)


def test_three_failed_requests_in_a_row_end_the_run_whatever_their_kind(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE, failing={"classifier": _status(403)})  # not retried: one attempt each
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 4)
    run_dir = tmp_path / "runs" / "live-1"

    stopped = eval_live(env, budget, run_dir, cases)

    assert stopped.exit_code == 5
    assert stopped.stderr.splitlines() == [
        "stopped: the model keeps failing (3 attempts in a row failed; the last request ended "
        "with HTTP 403): run the same command later to resume",
        "2 cases ended in a failure just before the stop and are not recorded: "
        "the same command runs them again (price-1, price-2)",
    ]
    assert sent == ["classifier"] * 3
    # The two cases before the third failure ended in a failure that an outage causes: they are
    # held back and dropped with the stop, as is the case the stop cut, so the resume runs all.
    assert recorded_cases(run_dir) == []
    assert not _holds(budget)


def test_a_request_that_fails_three_times_and_then_answers_does_not_stop_the_run(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    calls: list[str] = []

    def three_failures_then_fine() -> BaseException | None:
        calls.append("request")
        return _status(503)() if len(calls) <= 3 else None

    sent = gemini(PRICE, failing={"classifier": three_failures_then_fine})
    env = bootstrapped(tmp_path, businesses_dir, database)

    result = eval_live(env, write_budget(tmp_path), tmp_path / "live-1", write_cases(tmp_path, 2))

    assert result.exit_code == 0, result.output
    assert sent == ["classifier"] * 5  # the first request took four attempts, the second one
    assert "cases recorded: 2/2" in result.stdout


# --- The lock --------------------------------------------------------------------------------------


def test_the_whole_run_holds_the_budget_from_the_first_check_to_the_last_call(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 1)
    held: dict[str, bool] = {}

    def look(moment: str) -> None:
        held[moment] = _holds(budget)

    access_checks.on_call = lambda: look("access check")
    monkeypatch.setattr(
        "frontdesk.cli.live.typer.confirm", lambda *args, **kwargs: look("question") or True
    )
    gemini(PRICE, failing={"classifier": lambda: look("first request")})

    result = eval_live(env, budget, tmp_path / "live-1", cases, yes=False)

    assert result.exit_code == 0, result.output
    assert held == {"access check": True, "question": True, "first request": True}
    assert not _holds(budget)  # and it is given back when the run ends


def test_a_second_live_run_is_refused_before_it_checks_asks_or_writes_anything(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 1)
    run_dir = tmp_path / "runs" / "live-1"

    with budget.exclusive():  # another run is under way
        result = eval_live(env, budget, run_dir, cases)

    assert one_error_line(result) == (
        f"error: {budget.ledger_path}: another live run is using this budget; "
        "run this one when it has ended"
    )
    assert result.stdout == ""
    assert (sent, access_checks.calls) == ([], [])
    assert not run_dir.exists()
    assert not budget.ledger_path.exists()


def test_a_second_live_run_is_refused_before_the_database_is_opened(
    tmp_path: Path, businesses_dir: Path
) -> None:
    # Nothing of the run is opened, not even the pipeline, while another run holds the budget: with
    # the database down the answer is still the budget's, not the database's.
    env = env_file(tmp_path, businesses_dir / "demo-clinic")  # its database leads nowhere
    budget = write_budget(tmp_path)

    with budget.exclusive():
        result = eval_live(env, budget, tmp_path / "run", write_cases(tmp_path, 1))

    assert one_error_line(result).endswith(
        "another live run is using this budget; run this one when it has ended"
    )


def test_a_run_that_fails_to_start_gives_the_budget_back(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    access_checks.error = RuntimeError("the check itself broke")  # not a RouteAccessError

    result = eval_live(env, budget, tmp_path / "live-1", write_cases(tmp_path, 1))

    assert result.exit_code == 1  # an uncaught error, not a stop
    assert isinstance(result.exception, RuntimeError)
    assert not _holds(budget)


# --- A ledger the report cannot read (ruling 4) ------------------------------------------------------


def test_a_ledger_damaged_while_the_run_ends_is_one_error_line_and_not_a_traceback(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 1)
    run_dir = tmp_path / "runs" / "live-1"

    def damage() -> None:
        with budget.ledger_path.open("a", encoding="utf-8") as ledger_file:
            ledger_file.write("not a spend entry\n")

    sent = gemini(PRICE, failing={"classifier": lambda: damage()})

    result = eval_live(env, budget, run_dir, cases)

    line = one_error_line(result)
    assert line.startswith(f"error: {budget.ledger_path}: line ")
    assert line.endswith("is not a spend entry")
    assert sent == ["classifier"]
    assert recorded_cases(run_dir) == ["price-1"]  # the run itself was done and is recorded
    assert not _holds(budget)


def test_the_report_of_a_run_whose_ledger_is_damaged_is_one_error_line(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, run_dir = write_budget(tmp_path), tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, write_cases(tmp_path, 1)).exit_code == 0
    with budget.ledger_path.open("a", encoding="utf-8") as ledger_file:
        ledger_file.write("damaged\n")

    result = eval_report(run_dir)

    line = one_error_line(result)
    assert line.startswith(f"error: {budget.ledger_path}: line ")
    assert line.endswith("is not a spend entry")


def test_the_report_of_a_run_whose_ledger_cannot_be_read_is_one_error_line(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, run_dir = write_budget(tmp_path), tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, write_cases(tmp_path, 1)).exit_code == 0
    budget.ledger_path.unlink()
    budget.ledger_path.mkdir()  # a directory where the ledger was

    result = eval_report(run_dir)

    assert one_error_line(result) == f"error: {budget.ledger_path}: cannot be read (Is a directory)"


def test_a_report_that_cannot_be_written_is_one_error_line(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, run_dir = write_budget(tmp_path), tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, write_cases(tmp_path, 1)).exit_code == 0
    (run_dir / "report.md").unlink()
    (run_dir / "report.md").mkdir()  # a directory where the report goes

    result = eval_report(run_dir)

    assert one_error_line(result) == f"error: {run_dir / 'report.md'}: Is a directory"


def test_a_run_directory_that_cannot_take_the_run_is_one_error_line(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be", encoding="utf-8")
    budget = write_budget(tmp_path)

    result = eval_live(env, budget, blocker / "run", write_cases(tmp_path, 1))

    line = one_error_line(result)
    assert line.startswith(f"error: {blocker / 'run'}")
    assert "Not a directory" in line
    assert not budget.ledger_path.exists()  # refused before the ledger was touched
    assert not _holds(budget)
