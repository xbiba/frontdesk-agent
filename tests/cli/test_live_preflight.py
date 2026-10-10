"""What `eval turns --live` and `chat --live` check before their first paid call (M3 Task 8, ruling 3
and 5): the environment, the free request that proves the credentials, the API and the model, and
a ledger that takes a line. Both commands share the checks, and a command that fails one has
spent nothing, written nothing and given the budget back.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import Result

from frontdesk.model_gateway.access import ENVIRONMENT_OVERRIDES, RouteAccessError
from frontdesk.model_gateway.costs import Budget, load_budget
from frontdesk.model_gateway.routes import UnapprovedRouteError
from tests.cli.conftest import AccessChecks
from tests.cli.live_chat_helpers import (
    PRICE,
    QUESTION,
    Install,
    bootstrapped,
    env_file,
    ledger,
    live,
    refused,
)
from tests.cli.live_eval_helpers import eval_live, one_error_line, write_cases
from tests.model_gateway.budget_helpers import ROUTE, write_budget

LOCK = "spend.jsonl.lock"
# The two commands, each started the same way: (env file, budget, directory for a run's files).
Start = Callable[[Path, Budget, Path], Result]


def _chat(env: Path, budget: Budget, directory: Path) -> Result:
    return live(env, budget, input=f"{QUESTION}\nexit\n")


def _evaluation(env: Path, budget: Budget, directory: Path) -> Result:
    return eval_live(env, budget, directory / "run", write_cases(directory, 1))


COMMANDS = [
    pytest.param(_chat, id="chat --live"),
    pytest.param(_evaluation, id="eval turns --live"),
]


# --- The environment -------------------------------------------------------------------------------


@pytest.mark.parametrize("start", COMMANDS)
@pytest.mark.parametrize("name", ENVIRONMENT_OVERRIDES)
def test_a_variable_that_moves_a_request_or_its_billing_stops_a_live_command_cold(
    start: Start,
    name: str,
    tmp_path: Path,
    businesses_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    access_checks: AccessChecks,
) -> None:
    monkeypatch.setenv(name, "set-by-the-shell")
    env = env_file(tmp_path, businesses_dir / "demo-clinic")  # no database: nothing was opened
    budget = write_budget(tmp_path)

    result = start(env, budget, tmp_path)

    assert refused(result) == (
        "error: a live run does not go through proxies, other certificates or another billing "
        f"project: unset {name}"
    )
    assert access_checks.calls == []
    assert not budget.ledger_path.exists()
    assert not (tmp_path / LOCK).exists()  # it did not even take the budget
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("start", COMMANDS)
def test_every_variable_that_is_set_is_named_in_either_case(
    start: Start, tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("https_proxy", "http://proxy.stray.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/stray/ca.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "")  # set to nothing: not an override
    env = env_file(tmp_path, businesses_dir / "demo-clinic")

    result = start(env, write_budget(tmp_path), tmp_path)

    assert refused(result).endswith("unset HTTPS_PROXY, SSL_CERT_FILE")


@pytest.mark.db
def test_a_dry_run_on_the_offline_route_goes_through_a_proxy_for_nothing_so_it_is_not_refused(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    access_checks: AccessChecks,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.stray.example:3128")
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)

    result = eval_live(
        env, budget, tmp_path / "dry", write_cases(tmp_path, 1), route="offline-demo"
    )

    assert result.exit_code == 0, result.output
    assert access_checks.calls == []  # nothing to prove: the route calls no model
    assert not budget.ledger_path.exists()  # and nothing to count


# --- The free request that proves the route ---------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("start", COMMANDS)
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            RouteAccessError("Google credentials are missing or expired: run `gcloud ...`"),
            id="credentials",
        ),
        pytest.param(RouteAccessError("model 'x' is not served at https://h"), id="model"),
        pytest.param(UnapprovedRouteError("model route 'x' has no pinned endpoint"), id="route"),
    ],
)
def test_a_route_that_fails_the_free_request_stops_a_live_command_before_it_spends_anything(
    start: Start,
    error: Exception,
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    access_checks.error = error
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)

    result = start(env, budget, tmp_path)

    assert refused(result) == f"error: {error}"
    assert access_checks.calls == [(ROUTE, "demo-project")]
    assert sent == []
    assert not budget.ledger_path.exists()  # not even the proof: the check comes first
    assert not (
        tmp_path / "run"
    ).exists()  # an evaluation claims no directory for a run that failed
    with budget.exclusive():
        pass


@pytest.mark.db
@pytest.mark.parametrize("start", COMMANDS)
def test_the_free_request_is_made_for_the_route_and_the_project_before_the_first_paid_request(
    start: Start,
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    requests_when_checked: list[int] = []
    sent = gemini(PRICE)
    access_checks.on_call = lambda: requests_when_checked.append(len(sent))

    result = start(env, budget, tmp_path)

    assert result.exit_code == 0, result.output
    assert access_checks.calls == [(ROUTE, "demo-project")]
    assert requests_when_checked == [0]  # nothing had been sent when the route was proved
    assert sent == ["classifier"]


# --- A ledger that takes a line ----------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("start", COMMANDS)
def test_the_ledger_has_taken_its_proof_by_the_time_the_first_paid_request_goes_out(
    start: Start, tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    seen: list[list[str]] = []
    gemini(PRICE, failing={"classifier": lambda: seen.append(_kinds(budget))})

    result = start(env, budget, tmp_path)

    assert result.exit_code == 0, result.output
    assert seen == [["preflight"]]  # a line of zero dollars, and nothing paid yet
    assert _kinds(budget) == ["preflight", "chat"]
    assert ledger(budget)[0]["usd"] == 0.0


def _kinds(budget: Budget) -> list[str]:
    return [entry["kind"] for entry in ledger(budget)]


@pytest.mark.db
@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only file")
@pytest.mark.parametrize("start", COMMANDS)
def test_a_ledger_that_can_be_read_but_not_appended_to_stops_a_live_command_before_it_spends(
    start: Start,
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=1.0)
    budget.ledger_path.chmod(0o444)
    try:
        result = start(env, budget, tmp_path)
    finally:
        budget.ledger_path.chmod(0o644)

    # An evaluation has shown its plan by then; a chat has not shown its banner.
    assert one_error_line(result) == (
        f"error: {budget.ledger_path}: cannot be appended to (Permission denied)"
    )
    assert "Live:" not in result.stdout
    assert sent == []  # not one request went out on a ledger that could not have counted it
    # And the ledger holds only what it held.
    assert [(e["run"], e["kind"], e["usd"]) for e in ledger(budget)] == [("earlier", "chat", 1.0)]
    assert not (tmp_path / "run").exists()
    with budget.exclusive():
        pass


@pytest.mark.db
@pytest.mark.parametrize("start", COMMANDS)
def test_a_damaged_ledger_stops_a_live_command_before_it_spends(
    start: Start, tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("this is not a spend entry\n", encoding="utf-8")

    result = start(env, budget, tmp_path)

    assert refused(result) == f"error: {budget.ledger_path}: line 1 is not a spend entry"
    assert sent == []
    assert budget.ledger_path.read_text(encoding="utf-8") == "this is not a spend entry\n"


@pytest.mark.db
@pytest.mark.parametrize("start", COMMANDS)
def test_a_ledger_that_is_a_directory_stops_a_live_command_before_it_spends(
    start: Start, tmp_path: Path, businesses_dir: Path, database: tuple[str, str], gemini: Install
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()

    result = start(env, budget, tmp_path)

    assert refused(result) == f"error: {budget.ledger_path}: cannot be read (Is a directory)"
    assert sent == []


@pytest.mark.parametrize("start", COMMANDS)
def test_a_ledger_in_a_directory_that_does_not_exist_is_an_error_line(
    start: Start, tmp_path: Path, businesses_dir: Path
) -> None:
    # The lock lives beside the ledger, so the budget cannot even be taken.
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    path = write_budget(tmp_path).path
    path.write_text(
        path.read_text(encoding="utf-8").replace("ledger: spend.jsonl", "ledger: gone/spend.jsonl"),
        encoding="utf-8",
    )

    result = start(env, load_budget(path), tmp_path)

    line = refused(result)
    assert line.startswith(f"error: {tmp_path / 'gone' / 'spend.jsonl.lock'}: cannot be opened")
    assert line.endswith("the ledger's directory must exist")
