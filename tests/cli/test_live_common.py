"""The helpers `chat --live` and `eval turns --live` share, without a command around them (Task 8)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import typer

from frontdesk.cli.live_common import (
    check_access,
    eur_left,
    prove_ledger,
    refuse_environment_overrides,
    spent_usd,
)
from frontdesk.model_gateway.access import RouteAccessError
from frontdesk.model_gateway.costs import Budget, BudgetError
from frontdesk.model_gateway.routes import ROUTES
from tests.cli.conftest import AccessChecks
from tests.model_gateway.budget_helpers import ROUTE, write_budget

GEMINI = ROUTES[ROUTE]


def test_an_environment_without_overrides_is_accepted() -> None:
    refuse_environment_overrides({"PATH": "/bin", "NO_PROXY": "localhost"})


def test_the_process_environment_is_what_is_checked_unless_another_is_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SSL_CERT_DIR", "/etc/ssl/stray")
    with pytest.raises(typer.Exit) as raised:
        refuse_environment_overrides()
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err.endswith("unset SSL_CERT_DIR\n")
    refuse_environment_overrides({})  # an environment that is given is the one that counts


@pytest.mark.parametrize("project", [None, "", "  "])
def test_a_route_is_not_proved_without_a_project(
    project: str | None, access_checks: AccessChecks, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(typer.Exit) as raised:
        check_access(GEMINI, project)
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err == (
        "error: FRONTDESK_GCP_PROJECT is required for the vertex-eu-gemini-flash-lite route\n"
    )
    assert access_checks.calls == []


def test_the_route_is_proved_for_the_project_it_is_given(access_checks: AccessChecks) -> None:
    check_access(GEMINI, "demo-clinic")
    assert access_checks.calls == [(ROUTE, "demo-clinic")]


def test_a_route_that_fails_the_proof_is_the_error_line_the_check_wrote(
    access_checks: AccessChecks, capsys: pytest.CaptureFixture[str]
) -> None:
    access_checks.error = RouteAccessError(
        "Vertex AI is not answering (503 UNAVAILABLE): try again"
    )
    with pytest.raises(typer.Exit) as raised:
        check_access(GEMINI, "demo-clinic")
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err == (
        "error: Vertex AI is not answering (503 UNAVAILABLE): try again\n"
    )


def test_what_the_ledger_holds_is_what_was_recorded(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    assert spent_usd(budget) == 0.0  # no ledger yet: nothing spent
    budget.record(run_id="run", route_id=ROUTE, kind="chat", usd=1.5)
    budget.record(run_id="run", route_id=ROUTE, kind="chat", usd=0.25)
    assert spent_usd(budget) == 1.75


def test_a_ledger_that_is_a_directory_is_a_budget_error_that_names_it(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()
    with pytest.raises(BudgetError, match=r"spend\.jsonl: cannot be read \(Is a directory\)"):
        spent_usd(budget)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file whatever its mode")
def test_a_ledger_nobody_may_read_is_a_budget_error_that_names_it(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("", encoding="utf-8")
    budget.ledger_path.chmod(0o000)
    try:
        with pytest.raises(BudgetError, match=r"cannot be read \(Permission denied\)"):
            spent_usd(budget)
    finally:
        budget.ledger_path.chmod(0o600)


def test_a_ledger_that_is_damaged_is_the_budgets_own_error(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("nonsense\n", encoding="utf-8")
    with pytest.raises(BudgetError, match="line 1 is not a spend entry"):
        spent_usd(budget)


def test_what_is_left_is_the_cap_less_what_is_spent_in_euros_and_never_below_zero(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)  # a cap of 12.50 USD
    assert eur_left(budget, 0.0) == pytest.approx(10.0)
    assert eur_left(budget, 2.5) == pytest.approx(8.0)
    assert eur_left(budget, 12.5) == 0.0
    assert eur_left(budget, 20.0) == 0.0


# --- The proof that the ledger takes a line, for the commands that have no meter -------------------
# The embedder experiment cannot use `Meter.check_ledger` (a meter prices tokens), and makes the
# same proof before its first paid text: read the ledger, then append a line of zero dollars.

NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root can write to any file")


def test_a_ledger_that_does_not_exist_yet_is_created_with_a_line_of_zero_dollars(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    prove_ledger(budget, run_id="run-1", route_id=ROUTE)
    [line] = budget.ledger_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert (entry["run"], entry["route"], entry["kind"], entry["usd"]) == (
        "run-1",
        ROUTE,
        "preflight",
        0.0,
    )


def test_the_line_comes_after_what_the_ledger_holds_and_adds_nothing_to_it(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=2.5)
    prove_ledger(budget, run_id="run-1", route_id=ROUTE)
    kinds = [json.loads(line)["kind"] for line in budget.ledger_path.read_text().splitlines()]
    assert kinds == ["chat", "preflight"]
    assert budget.spent_usd() == 2.5


def test_a_damaged_ledger_is_named_before_anything_is_written(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text('{"usd": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(BudgetError, match=r"spend\.jsonl: line 2 is not a spend entry"):
        prove_ledger(budget, run_id="run-1", route_id=ROUTE)
    assert budget.ledger_path.read_text(encoding="utf-8") == '{"usd": 1}\nnot json\n'


def test_a_ledger_that_is_a_directory_is_named(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()
    with pytest.raises(BudgetError, match=r"spend\.jsonl: cannot be read \(Is a directory\)"):
        prove_ledger(budget, run_id="run-1", route_id=ROUTE)


@NOT_ROOT
def test_a_ledger_that_may_be_read_but_not_appended_to_is_named_and_left_as_it_was(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=1.0)
    before = budget.ledger_path.read_text(encoding="utf-8")
    budget.ledger_path.chmod(0o444)
    try:
        with pytest.raises(BudgetError, match=r"cannot be appended to \(Permission denied\)"):
            prove_ledger(budget, run_id="run-1", route_id=ROUTE)
    finally:
        budget.ledger_path.chmod(0o600)
    assert budget.ledger_path.read_text(encoding="utf-8") == before


def test_a_ledger_whose_directory_is_missing_cannot_be_appended_to(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    nowhere = Budget(tmp_path / "gone" / "budget.yaml", budget.config)
    with pytest.raises(BudgetError, match=r"cannot be appended to \(No such file or directory\)"):
        prove_ledger(nowhere, run_id="run-1", route_id=ROUTE)


def test_a_line_that_cannot_be_written_is_a_budget_error_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    budget = write_budget(tmp_path)
    monkeypatch.setattr(Budget, "record", disk_full)
    with pytest.raises(BudgetError) as caught:
        prove_ledger(budget, run_id="run-1", route_id=ROUTE)
    assert (
        str(caught.value)
        == f"{budget.ledger_path}: cannot be appended to (No space left on device)"
    )
