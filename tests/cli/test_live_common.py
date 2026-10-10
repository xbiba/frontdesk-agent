"""The helpers `chat --live` and `eval turns --live` share, without a command around them (Task 8)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer

from frontdesk.cli.live_common import (
    check_access,
    eur_left,
    refuse_environment_overrides,
    spent_usd,
)
from frontdesk.model_gateway.access import RouteAccessError
from frontdesk.model_gateway.costs import BudgetError
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
    check_access(GEMINI, "clinica-mcu")
    assert access_checks.calls == [(ROUTE, "clinica-mcu")]


def test_a_route_that_fails_the_proof_is_the_error_line_the_check_wrote(
    access_checks: AccessChecks, capsys: pytest.CaptureFixture[str]
) -> None:
    access_checks.error = RouteAccessError(
        "Vertex AI is not answering (503 UNAVAILABLE): try again"
    )
    with pytest.raises(typer.Exit) as raised:
        check_access(GEMINI, "clinica-mcu")
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
