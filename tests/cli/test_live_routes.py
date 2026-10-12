"""Which routes a live run accepts, how it picks its cases, and what the command line refuses on its
own account (M3 Task 8, design §4.4): the checks that need no database and no model.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from frontdesk.agent.classifier import CLASSIFIER_LIMITS
from frontdesk.agent.responder import RESPONDER_LIMITS
from frontdesk.cli.live import (
    AGENT_REQUESTS_PER_TURN,
    REQUESTS_PER_TURN,
    evaluation_route,
    read_cases,
    select_cases,
)
from frontdesk.cli.main import app
from frontdesk.conversation.evaluation import TurnCase
from frontdesk.conversation.reception import RECEPTION_LIMITS
from frontdesk.model_gateway.routes import ROUTES

runner = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GEMINI = "vertex-eu-gemini-flash-lite"
NO_DATABASE = "postgresql+psycopg://nobody:x@127.0.0.1:1/none"


def _case(case_id: str, *tags: str) -> TurnCase:
    return TurnCase.model_validate(
        {"id": case_id, "message": "Ciao", "tags": list(tags), "expect": {}}
    )


# --- Which cases ------------------------------------------------------------------------------------


def test_without_a_tag_every_case_is_chosen() -> None:
    cases = [_case("a"), _case("b", "rules")]
    assert select_cases(cases, []) == cases


def test_a_case_with_any_of_the_tags_is_chosen_in_the_order_of_the_file() -> None:
    cases = [_case("a", "rules"), _case("b", "price"), _case("c"), _case("d", "price", "rules")]
    assert [case.id for case in select_cases(cases, ["price"])] == ["b", "d"]
    assert [case.id for case in select_cases(cases, ["price", "rules"])] == ["a", "b", "d"]


def test_a_cases_file_is_read_through_the_tags(tmp_path: Path) -> None:
    path = tmp_path / "turns.yaml"
    path.write_text(
        "- {id: a, message: x, tags: [rules], expect: {}}\n- {id: b, message: y, expect: {}}\n",
        encoding="utf-8",
    )
    assert [case.id for case in read_cases(path, ["rules"])] == ["a"]
    assert [case.id for case in read_cases(path, [])] == ["a", "b"]


def test_no_case_with_the_tag_is_an_error_naming_the_file_and_the_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "turns.yaml"
    path.write_text("- {id: a, message: x, expect: {}}\n", encoding="utf-8")
    with pytest.raises(typer.Exit) as raised:
        read_cases(path, ["rules", "price"])
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err == f"error: {path}: no case has the tags rules, price\n"


def test_a_cases_file_that_is_wrong_is_reported_as_it_is_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "turns.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(typer.Exit) as raised:
        read_cases(path, [])
    assert raised.value.exit_code == 1
    assert f"{path}: no cases" in capsys.readouterr().err


# --- Which routes -----------------------------------------------------------------------------------


@pytest.mark.parametrize("route_id", ["offline-demo", GEMINI, "vertex-eu-opus"])
def test_a_route_that_evaluation_data_may_reach_and_that_answers_messages_is_accepted(
    route_id: str,
) -> None:
    assert evaluation_route(route_id).route_id == route_id


@pytest.mark.parametrize(
    ("route_id", "line"),
    [
        pytest.param("openai-us", "error: model route 'openai-us' is not registered", id="unknown"),
        pytest.param(
            "vertex-ew1-gemini-embedding-001",
            "error: model route 'vertex-ew1-gemini-embedding-001' cannot answer messages: "
            "its provider is google-vertex-embedding",
            id="an-embedding-route",
        ),
    ],
)
def test_a_route_that_cannot_run_the_turns_is_refused_by_name(
    route_id: str, line: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(typer.Exit) as raised:
        evaluation_route(route_id)
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err == line + "\n"


def test_a_route_that_evaluation_data_may_not_reach_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    only_patients = replace(ROUTES[GEMINI], approved_for=frozenset())
    monkeypatch.setitem(ROUTES, GEMINI, only_patients)
    with pytest.raises(typer.Exit):
        evaluation_route(GEMINI)
    assert (
        capsys.readouterr().err == f"error: model route '{GEMINI}' is not approved for evaluation\n"
    )


def test_a_route_without_a_pinned_host_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(ROUTES, GEMINI, replace(ROUTES[GEMINI], region="us-central1"))
    with pytest.raises(typer.Exit) as raised:
        evaluation_route(GEMINI)
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err == (
        f"error: model route '{GEMINI}' has no pinned endpoint for region 'us-central1'\n"
    )


def test_a_model_google_has_retired_is_refused_from_the_day_it_retires(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    today = datetime.now(UTC).date()
    monkeypatch.setitem(ROUTES, GEMINI, replace(ROUTES[GEMINI], retires_on=today))
    with pytest.raises(typer.Exit) as raised:
        evaluation_route(GEMINI)
    assert raised.value.exit_code == 1
    assert capsys.readouterr().err == (
        f"error: route '{GEMINI}': Google retired gemini-3.1-flash-lite on {today}\n"
    )


def test_a_model_that_retires_tomorrow_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    monkeypatch.setitem(ROUTES, GEMINI, replace(ROUTES[GEMINI], retires_on=tomorrow))
    assert evaluation_route(GEMINI).retires_on == tomorrow


# --- The ceiling shown before a run -----------------------------------------------------------------


def test_the_ceiling_counts_every_request_the_pipeline_may_make_in_a_turn() -> None:
    # Two requests for the classifier and two for the responder: if a limit changes, so must this.
    limits = (CLASSIFIER_LIMITS, RESPONDER_LIMITS)
    assert all(item.request_limit is not None for item in limits)
    assert sum(item.request_limit or 0 for item in limits) == REQUESTS_PER_TURN
    # The reception agent instead: its tool calls and its reply, within its own limit.
    assert RECEPTION_LIMITS.request_limit == AGENT_REQUESTS_PER_TURN


# --- What the command line refuses on its own account ----------------------------------------------


def _env(tmp_path: Path, businesses_dir: Path, route: str = "offline-demo") -> str:
    env = tmp_path / "clinic.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={businesses_dir / 'demo-clinic'}\n"
        f"FRONTDESK_DATABASE_URL={NO_DATABASE}\n"
        f"FRONTDESK_MIGRATION_DATABASE_URL={NO_DATABASE}\n"
        f"FRONTDESK_MODEL_ROUTE={route}\n"
        "FRONTDESK_EMBEDDING_BACKEND=hash\n",
        encoding="utf-8",
    )
    return str(env)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["--live"], id="nothing"),
        pytest.param(["--live", "--route", "offline-demo"], id="no-run-dir"),
        pytest.param(["--live", "--run-dir", "runs/x"], id="no-route"),
    ],
)
def test_a_live_run_needs_a_route_and_a_run_directory(
    tmp_path: Path, businesses_dir: Path, arguments: list[str]
) -> None:
    env = _env(tmp_path, businesses_dir)
    result = runner.invoke(app, ["eval", "turns", "--env-file", env, *arguments])
    assert result.exit_code == 2
    assert result.stderr.splitlines() == ["error: --live needs --route and --run-dir"]


def test_a_live_run_does_not_silently_drop_the_output_file_it_was_asked_for(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = _env(tmp_path, businesses_dir)
    arguments = ["--live", "--route", "offline-demo", "--run-dir", str(tmp_path / "r")]
    result = runner.invoke(
        app, ["eval", "turns", "--env-file", env, *arguments, "--output", str(tmp_path / "o.json")]
    )
    assert result.exit_code == 2
    assert result.stderr.splitlines() == [
        "error: --output is for offline runs: a live run writes report.json in --run-dir"
    ]
    assert not (tmp_path / "r").exists()


def test_the_help_of_eval_turns_names_the_live_options_and_every_exit_code() -> None:
    result = runner.invoke(app, ["eval", "turns", "--help"])
    assert result.exit_code == 0
    text = ANSI_ESCAPE.sub("", result.stdout)
    for option in ("--live", "--route", "--run-dir", "--budget", "--limit", "--yes", "--tag"):
        assert option in text
    for code in ("0", "1", "3", "4", "5", "130"):
        assert re.search(rf"\b{code}\b", text), code


def test_eval_report_names_its_options() -> None:
    result = runner.invoke(app, ["eval", "report", "--help"])
    assert result.exit_code == 0
    text = ANSI_ESCAPE.sub("", result.stdout)
    for option in ("--run-dir", "--budget", "--tag"):
        assert option in text


def test_eval_report_of_a_directory_that_is_no_run_is_an_error_line(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "report", "--run-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        f"error: {tmp_path / 'run.json'}: cannot be read (No such file or directory)"
    ]


# --- A budget file a live run cannot use ------------------------------------------------------------


def _budget_file(directory: Path, *, route_lines: str) -> Path:
    path = directory / "budget.yaml"
    path.write_text(
        "cap_eur: 10\n"
        "exchange_rate:\n  usd_per_eur: 1.06\n  source: https://ecb.example\n  date: 2026-10-09\n"
        "prices:\n  source: https://prices.example\n  retrieved: 2026-10-10\n  routes:\n"
        f"{route_lines}",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("route_lines", "problem"),
    [
        pytest.param(
            "    vertex-eu-opus:\n      unit: tokens_1m\n      input_usd: 1\n      output_usd: 5\n",
            "no price for route 'vertex-eu-gemini-flash-lite'",
            id="the-route-has-no-price",
        ),
        pytest.param(
            f"    {GEMINI}:\n      unit: characters_1k\n      input_usd: 1\n",
            f"route '{GEMINI}' is priced per characters_1k, and the meter counts tokens",
            id="a-price-per-character",
        ),
    ],
)
def test_a_budget_that_cannot_pay_for_the_route_stops_a_live_run_before_it_starts(
    tmp_path: Path, businesses_dir: Path, route_lines: str, problem: str
) -> None:
    env = _env(tmp_path, businesses_dir, GEMINI)
    budget = _budget_file(tmp_path, route_lines=route_lines)
    arguments = ["--live", "--route", GEMINI, "--run-dir", str(tmp_path / "r")]
    result = runner.invoke(
        app, ["eval", "turns", "--env-file", env, *arguments, "--budget", str(budget)]
    )
    assert result.exit_code == 1
    [line] = result.stderr.splitlines()
    assert line.startswith(f"error: {budget}: ")
    assert problem in line
    assert not (tmp_path / "r").exists()
    assert result.stdout == ""


def test_a_budget_file_that_is_not_yaml_stops_a_live_run_before_it_starts(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = _env(tmp_path, businesses_dir, GEMINI)
    budget = tmp_path / "budget.yaml"
    budget.write_text("cap_eur: [", encoding="utf-8")
    arguments = ["--live", "--route", GEMINI, "--run-dir", str(tmp_path / "r")]
    result = runner.invoke(
        app, ["eval", "turns", "--env-file", env, *arguments, "--budget", str(budget)]
    )
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"error: {budget}: invalid YAML"]
