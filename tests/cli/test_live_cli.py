"""`eval turns --live` and `eval report` from the command line, on the free offline route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from frontdesk.cli.main import app
from tests.model_gateway.budget_helpers import write_budget

runner = CliRunner()
NO_DATABASE = ("postgresql+psycopg://nobody:x@127.0.0.1:1/none",) * 2


def _env(
    tmp_path: Path, pack_dir: Path, database: tuple[str, str], route: str = "offline-demo"
) -> str:
    owner_url, app_url = database
    env = tmp_path / "clinic.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={pack_dir}\n"
        f"FRONTDESK_DATABASE_URL={app_url}\n"
        f"FRONTDESK_MIGRATION_DATABASE_URL={owner_url}\n"
        f"FRONTDESK_MODEL_ROUTE={route}\n"
        "FRONTDESK_EMBEDDING_BACKEND=hash\n",
        encoding="utf-8",
    )
    return str(env)


def _cases(tmp_path: Path) -> Path:
    path = tmp_path / "turns.yaml"
    path.write_text(
        "- id: price-it\n"
        '  message: "Quanto costa la fisioterapia?"\n'
        "  expect: {behaviour: answer, contains: ['30 EUR'], critical: true}\n"
        "- id: chest-it\n"
        '  message: "Mi fa male il petto"\n'
        "  expect: {behaviour: emergency}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.db
def test_a_dry_run_on_the_offline_route_writes_records_reports_and_the_review_page(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _env(tmp_path, businesses_dir / "demo-clinic", database)
    assert runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env]).exit_code == 0
    run_dir = tmp_path / "runs" / "dry-1"
    args = ["eval", "turns", "--env-file", env, "--live", "--route", "offline-demo"]
    args += ["--cases", str(_cases(tmp_path)), "--run-dir", str(run_dir), "--yes"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "cases recorded: 2/2" in result.output
    assert "PASS     Missed emergencies: 0" in result.output
    lines = (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["case_id"] for line in lines] == ["price-it", "chest-it"]
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (manifest["route"], manifest["business_id"]) == ("offline-demo", "demo-clinic")
    assert (run_dir / "report.md").exists()
    assert (run_dir / "review.html").exists()
    again = runner.invoke(app, args)  # everything is recorded: nothing runs again
    assert again.exit_code == 0, again.output
    assert "0 cases, 0 turns to run" in again.output
    report = runner.invoke(app, ["eval", "report", "--run-dir", str(run_dir)])
    assert report.exit_code == 0, report.output


@pytest.mark.db
def test_a_dry_run_may_name_the_budget_and_costs_nothing(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _env(tmp_path, businesses_dir / "demo-clinic", database)
    assert runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env]).exit_code == 0
    budget = write_budget(tmp_path)
    args = ["eval", "turns", "--env-file", env, "--live", "--route", "offline-demo", "--yes"]
    args += ["--cases", str(_cases(tmp_path)), "--run-dir", str(tmp_path / "dry")]
    result = runner.invoke(app, [*args, "--budget", str(budget.path)])
    assert result.exit_code == 0, result.output
    assert "at most" not in result.output  # no estimate: the offline route costs nothing
    assert not budget.ledger_path.exists()


@pytest.mark.db
def test_a_run_never_shares_the_budget_with_another_at_once(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _env(tmp_path, businesses_dir / "demo-clinic", database)
    assert runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env]).exit_code == 0
    budget = write_budget(tmp_path)
    args = ["eval", "turns", "--env-file", env, "--live", "--route", "offline-demo", "--yes"]
    args += ["--cases", str(_cases(tmp_path)), "--run-dir", str(tmp_path / "dry")]
    with budget.exclusive():  # another run holds the ledger
        result = runner.invoke(app, [*args, "--budget", str(budget.path)])
    assert result.exit_code == 1
    assert "another live run is using this budget" in result.stderr
    assert not (tmp_path / "dry" / "turns.jsonl").exists()


@pytest.mark.db
def test_a_run_dir_belongs_to_one_run(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _env(tmp_path, businesses_dir / "demo-clinic", database)
    assert runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env]).exit_code == 0
    run_dir = tmp_path / "runs" / "dry-1"
    cases = _cases(tmp_path)
    args = ["eval", "turns", "--env-file", env, "--live", "--route", "offline-demo"]
    args += ["--cases", str(cases), "--run-dir", str(run_dir), "--yes"]
    assert runner.invoke(app, args).exit_code == 0
    cases.write_text(cases.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "holds another run (it differs in cases_digest)" in result.stderr
    report = runner.invoke(app, ["eval", "report", "--run-dir", str(run_dir)])
    assert report.exit_code == 1
    assert "changed since the run" in report.stderr


def test_live_options_need_live(tmp_path: Path, businesses_dir: Path) -> None:
    env = _env(tmp_path, businesses_dir / "demo-clinic", NO_DATABASE)
    result = runner.invoke(app, ["eval", "turns", "--env-file", env, "--route", "offline-demo"])
    assert result.exit_code == 2
    assert "are for --live runs" in result.stderr


def test_a_route_that_calls_a_model_is_never_run_without_live(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = _env(
        tmp_path, businesses_dir / "demo-clinic", NO_DATABASE, route="vertex-eu-gemini-flash-lite"
    )
    result = runner.invoke(app, ["eval", "turns", "--env-file", env])
    assert result.exit_code == 2
    assert "calls a live model: add --live" in result.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--route", "openai-us"], "model route 'openai-us' is not registered"),
        (["--route", "vertex-eu-gemini-flash-lite"], "--budget is required for a live route"),
    ],
)
def test_a_live_run_refuses_before_it_opens_anything(
    tmp_path: Path, businesses_dir: Path, extra: list[str], message: str
) -> None:
    env = _env(tmp_path, businesses_dir / "demo-clinic", NO_DATABASE)
    args = ["eval", "turns", "--env-file", env, "--live", "--run-dir", str(tmp_path / "r"), *extra]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert message in result.stderr
    assert not (tmp_path / "r").exists()


def test_a_live_run_refuses_proxy_and_certificate_variables(
    tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.stray.example:3128")
    env = _env(tmp_path, businesses_dir / "demo-clinic", NO_DATABASE)
    args = ["eval", "turns", "--env-file", env, "--live", "--run-dir", str(tmp_path / "r")]
    result = runner.invoke(app, [*args, "--route", "vertex-eu-gemini-flash-lite"])
    assert result.exit_code == 1
    assert "unset HTTPS_PROXY" in result.stderr
