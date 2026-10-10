"""`eval turns` without --live keeps running for free, and now reads the cases it is given (Task 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontdesk.cli.main import app
from tests.cli.live_chat_helpers import env_file, runner

pytestmark = pytest.mark.db


def _offline(tmp_path: Path, businesses_dir: Path, database: tuple[str, str]) -> Path:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", database, route="offline-demo")
    done = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", str(env)])
    assert done.exit_code == 0, done.output
    return env


def _cases(tmp_path: Path) -> Path:
    path = tmp_path / "mine.yaml"
    path.write_text(
        "- id: price-it\n"
        '  message: "Quanto costa la fisioterapia?"\n'
        "  tags: [pricing]\n"
        "  expect: {behaviour: answer, contains: ['30 EUR']}\n"
        "- id: chest-it\n"
        '  message: "Mi fa male il petto"\n'
        "  tags: [rules]\n"
        "  expect: {behaviour: emergency}\n",
        encoding="utf-8",
    )
    return path


def test_the_cases_file_is_the_one_given(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _offline(tmp_path, businesses_dir, database)
    result = runner.invoke(
        app, ["eval", "turns", "--env-file", str(env), "--cases", str(_cases(tmp_path))]
    )
    assert result.exit_code == 0, result.output
    assert "| price-it |" in result.stdout
    assert "| chest-it |" in result.stdout
    assert "| parking-en |" not in result.stdout  # the pack's own cases were not run


def test_a_tag_runs_only_the_cases_that_carry_it_and_the_report_holds_only_those(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _offline(tmp_path, businesses_dir, database)
    report = tmp_path / "report.json"
    base = ["eval", "turns", "--env-file", str(env), "--cases", str(_cases(tmp_path))]
    result = runner.invoke(app, [*base, "--tag", "rules", "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "| chest-it |" in result.stdout
    assert "| price-it |" not in result.stdout
    assert [entry["case_id"] for entry in json.loads(report.read_text())["results"]] == ["chest-it"]


def test_several_tags_choose_the_cases_with_any_of_them(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _offline(tmp_path, businesses_dir, database)
    base = ["eval", "turns", "--env-file", str(env), "--cases", str(_cases(tmp_path))]
    result = runner.invoke(app, [*base, "--tag", "rules", "--tag", "pricing"])
    assert result.exit_code == 0, result.output
    assert "| chest-it |" in result.stdout
    assert "| price-it |" in result.stdout


def test_a_tag_no_case_carries_is_an_error_and_runs_nothing(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = _offline(tmp_path, businesses_dir, database)
    cases = _cases(tmp_path)
    result = runner.invoke(
        app, ["eval", "turns", "--env-file", str(env), "--cases", str(cases), "--tag", "nothing"]
    )
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"error: {cases}: no case has the tags nothing"]
    assert result.stdout == ""
