"""What `eval embedders` does with the files it reads and writes, and with Ctrl-C (M3 Task 9, fix
round 1): one clear line for each, the results already saved kept, and nothing measured for money
when an input is wrong."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from frontdesk.cli import experiment
from frontdesk.cli.main import app
from frontdesk.knowledge.embedder_experiment import Candidate
from frontdesk.knowledge.embedders import HashEmbedder
from tests.cli.experiment_helpers import (
    CANDIDATE,
    ELSEWHERE,
    env_file,
    experiment_args,
    offline_models,
)
from tests.knowledge.test_embedder_experiment import SmallHashEmbedder
from tests.model_gateway.attempt_helpers import ledger
from tests.model_gateway.budget_helpers import write_budget

runner = CliRunner()
NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root can write to any file")


def _run(
    tmp_path: Path,
    businesses_dir: Path,
    url: str,
    *,
    cases: Path | None = None,
    run_dir: Path | None = None,
) -> Any:
    budget = write_budget(tmp_path)
    env = env_file(tmp_path, businesses_dir / "demo-clinic", url)
    return runner.invoke(
        app, experiment_args(env, tmp_path, businesses_dir, budget.path, cases, run_dir)
    )


# --- Ctrl-C -------------------------------------------------------------------------------------


class Interrupted:
    """An embedder whose user presses Ctrl-C while it works."""

    dimension = 16
    identity = "interrupted"

    def embed(self, texts: Any) -> Any:
        raise KeyboardInterrupt


@pytest.mark.db
def test_ctrl_c_ends_the_run_with_the_line_a_live_run_gives_and_keeps_what_is_saved(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = SmallHashEmbedder(16)
    monkeypatch.setattr(experiment, "build_embedder", lambda backend: HashEmbedder())
    monkeypatch.setattr(
        experiment,
        "vertex_candidate",
        lambda route, settings, budget, run_id: Candidate(route, 16, small, Interrupted()),
    )
    result = _run(tmp_path, businesses_dir, scratch_url)
    assert result.exit_code == 130
    assert result.stderr == "stopped: run the same command again to resume\n"
    assert [path.name for path in (tmp_path / "embedders").iterdir()] == ["reference.json"]


# --- Files that cannot be written or read -----------------------------------------------------------


@pytest.mark.db
@NOT_ROOT
def test_a_run_directory_that_cannot_be_made_is_one_error_line_before_anything_is_measured(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        result = _run(tmp_path, businesses_dir, scratch_url, run_dir=locked / "embedders")
    finally:
        locked.chmod(0o700)
    assert result.exit_code == 1
    assert result.stderr == f"error: {locked / 'embedders'}: Permission denied\n"
    assert isinstance(result.exception, SystemExit)  # an exit, not a crash


@pytest.mark.db
def test_a_result_that_is_a_directory_is_named_before_money_is_spent(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    (tmp_path / "embedders" / "reference.json").mkdir(parents=True)
    result = _run(tmp_path, businesses_dir, scratch_url)
    assert result.exit_code == 1
    assert result.stderr == f"error: {tmp_path / 'embedders' / 'reference.json'}: Is a directory\n"


@pytest.mark.db
def test_a_report_that_cannot_be_written_is_named_and_the_results_are_kept(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    (tmp_path / "embedders" / "embedders.md").mkdir(parents=True)
    result = _run(tmp_path, businesses_dir, scratch_url)
    assert result.exit_code == 1
    assert result.stderr == f"error: {tmp_path / 'embedders' / 'embedders.md'}: Is a directory\n"
    assert {p.name for p in (tmp_path / "embedders").iterdir()} == {
        "embedders.md",
        "reference.json",
        f"{CANDIDATE}.json",
    }


def test_a_file_is_written_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "reference.json"
    experiment._save(target, "first\n")
    assert [path.name for path in tmp_path.iterdir()] == ["reference.json"]  # nothing left over
    assert target.read_text(encoding="utf-8") == "first\n"

    def stopped(self: Path, *args: object) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(Path, "replace", stopped)  # the stop comes between the two steps
    with pytest.raises(OSError, match="Input/output error"):
        experiment._save(target, "second\n")
    assert target.read_text(encoding="utf-8") == "first\n"  # the old file, never half of the new
    assert [path.name for path in tmp_path.iterdir()] == ["reference.json"]  # and no debris


@pytest.mark.db
def test_a_saved_result_that_cannot_be_read_back_is_one_error_line_and_costs_nothing(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    saved = tmp_path / "embedders" / "reference.json"
    saved.parent.mkdir()
    for content in ('{"name": "reference", "identi', "", "[]", "{}", "null"):
        saved.write_text(content, encoding="utf-8")  # a stop in the middle of a write, or worse
        result = _run(tmp_path, businesses_dir, scratch_url)
        assert result.exit_code == 1, content
        assert result.stderr == (
            f"error: {saved}: is not a result of this experiment; "
            "delete it to measure that embedder again\n"
        ), content
    assert [kind for kind, _ in ledger(write_budget(tmp_path))] == ["preflight"] * 5


# --- What the command is asked to measure ---------------------------------------------------------


def test_a_candidate_given_twice_is_refused_before_anything_is_built(
    tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, businesses_dir / "demo-clinic", ELSEWHERE)
    args = experiment_args(env, tmp_path, businesses_dir, budget)
    result = runner.invoke(app, [*args, "--candidate", CANDIDATE])
    assert result.exit_code == 1
    assert result.stderr == f"error: candidate '{CANDIDATE}' is given more than once\n"


@pytest.mark.db
def test_a_case_that_names_no_document_of_the_knowledge_is_refused_before_anything_is_measured(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A mistyped key can never be found, so it scores 0 for every embedder, without a word.
    offline_models(monkeypatch)
    cases = tmp_path / "retrieval.yaml"
    cases.write_text(
        "- id: parking\n  question: Is there parking?\n  language: en\n  relevant: [parking-en]\n"
        "- id: typo\n  question: Quanto costa?\n  language: it\n  relevant: [prices-it, parking-ne]\n",
        encoding="utf-8",
    )
    result = _run(tmp_path, businesses_dir, scratch_url, cases=cases)
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        f"error: {cases}: case 'typo': 'prices-it' is not a document of the active knowledge "
        "version (a document key is its file name without .md)",
        f"error: {cases}: case 'typo': 'parking-ne' is not a document of the active knowledge "
        "version (a document key is its file name without .md)",
    ]
    assert not (tmp_path / "embedders").exists()  # nothing was measured
