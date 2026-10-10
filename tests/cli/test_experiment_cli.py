"""`eval embedders`: scratch databases only, the reference and each candidate, then the decision."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import typer
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from typer.testing import CliRunner

from frontdesk.cli import experiment
from frontdesk.cli.main import app
from frontdesk.knowledge.embedder_experiment import Candidate, CandidateResult
from frontdesk.knowledge.evaluation import load_cases
from frontdesk.model_gateway.costs import BudgetExhaustedError
from frontdesk.model_gateway.embeddings import EmbeddingRateLimitedError, VertexEmbedder
from frontdesk.model_gateway.routes import ROUTES
from frontdesk.settings import Settings
from tests.cli.experiment_helpers import (
    CANDIDATE,
    ELSEWHERE,
    env_file,
    experiment_args,
    offline_models,
)
from tests.model_gateway.budget_helpers import write_budget

runner = CliRunner()


@pytest.mark.db
def test_the_experiment_compares_the_reference_and_the_candidates(
    tmp_path: Path,
    businesses_dir: Path,
    scratch_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offline_models(monkeypatch)
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, businesses_dir / "demo-clinic", scratch_url)
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget))
    assert result.exit_code == 0, result.output
    assert "**Decision: " in result.output
    run_dir = tmp_path / "embedders"
    assert {path.name for path in run_dir.iterdir()} == {
        "reference.json",
        f"{CANDIDATE}.json",
        "embedders.md",
    }


def test_the_experiment_never_writes_into_a_database_that_is_not_a_scratch_one(
    tmp_path: Path, businesses_dir: Path
) -> None:
    budget = write_budget(tmp_path).path
    env = env_file(
        tmp_path, businesses_dir / "demo-clinic", "postgresql+psycopg://u:p@h/frontdesk_mcu"
    )
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget))
    assert result.exit_code == 1
    assert "ends in '_experiment', not 'frontdesk_mcu'" in result.stderr


def test_the_reference_is_the_demo_s_local_model(tmp_path: Path, businesses_dir: Path) -> None:
    budget = write_budget(tmp_path).path
    url = "postgresql+psycopg://u:p@h/frontdesk_mcu_experiment"
    env = env_file(tmp_path, businesses_dir / "demo-clinic", url, backend="hash")
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget))
    assert result.exit_code == 1
    assert "FRONTDESK_EMBEDDING_BACKEND=fastembed" in result.stderr


# --- The server has the last word on which database this is -------------------------------------


def _scalar(url: str, statement: str) -> object:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return connection.scalar(text(statement))
    finally:
        engine.dispose()


@pytest.mark.db
def test_a_url_whose_query_names_another_database_is_refused_before_anything_is_written(
    tmp_path: Path,
    businesses_dir: Path,
    new_server_database: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The path of the URL names a scratch database, which is all that the name check reads; the
    # driver connects to the database that `dbname=` names.
    offline_models(monkeypatch)
    scratch, plain = new_server_database("_experiment"), new_server_database("")
    plain_name = make_url(plain).database
    crafted = (
        make_url(scratch)
        .update_query_dict({"dbname": plain_name or ""})
        .render_as_string(hide_password=False)
    )
    assert _scalar(crafted, "SELECT current_database()") == plain_name  # the gap, as it was
    budget = write_budget(tmp_path)
    env = env_file(tmp_path, businesses_dir / "demo-clinic", crafted)
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget.path))
    assert result.exit_code == 1
    assert result.stderr == (
        "error: the experiment writes tables: the server says this URL reaches "
        f"'{plain_name}', whose name does not end in '_experiment' "
        "(FRONTDESK_MIGRATION_DATABASE_URL)\n"
    )
    tables = "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
    assert _scalar(plain, tables) == 0  # not migrated, not written to
    assert not (tmp_path / "embedders").exists()
    assert not budget.ledger_path.exists()


def test_a_database_that_does_not_answer_is_one_error_line(
    tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)

    class Down:
        def connect(self) -> None:
            raise OperationalError("SELECT current_database()", {}, Exception("refused"))

        def dispose(self) -> None:
            pass

    monkeypatch.setattr(experiment, "make_engine", lambda url: Down())
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, businesses_dir / "demo-clinic", ELSEWHERE)
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget))
    assert result.exit_code == 1
    assert result.stderr == (
        "error: cannot reach the database (h); start it with: docker compose up -d --wait postgres\n"
    )
    assert not (tmp_path / "embedders").exists()


# --- Before it asks or spends: the same refusals as the other live commands --------------------


def test_the_experiment_refuses_proxy_and_certificate_variables(
    tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Google's token refresh would read them, and the candidates would be reached another way.
    monkeypatch.setenv("https_proxy", "http://proxy.stray.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/stray-ca.pem")
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, businesses_dir / "demo-clinic", ELSEWHERE)
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget))
    assert result.exit_code == 1
    assert result.stderr == (
        "error: a live run does not go through proxies, other certificates or another billing "
        "project: unset https_proxy, SSL_CERT_FILE\n"
    )
    assert not (tmp_path / "embedders").exists()


def test_a_pack_that_cannot_be_loaded_is_one_error_line(
    tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, tmp_path / "no-such-pack", ELSEWHERE)
    result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget))
    assert result.exit_code == 1
    assert result.stderr.startswith("error: business pack directory not found")
    assert "Traceback" not in result.output


def test_the_experiment_asks_before_it_starts(
    tmp_path: Path, businesses_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, businesses_dir / "demo-clinic", ELSEWHERE)
    args = [arg for arg in experiment_args(env, tmp_path, businesses_dir, budget) if arg != "--yes"]
    cases = len(load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml"))
    result = runner.invoke(app, args, input="n\n")
    assert result.exit_code == 1
    assert (
        f"{cases} questions, 1 candidates; the budget has 10.00 of 10.00 EUR left" in result.output
    )
    assert "Start the experiment?" in result.output
    # Refused before the database is opened (its host leads nowhere) or a file is written.
    assert not (tmp_path / "embedders").exists()


# --- A Vertex AI candidate, as the command builds it --------------------------------------------


def _vertex(route_id: str, tmp_path: Path, project: str | None = "demo-project") -> Candidate:
    return experiment.vertex_candidate(
        route_id, Settings(gcp_project=project), write_budget(tmp_path), "run-1"
    )


def test_a_candidate_has_one_embedder_for_passages_and_one_for_questions(tmp_path: Path) -> None:
    candidate = _vertex(CANDIDATE, tmp_path)
    assert (candidate.name, candidate.dimension) == (CANDIDATE, 3072)
    documents, queries = candidate.documents, candidate.queries
    assert isinstance(documents, VertexEmbedder)
    assert isinstance(queries, VertexEmbedder)
    assert (documents._task, queries._task) == ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")
    for embedder in (documents, queries):
        api = embedder._client._api_client
        assert (api.project, api.location) == ("demo-project", "europe-west1")
        assert embedder._run_id == "run-1"


@pytest.mark.parametrize(
    ("route", "message"),
    [
        pytest.param(
            "no-such-route", "model route 'no-such-route' is not registered", id="unknown"
        ),
        pytest.param(
            "vertex-eu-gemini-flash-lite",
            "route 'vertex-eu-gemini-flash-lite' is not an embedding route",
            id="a chat route",
        ),
        pytest.param(
            "vertex-eu-opus", "route 'vertex-eu-opus' is not an embedding route", id="claude"
        ),
        pytest.param(
            "vertex-ew1-text-multilingual-embedding-002",
            "no price for route 'vertex-ew1-text-multilingual-embedding-002'",
            id="a route the budget does not price",
        ),
    ],
)
def test_a_candidate_that_cannot_be_used_is_one_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], route: str, message: str
) -> None:
    with pytest.raises(typer.Exit) as raised:
        _vertex(route, tmp_path)
    assert raised.value.exit_code == 1
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert message in error


@pytest.mark.parametrize("project", [None, "", "  "])
def test_a_candidate_needs_a_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], project: str | None
) -> None:
    with pytest.raises(typer.Exit):
        _vertex(CANDIDATE, tmp_path, project)
    assert capsys.readouterr().err == (
        "error: FRONTDESK_GCP_PROJECT is required for the Vertex AI candidates\n"
    )


def test_a_model_google_has_retired_is_not_a_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    today = datetime.now(UTC).date()
    monkeypatch.setitem(ROUTES, CANDIDATE, replace(ROUTES[CANDIDATE], retires_on=today))
    with pytest.raises(typer.Exit):
        _vertex(CANDIDATE, tmp_path)
    assert capsys.readouterr().err == (
        f"error: route '{CANDIDATE}': Google retired gemini-embedding-001 on {today}\n"
    )


# --- Stops, and the budget's ledger -------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(BudgetExhaustedError("the budget has 0.0000 EUR left"), id="the budget"),
        pytest.param(EmbeddingRateLimitedError("the model is rate limited"), id="a rate limit"),
    ],
)
def test_a_stop_keeps_the_results_so_far_and_the_same_command_resumes_with_the_next(
    tmp_path: Path,
    businesses_dir: Path,
    scratch_url: str,
    monkeypatch: pytest.MonkeyPatch,
    stop: Exception,
) -> None:
    offline_models(monkeypatch)
    measured: list[str] = []
    failing = {CANDIDATE}  # fails once, the first time it is measured
    real = experiment.run_candidate

    def run(session: Any, **kwargs: Any) -> CandidateResult:
        name = kwargs["candidate"].name
        if name in failing:
            failing.discard(name)
            raise stop
        measured.append(name)
        return real(session, **kwargs)

    monkeypatch.setattr(experiment, "run_candidate", run)
    budget = write_budget(tmp_path).path
    env = env_file(tmp_path, businesses_dir / "demo-clinic", scratch_url)
    args = experiment_args(env, tmp_path, businesses_dir, budget)
    run_dir = tmp_path / "embedders"

    first = runner.invoke(app, args)
    assert first.exit_code == 1
    assert first.stderr == (
        f"error: {stop}: the results so far are kept; run the same command to resume\n"
    )
    assert [path.name for path in run_dir.iterdir()] == ["reference.json"]

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert measured == ["reference", CANDIDATE]  # the reference is not measured a second time
    assert {path.name for path in run_dir.iterdir()} == {
        "reference.json",
        f"{CANDIDATE}.json",
        "embedders.md",
    }
    assert "**Decision: " in second.output


@pytest.mark.db
def test_the_experiment_is_refused_while_another_run_holds_the_budget(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_models(monkeypatch)
    budget = write_budget(tmp_path)
    env = env_file(tmp_path, businesses_dir / "demo-clinic", scratch_url)
    with budget.exclusive():  # a live run, or a chat, is using the ledger
        result = runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget.path))
    assert result.exit_code == 1
    assert "another live run is using this budget" in result.stderr
    assert not (tmp_path / "embedders" / "reference.json").exists()  # nothing was measured
