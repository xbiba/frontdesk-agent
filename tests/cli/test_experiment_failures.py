"""A failure of Vertex AI ends `eval embedders` in one clear line, and no text is sent that the
ledger does not count (M3 Task 9, fix round 1, I-2).

The embedders are the real ones, built by the command's own `vertex_candidate`, on a scripted
client: each request is answered as the test says, nothing waits, and nothing leaves the process.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from frontdesk.cli import experiment
from frontdesk.cli.main import app
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.model_gateway import embeddings
from frontdesk.model_gateway.costs import Budget
from tests.cli.experiment_helpers import env_file, experiment_args
from tests.model_gateway.attempt_helpers import ledger, read_timeout, refused
from tests.model_gateway.budget_helpers import write_budget
from tests.model_gateway.embedding_helpers import (
    ScriptedEmbeddings,
    Step,
    answer,
    expired_credentials,
    no_credentials,
    status,
)

pytestmark = pytest.mark.db

runner = CliRunner()
HINT = "the results so far are kept; run the same command to resume"
PRICE_PER_CHARACTER = 0.5 / 1_000  # the budget of the tests prices a thousand characters at 0.5
NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root can write to any file")


@dataclass(frozen=True)
class Failure:
    script: tuple[Step, ...]  # how each request ends, in order
    line: str  # what the person reads, before the hint
    sent: int  # the requests that went out
    unconfirmed: int = 0  # of them, those sent without an answer: each is counted at its reserve


FAILURES = {
    "403, the API is not enabled for the project": Failure(
        (status(403),),
        "Vertex AI refused project 'demo-project' (SCRIPTED: scripted). Check that the API is "
        "enabled (`gcloud services enable aiplatform.googleapis.com --project demo-project`) "
        "and that your account may use it",
        sent=1,
    ),
    "404, the model is not served there": Failure(
        (status(404),),
        "model 'gemini-embedding-001' is not served at https://europe-west1-aiplatform.googleapis.com",
        sent=1,
    ),
    "503 at every attempt": Failure(
        (status(503),) * 4, "Vertex AI is not answering (503 SCRIPTED): try again later", sent=4
    ),
    "no answer at any attempt": Failure(
        (read_timeout,) * 4,
        "Vertex AI cannot be reached (ReadTimeout): check the network connection, then try again",
        sent=4,
        unconfirmed=4,  # sent, and nothing came back: each may have been billed, each is counted
    ),
    "no network": Failure(
        (refused,) * 4,
        "Vertex AI cannot be reached (ConnectError): check the network connection, then try again",
        sent=4,
    ),
    "credentials that expired": Failure(
        (expired_credentials,),
        "Google credentials are missing or expired: run `gcloud auth application-default login`, "
        "then try again",
        sent=1,
    ),
    "no credentials at all": Failure(
        (no_credentials,),
        "Google credentials are missing or expired: run `gcloud auth application-default login`, "
        "then try again",
        sent=1,
    ),
}


@pytest.fixture
def vertex(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """Puts a scripted client where the command's embedders look for the pinned one, so that the
    real `vertex_candidate` and `VertexEmbedder` run; the reference is a stand-in for the local
    model. The waits between attempts are zero."""

    def install(client: Any) -> Any:
        monkeypatch.setattr(
            embeddings, "google_client", lambda route, *, gcp_project: client.client
        )
        monkeypatch.setattr(embeddings, "RETRY_WAITS", (0.0, 0.0, 0.0))
        monkeypatch.setattr(experiment, "build_embedder", lambda backend: HashEmbedder())
        return client

    return install


def _run(tmp_path: Path, businesses_dir: Path, scratch_url: str, budget: Budget) -> Any:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", scratch_url)
    return runner.invoke(app, experiment_args(env, tmp_path, businesses_dir, budget.path))


@pytest.mark.parametrize("name", list(FAILURES))
def test_a_failure_of_vertex_ai_is_one_clear_line_and_leaves_no_text_paid_and_uncounted(
    name: str, tmp_path: Path, businesses_dir: Path, scratch_url: str, vertex: Callable[..., Any]
) -> None:
    failure = FAILURES[name]
    client = vertex(ScriptedEmbeddings(*failure.script))
    budget = write_budget(tmp_path)
    result = _run(tmp_path, businesses_dir, scratch_url, budget)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # an exit, not a crash
    assert result.stderr == f"error: {failure.line}; {HINT}\n"
    assert "Traceback" not in result.output
    assert len(client.requests) == failure.sent  # no attempt more, none fewer
    lines = ledger(budget)
    assert lines[0] == ("preflight", 0.0)  # the ledger took a line before the first text was sent
    reserve = len(client.requests[0]["contents"][0].encode()) * PRICE_PER_CHARACTER
    # What went out without an answer is counted in full; an error status or a request that never
    # left costs nothing: every text that may have been billed has its line.
    assert lines[1:] == [("embedding-unconfirmed", pytest.approx(reserve))] * failure.unconfirmed
    assert [path.name for path in (tmp_path / "embedders").iterdir()] == ["reference.json"]


class PassagesThenOutage:
    """Answers every passage, and every question with a 503: the failure comes after money was
    spent."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def embed_content(self, **request: Any) -> object:
        self.requests.append(request)
        if request["config"].task_type == "RETRIEVAL_QUERY":
            raise status(503)()
        return answer(billed=500)()

    @property
    def client(self) -> Any:
        return SimpleNamespace(models=self)


def test_the_passages_paid_for_before_a_failure_are_all_in_the_ledger(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, vertex: Callable[..., Any]
) -> None:
    client = vertex(PassagesThenOutage())
    budget = write_budget(tmp_path)
    result = _run(tmp_path, businesses_dir, scratch_url, budget)

    assert result.exit_code == 1
    assert result.stderr == (
        f"error: Vertex AI is not answering (503 SCRIPTED): try again later; {HINT}\n"
    )
    passages = [r for r in client.requests if r["config"].task_type == "RETRIEVAL_DOCUMENT"]
    questions = [r for r in client.requests if r["config"].task_type == "RETRIEVAL_QUERY"]
    assert passages
    assert len(questions) == 4  # the first question, at all its attempts
    paid = [
        max(500, len(request["contents"][0].encode())) * PRICE_PER_CHARACTER for request in passages
    ]
    assert ledger(budget) == [("preflight", 0.0), *[("embedding", pytest.approx(p)) for p in paid]]
    assert budget.spent_usd() == pytest.approx(sum(paid))


# --- A ledger that cannot take a line ------------------------------------------------------------


@NOT_ROOT
def test_a_ledger_that_cannot_be_appended_to_stops_the_experiment_before_any_text_is_sent(
    tmp_path: Path, businesses_dir: Path, scratch_url: str, vertex: Callable[..., Any]
) -> None:
    client = vertex(ScriptedEmbeddings(*([answer()] * 100)))
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("", encoding="utf-8")
    budget.ledger_path.chmod(0o444)  # it may be read, and not appended to
    try:
        result = _run(tmp_path, businesses_dir, scratch_url, budget)
    finally:
        budget.ledger_path.chmod(0o600)

    assert result.exit_code == 1
    assert result.stderr == (
        f"error: {budget.ledger_path}: cannot be appended to (Permission denied)\n"
    )
    assert client.requests == []  # nothing was sent that could not be counted
    assert budget.ledger_path.stat().st_size == 0
    assert not (tmp_path / "embedders").exists()  # and nothing was measured or saved


def test_a_ledger_that_fails_after_the_proof_stops_the_text_without_another_attempt(
    tmp_path: Path,
    businesses_dir: Path,
    scratch_url: str,
    vertex: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The proof passes; the disk fills while the first text is away. Its line cannot be written, so
    # no other attempt is sent: it would be paid and uncounted too.
    client = vertex(ScriptedEmbeddings(read_timeout, answer()))
    real, calls = Budget.record, []

    def fills_up(self: Budget, **entry: Any) -> None:
        calls.append(entry["kind"])
        if len(calls) > 1:
            raise OSError(28, "No space left on device")
        real(self, **entry)

    monkeypatch.setattr(Budget, "record", fills_up)
    budget = write_budget(tmp_path)
    result = _run(tmp_path, businesses_dir, scratch_url, budget)

    assert result.exit_code == 1
    assert result.stderr == (
        f"error: {budget.ledger_path}: cannot be appended to (No space left on device)\n"
    )
    assert len(client.requests) == 1
    assert calls == ["preflight", "embedding-unconfirmed"]
