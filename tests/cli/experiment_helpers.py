"""What the tests of `eval embedders` share: its env file and command line, and stand-ins for the
local model and for Vertex AI that cost nothing and reach nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontdesk.cli import experiment
from frontdesk.knowledge.embedder_experiment import Candidate
from frontdesk.knowledge.embedders import HashEmbedder
from tests.knowledge.test_embedder_experiment import SmallHashEmbedder

CANDIDATE = "vertex-ew1-gemini-embedding-001"
ELSEWHERE = "postgresql+psycopg://u:p@h/frontdesk_mcu_experiment"  # a scratch name, never opened


def env_file(tmp_path: Path, pack_dir: Path, owner_url: str, backend: str = "fastembed") -> str:
    env = tmp_path / "experiment.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={pack_dir}\n"
        f"FRONTDESK_DATABASE_URL={owner_url}\n"
        f"FRONTDESK_MIGRATION_DATABASE_URL={owner_url}\n"
        f"FRONTDESK_EMBEDDING_BACKEND={backend}\n"
        "FRONTDESK_GCP_PROJECT=demo-project\n",
        encoding="utf-8",
    )
    return str(env)


def experiment_args(
    env: str,
    tmp_path: Path,
    businesses_dir: Path,
    budget: Path,
    cases: Path | None = None,
    run_dir: Path | None = None,
) -> list[str]:
    cases = cases or businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml"
    return [
        "eval", "embedders", "--env-file", env, "--cases", str(cases),
        "--run-dir", str(run_dir or tmp_path / "embedders"), "--budget", str(budget),
        "--candidate", CANDIDATE, "--yes",
    ]  # fmt: skip


def offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reference and the candidate stand in for the local model and for Vertex AI."""
    small = SmallHashEmbedder(16)
    monkeypatch.setattr(experiment, "build_embedder", lambda backend: HashEmbedder())
    monkeypatch.setattr(
        experiment,
        "vertex_candidate",
        lambda route, settings, budget, run_id: Candidate(route, 16, small, small),
    )
