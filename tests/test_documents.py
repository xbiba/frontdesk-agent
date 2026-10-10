"""The documents name only what exists (M3 Task 9): the decision records point at the code and the
routes they decide on, the README at commands the program has, and no link leads nowhere."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import click
import pytest
import typer.main

from frontdesk.cli.main import app
from frontdesk.model_gateway import factory
from frontdesk.model_gateway.routes import ROUTES
from frontdesk.settings import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ADR_DIRECTORY = REPO_ROOT / "docs" / "adr"
ADR_4 = ADR_DIRECTORY / "0004-model-on-vertex-ai-eu.md"
ADR_5 = ADR_DIRECTORY / "0005-embeddings-chosen-by-evaluation.md"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)(?:#[^)]*)?\)")
BACKTICKED = re.compile(r"`([^`\n]+)`")
COMMAND = re.compile(r"\bfrontdesk ((?:db|knowledge|eval) [a-z-]+|chat)\b")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _commands() -> set[str]:
    """Every command path of the program, as it is typed after `frontdesk`."""
    found: set[str] = set()

    def walk(command: click.Command, prefix: str) -> None:
        for name, child in getattr(command, "commands", {}).items():
            found.add(f"{prefix} {name}".strip())
            walk(child, f"{prefix} {name}".strip())

    walk(typer.main.get_command(app), "")
    return found


@pytest.mark.parametrize(
    "document",
    [
        "README.md",
        "docs/adr/README.md",
        "docs/adr/0004-model-on-vertex-ai-eu.md",
        "docs/adr/0005-embeddings-chosen-by-evaluation.md",
        "docs/evals/README.md",
    ],
)
def test_every_relative_link_leads_to_a_file(document: str) -> None:
    path = REPO_ROOT / document
    missing = [
        target
        for target in RELATIVE_LINK.findall(_read(path))
        if not (path.parent / target).exists()
    ]
    assert missing == []


def test_no_document_links_to_the_old_name_of_adr_0004() -> None:
    documents = [*REPO_ROOT.glob("*.md"), *(REPO_ROOT / "docs").rglob("*.md")]
    assert [path for path in documents if "0004-claude" in _read(path)] == []


def test_adr_0004_is_named_for_what_it_decides_and_the_index_links_to_it() -> None:
    assert sorted(path.name for path in ADR_DIRECTORY.glob("0004-*.md")) == [ADR_4.name]
    index = _read(ADR_DIRECTORY / "README.md")
    assert f"| [0004]({ADR_4.name}) |" in index
    assert "Gemini 3.1 Flash-Lite" in index
    assert _read(ADR_4).startswith(
        "# ADR-0004: A model on Vertex AI in the EU, chosen by evaluation"
    )


def test_adr_0004_names_code_and_routes_that_exist() -> None:
    text = _read(ADR_4)
    assert "VERTEX_HOSTS" in text
    assert "VERTEX_ENDPOINTS" not in text
    assert hasattr(factory, "VERTEX_HOSTS")
    named = {name for name in BACKTICKED.findall(text) if name.startswith("vertex-")}
    assert {"vertex-eu-gemini-flash-lite", "vertex-eu-opus"} <= named <= set(ROUTES)
    gemini = ROUTES["vertex-eu-gemini-flash-lite"]
    assert f"`{gemini.model_name}`" in text
    assert factory.VERTEX_HOSTS[gemini.region or ""] in text
    modules = [
        name for name in BACKTICKED.findall(text) if re.fullmatch(r"frontdesk(\.[a-z_]+)+", name)
    ]
    assert modules
    for name in modules:
        importlib.import_module(name)


def test_adr_0005_names_the_candidates_the_experiment_compares() -> None:
    text = _read(ADR_5)
    candidates = {
        route.model_name: route
        for route in ROUTES.values()
        if route.provider == "google-vertex-embedding"
    }
    assert len(candidates) == 2
    for model_name, route in candidates.items():
        assert f"`{model_name}`" in text
        assert route.region == "europe-west1"
    assert "`frontdesk eval embedders`" in text
    assert "at least 10 points" in text


def test_the_readme_names_only_commands_the_program_has() -> None:
    mentioned = set(COMMAND.findall(_read(REPO_ROOT / "README.md")))
    assert {"chat", "eval turns", "eval report", "eval embedders", "eval retrieval"} <= mentioned
    assert mentioned <= _commands()


def test_the_readme_tells_the_state_of_m3() -> None:
    readme = _read(REPO_ROOT / "README.md")
    row = next(line for line in readme.splitlines() if line.startswith("| M3 |"))
    assert "Gemini 3.1 Flash-Lite" in row
    assert row.endswith("| In progress |")
    assert "Claude on Vertex AI" not in readme


def test_the_env_example_shows_every_setting_and_loads() -> None:
    shown = set(re.findall(r"^#?\s*(FRONTDESK_[A-Z_]+)=", _read(ENV_EXAMPLE), re.MULTILINE))
    assert shown == {f"FRONTDESK_{name.upper()}" for name in Settings.model_fields}
    settings = load_settings(ENV_EXAMPLE)  # every value it sets is one the program accepts
    assert settings.min_retrieval_similarity == 0.0  # the floor is off until an embedder is tuned
    assert "FRONTDESK_MIN_RETRIEVAL_SIMILARITY=0.0" in _read(ENV_EXAMPLE).splitlines()


def test_the_env_example_says_the_project_is_for_every_live_route() -> None:
    line = next(line for line in _read(ENV_EXAMPLE).splitlines() if "FRONTDESK_GCP_PROJECT" in line)
    assert "only for live routes (ADR-0004)" in line
