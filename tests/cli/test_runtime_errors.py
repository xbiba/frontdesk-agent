"""One wording for each way the turn pipeline fails to build, whichever command asked (M3 Task 8,
ruling 5): `chat`, `eval turns` and their live forms used to keep a copy of the mapping each.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import typer
from click.testing import Result
from sqlalchemy.exc import OperationalError

from frontdesk.cli import runtime as cli_runtime
from frontdesk.cli.main import app
from frontdesk.cli.runtime import build_runtime, database_location, fail, without_knowledge
from frontdesk.composition import Deployment, NoActiveKnowledgeError
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.versions import EmbedderMismatchError
from frontdesk.model_gateway.routes import DataClass, UnapprovedRouteError
from tests.cli.live_chat_helpers import NOWHERE, env_file, runner
from tests.cli.live_eval_helpers import eval_live, write_cases
from tests.model_gateway.budget_helpers import write_budget

ENV = Path("clinic.env")
DEPLOYMENT = cast(
    Deployment,
    SimpleNamespace(
        settings=SimpleNamespace(
            database_url="postgresql+psycopg://frontdesk_app:s3cret-pw@db.internal:6543/frontdesk"
        )
    ),
)
NO_KNOWLEDGE_LINE = (
    "error: business 'demo-clinic' has no active knowledge version. Draft, approve and activate a "
    "version first (demo businesses: frontdesk knowledge bootstrap-demo --env-file {env})."
)
DATABASE_DOWN_LINE = (
    "error: cannot reach the database (127.0.0.1:1); "
    "start it with: docker compose up -d --wait postgres"
)


def _raising(error: Exception) -> Callable[..., TurnRuntime]:
    def build(deployment: Deployment, **options: Any) -> TurnRuntime:
        raise error

    return build


@pytest.mark.parametrize(
    ("error", "line"),
    [
        pytest.param(
            NoActiveKnowledgeError("demo-clinic"),
            NO_KNOWLEDGE_LINE.format(env=ENV),
            id="no-knowledge",
        ),
        pytest.param(
            EmbedderMismatchError("active knowledge was embedded by 'a'; run `reembed`"),
            "error: active knowledge was embedded by 'a'; run `reembed` --env-file clinic.env",
            id="another-embedder",
        ),
        pytest.param(
            UnapprovedRouteError("model route 'x' is not approved for patient_derived"),
            "error: model route 'x' is not approved for patient_derived (--env-file clinic.env)",
            id="route-not-approved",
        ),
        pytest.param(
            OperationalError("SELECT 1", {}, Exception("connection to server at db failed")),
            "error: cannot reach the database (db.internal:6543); "
            "start it with: docker compose up -d --wait postgres",
            id="database-down",
        ),
        pytest.param(
            ValueError("FRONTDESK_GCP_PROJECT is required for the x route"),
            "error: FRONTDESK_GCP_PROJECT is required for the x route",
            id="a-setting-the-route-needs",
        ),
        pytest.param(
            ValueError("prompt file is blank: agents/clinic/prompts/classifier.md"),
            "error: prompt file is blank: agents/clinic/prompts/classifier.md",
            id="a-blank-prompt",
        ),
    ],
)
def test_each_way_the_pipeline_fails_to_build_is_one_error_line_and_exit_1(
    error: Exception,
    line: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_runtime, "build_turn_runtime", _raising(error))
    with pytest.raises(typer.Exit) as raised:
        build_runtime(DEPLOYMENT, ENV)
    assert raised.value.exit_code == 1
    assert raised.value.__cause__ is error
    captured = capsys.readouterr()
    assert captured.err == line + "\n"
    assert captured.out == ""
    assert "s3cret-pw" not in captured.err  # the database's password is never printed


def test_a_failure_that_is_not_one_of_these_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_runtime, "build_turn_runtime", _raising(RuntimeError("a bug")))
    with pytest.raises(RuntimeError, match="a bug"):
        build_runtime(DEPLOYMENT, ENV)


def test_the_options_of_the_caller_reach_the_builder_and_its_runtime_comes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = cast(TurnRuntime, object())
    seen: list[tuple[Deployment, dict[str, Any]]] = []

    def build(deployment: Deployment, **options: Any) -> TurnRuntime:
        seen.append((deployment, options))
        return runtime

    monkeypatch.setattr(cli_runtime, "build_turn_runtime", build)
    wrap = cast(Any, lambda model, role: model)
    assert build_runtime(DEPLOYMENT, ENV) is runtime
    assert (
        build_runtime(DEPLOYMENT, ENV, data_class=DataClass.EVALUATION, wrap_model=wrap) is runtime
    )
    assert seen == [
        (DEPLOYMENT, {"data_class": DataClass.PATIENT_DERIVED, "wrap_model": None}),
        (DEPLOYMENT, {"data_class": DataClass.EVALUATION, "wrap_model": wrap}),
    ]


def test_fail_prints_the_line_and_gives_back_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    exit_ = fail("it did not work")
    assert (exit_.exit_code, capsys.readouterr().err) == (1, "error: it did not work\n")


def test_without_knowledge_says_what_to_run(capsys: pytest.CaptureFixture[str]) -> None:
    exit_ = without_knowledge(NoActiveKnowledgeError("shop"), Path("deploy/shop.env"))
    assert exit_.exit_code == 1
    assert capsys.readouterr().err == (
        "error: business 'shop' has no active knowledge version. Draft, approve and activate a "
        "version first (demo businesses: frontdesk knowledge bootstrap-demo "
        "--env-file deploy/shop.env).\n"
    )


# --- The same line from every command that builds the pipeline ---------------------------------------


class Commands:
    """The five ways to ask for the pipeline: both offline commands on an offline env file, and the
    three live ones on a live env file that names the same database."""

    def __init__(self, tmp_path: Path, pack: Path, urls: tuple[str, str]) -> None:
        self.tmp_path = tmp_path
        self.offline = env_file(tmp_path, pack, urls, name="offline", route="offline-demo")
        self.live = env_file(tmp_path, pack, urls, name="live")
        self.budget = write_budget(tmp_path)
        self.cases = write_cases(tmp_path, 1)

    def run(self) -> dict[str, Result]:
        live_args = ["--budget", str(self.budget.path), "--env-file", str(self.live)]
        return {
            "chat": runner.invoke(app, ["chat", "--env-file", str(self.offline)], input="exit\n"),
            "eval turns": runner.invoke(app, ["eval", "turns", "--env-file", str(self.offline)]),
            "chat --live": runner.invoke(app, ["chat", "--live", *live_args], input="exit\n"),
            "eval turns --live (dry)": eval_live(
                self.live, self.budget, self.tmp_path / "dry", self.cases, route="offline-demo"
            ),
            "eval turns --live": eval_live(
                self.live, self.budget, self.tmp_path / "run", self.cases
            ),
        }

    def lines(self) -> dict[str, str]:
        """What each printed, with the env file's name left out (the two kinds have two files)."""
        printed: dict[str, str] = {}
        for name, result in self.run().items():
            assert result.exit_code == 1, (name, result.output)
            [line] = result.stderr.splitlines()
            printed[name] = line.replace(str(self.offline), "ENV").replace(str(self.live), "ENV")
        return printed


@pytest.mark.db
def test_every_command_says_the_same_when_the_business_has_no_active_knowledge(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    commands = Commands(tmp_path, businesses_dir / "demo-clinic", database)
    expected = NO_KNOWLEDGE_LINE.format(env="ENV")
    assert commands.lines() == dict.fromkeys(
        ["chat", "eval turns", "chat --live", "eval turns --live (dry)", "eval turns --live"],
        expected,
    )


@pytest.mark.db
def test_every_command_says_the_same_when_the_knowledge_was_embedded_by_another_model(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = Commands(tmp_path, businesses_dir / "demo-clinic", database)
    done = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", str(commands.offline)])
    assert done.exit_code == 0, done.output

    class OtherEmbedder(HashEmbedder):
        identity = "other-v1"

    monkeypatch.setattr("frontdesk.composition.build_embedder", lambda backend: OtherEmbedder())

    lines = commands.lines()

    assert len(set(lines.values())) == 1, lines
    [line] = set(lines.values())
    assert line.startswith("error: active knowledge ")
    assert "this deployment uses 'other-v1'" in line
    assert line.endswith("run `frontdesk knowledge reembed` --env-file ENV")


def test_every_command_says_the_same_when_the_database_is_down(
    tmp_path: Path, businesses_dir: Path
) -> None:
    commands = Commands(tmp_path, businesses_dir / "demo-clinic", (NOWHERE, NOWHERE))
    assert commands.lines() == dict.fromkeys(
        ["chat", "eval turns", "chat --live", "eval turns --live (dry)", "eval turns --live"],
        DATABASE_DOWN_LINE,
    )


def test_database_location_shows_the_host_and_the_port_only() -> None:
    assert database_location("postgresql+psycopg://user:pw@db.example:5433/x") == "db.example:5433"
    assert database_location("postgresql+psycopg://user:pw@db.example/x") == "db.example"
    assert database_location("postgresql+psycopg://user:pw@/x") == "localhost"
