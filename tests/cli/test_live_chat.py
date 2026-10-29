"""A chat with the real model on a demo pack, metered against the budget (M3 Task 6b).

No test here calls a model or reaches the network. The model that `build_live_model` would return is
a fake, wrapped by the same `wrap_model` path as a real one; where a real google-genai client is
built, nothing is sent to it (`models.ALLOW_MODEL_REQUESTS` is off, and no test types a message).
The shared helpers are in `live_chat_helpers.py` and the `gemini` fixture in `conftest.py`. The
lines the chat prints and the loop that prints them are in `test_live_costs.py`; what a person
sees when something goes wrong is in `test_live_chat_failures.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel

from frontdesk.cli import runtime as cli_runtime
from frontdesk.cli.main import app
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.model_gateway.costs import load_budget
from frontdesk.model_gateway.metering import Meter, MeteredModel
from frontdesk.model_gateway.routes import DataClass
from frontdesk.settings import load_settings
from tests.cli.live_chat_helpers import (
    BANNER,
    PARKING,
    PRICE,
    QUESTION,
    RUN_ID,
    USAGE,
    Install,
    bootstrapped,
    env_file,
    ledger,
    live,
    refused,
    runner,
)
from tests.model_gateway.budget_helpers import ROUTE, write_budget

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- The command refuses before it spends -----------------------------------------------------------


def test_live_needs_a_budget(tmp_path: Path, businesses_dir: Path) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    result = runner.invoke(app, ["chat", "--live", "--env-file", str(env)], input="Ciao\n")
    assert refused(result).startswith("error: --live needs --budget")


def test_a_budget_goes_with_live_only(tmp_path: Path, businesses_dir: Path) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", route="offline-demo", project=None)
    budget = write_budget(tmp_path)
    result = runner.invoke(
        app, ["chat", "--budget", str(budget.path), "--env-file", str(env)], input="Ciao\n"
    )
    assert refused(result) == "error: --budget only goes with --live"


def test_a_budget_file_that_is_missing_is_an_error_and_not_a_traceback(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    missing = tmp_path / "nowhere" / "budget.yaml"
    result = runner.invoke(
        app, ["chat", "--live", "--budget", str(missing), "--env-file", str(env)], input="Ciao\n"
    )
    assert refused(result) == f"error: {missing}: cannot be read (No such file or directory)"


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param("cap_eur: [", "invalid YAML", id="not-yaml"),
        pytest.param("cap_eur: -1\n", "cap_eur: Input should be greater than 0", id="bad-cap"),
        pytest.param("- nothing\n- useful\n", "Input should be a valid dictionary", id="not-a-map"),
    ],
)
def test_a_budget_file_that_is_wrong_is_an_error_and_not_a_traceback(
    tmp_path: Path, businesses_dir: Path, content: str, problem: str
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    broken = tmp_path / "broken.yaml"
    broken.write_text(content, encoding="utf-8")
    result = runner.invoke(
        app, ["chat", "--live", "--budget", str(broken), "--env-file", str(env)], input="Ciao\n"
    )
    line = refused(result)
    assert line.startswith(f"error: {broken}: ")
    assert problem in line


def test_a_budget_that_does_not_price_the_route_is_an_error(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", route="vertex-eu-opus")
    budget = write_budget(tmp_path)  # prices the Gemini route and an embedder, not Claude
    line = refused(live(env, budget, input="Ciao\n"))
    assert line == f"error: {budget.path}: no price for route 'vertex-eu-opus'"


@pytest.mark.parametrize(
    "route", ["offline-demo", "vertex-ew1-gemini-embedding-001"], ids=["offline", "embedding"]
)
def test_a_route_that_answers_no_live_message_is_refused(
    tmp_path: Path, businesses_dir: Path, route: str
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", route=route, project=None)
    line = refused(live(env, write_budget(tmp_path), input="Ciao\n"))
    assert line == f"error: --live needs a live model route; FRONTDESK_MODEL_ROUTE is {route}"


def test_a_route_nobody_registered_is_refused_by_name(tmp_path: Path, businesses_dir: Path) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", route="openai-us")
    line = refused(live(env, write_budget(tmp_path), input="Ciao\n"))
    assert line == f"error: model route 'openai-us' is not registered (--env-file {env})"


def test_a_second_session_is_refused_while_the_first_holds_the_budget(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    budget = write_budget(tmp_path)
    with budget.exclusive():  # the first session
        line = refused(live(env, budget, input="Ciao\n"))
    assert line == (
        f"error: {budget.ledger_path}: another live run is using this budget; "
        "run this one when it has ended"
    )
    with budget.exclusive():  # the refused one took nothing: the budget is free again
        pass


def test_without_live_the_gemini_route_is_still_refused_for_chat(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    result = runner.invoke(app, ["chat", "--env-file", str(env)], input="Ciao\n")
    assert result.exit_code == 1
    assert "not approved for patient_derived" in result.stderr
    assert "Live:" not in result.output  # nothing was metered, nothing was live


# Rich styles the help on a CI terminal (GitHub Actions): read it without the escape codes.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_the_help_names_the_live_options() -> None:
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    help_text = ANSI_ESCAPE.sub("", result.stdout)
    assert "--live" in help_text
    assert "--budget" in help_text


# --- The env files ----------------------------------------------------------------------------------


@pytest.mark.parametrize("business", ["clinic", "shop"])
def test_a_live_env_file_is_the_demo_file_with_the_gemini_route_and_no_project(
    business: str,
) -> None:
    demo = REPO_ROOT / "deploy" / f"demo-{business}.env"
    live = REPO_ROOT / "deploy" / f"demo-{business}-live.env"
    expected = load_settings(demo).model_dump() | {"model_route": ROUTE}
    assert load_settings(live).model_dump() == expected  # same database, pack and embedder
    assert expected["gcp_project"] is None
    lines = live.read_text(encoding="utf-8").splitlines()
    assignments = [line for line in lines if not line.lstrip().startswith("#")]
    assert not any("GCP_PROJECT" in line for line in assignments)
    assert any(line.startswith("#") and "FRONTDESK_GCP_PROJECT" in line for line in lines)


# --- The README -------------------------------------------------------------------------------------


def _readme_section() -> str:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Try it with the real model")
    return text[start : text.index("\n## ", start + 1)]


def test_the_budget_file_the_readme_shows_is_a_valid_budget_for_the_live_route(
    tmp_path: Path,
) -> None:
    example = re.search(r"```yaml\n(.*?)```", _readme_section(), re.DOTALL)
    assert example, "the section shows no budget file"
    path = tmp_path / "budget.yaml"
    path.write_text(example.group(1), encoding="utf-8")
    budget = load_budget(path)  # a wrong field or a missing price would raise
    assert budget.price(ROUTE).unit == "tokens_1m"
    assert budget.config.cap_eur > 0


def test_the_readme_says_what_the_chat_needs_and_where_the_budget_file_lives() -> None:
    section = _readme_section()
    assert "private pack repository" in section
    assert "`runs/budget.yaml`" in section
    for part in ("cap_eur", "ledger", "exchange_rate", "prices"):
        assert f"`{part}`" in section, f"the section does not name `{part}`"
    # The Google Cloud CLI is listed as a prerequisite, before the first command that needs it.
    assert 0 <= section.index("gcloud") < section.index("```bash")


# --- A session --------------------------------------------------------------------------------------


@pytest.mark.db
def test_a_session_shows_the_banner_each_reply_and_what_it_cost(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=2.5)  # 2.00 EUR spent before

    result = live(env, budget, input=f"{QUESTION}\n{QUESTION}\nexit\n")

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    banner = BANNER.format(path=budget.path, left="8.00", cap="10.00")
    replies = [number for number, line in enumerate(lines) if line.startswith("bot> ")]
    assert len(replies) == 2
    assert lines.index(banner) < replies[0]  # before the first prompt
    assert "Fisioterapia: 30 EUR" in result.stdout  # the reply is a multi-line message
    assert [line for line in lines if line.startswith("cost: ")] == [
        "cost: 0.0104 EUR \N{MIDDLE DOT} session 0.0104 EUR \N{MIDDLE DOT} left 7.99 EUR",
        "cost: 0.0104 EUR \N{MIDDLE DOT} session 0.0208 EUR \N{MIDDLE DOT} left 7.98 EUR",
    ]
    # The session proves the ledger first (a line of zero dollars), then one line per paid
    # request, all under the one run of this session.
    earlier, proof, *paid = ledger(budget)
    assert earlier["run"] == "earlier"
    assert (proof["route"], proof["kind"], proof["usd"]) == (ROUTE, "preflight", 0.0)
    assert len(paid) == 2
    for entry in paid:
        assert (entry["route"], entry["kind"]) == (ROUTE, "chat")
        assert entry["usd"] == pytest.approx(0.013)
    assert len({entry["run"] for entry in [proof, *paid]}) == 1
    assert RUN_ID.fullmatch(paid[0]["run"])
    assert sent == ["classifier", "classifier"]


@pytest.mark.db
def test_the_cost_line_comes_after_the_reply_and_its_trace(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    result = live(env, write_budget(tmp_path), input=f"{QUESTION}\nexit\n", trace=True)
    assert result.exit_code == 0, result.output
    kinds = [
        "reply"
        if line.startswith("bot> ")
        else "trace"
        if line.lstrip().startswith("[rule=")
        else "cost"
        if line.startswith("cost: ")
        else "other"
        for line in result.stdout.splitlines()
    ]
    assert [kind for kind in kinds if kind != "other"] == ["reply", "trace", "cost"]


@pytest.mark.db
def test_both_agents_are_metered_under_their_own_roles(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles: list[str] = []
    wrap = Meter.wrap

    def spy(self: Meter, model: Model, role: str) -> Model:
        roles.append(role)
        return wrap(self, model, role)

    monkeypatch.setattr(Meter, "wrap", spy)
    sent = gemini(PARKING)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)

    result = live(env, budget, input="Is there parking near the clinic?\nexit\n")

    assert result.exit_code == 0, result.output
    assert "Klinika Demo has six free parking spaces behind the building." in result.stdout
    assert roles == ["classifier", "responder"]
    assert sent == ["classifier", "responder"]
    assert [entry["kind"] for entry in ledger(budget)] == ["preflight", "chat", "chat"]
    assert "cost: 0.0208 EUR \N{MIDDLE DOT} session 0.0208 EUR" in result.stdout


@pytest.mark.db
@pytest.mark.parametrize("farewell", ["exit\n", "quit\n", "", "\n  \n"])
def test_exit_and_the_end_of_input_end_the_session_normally(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    farewell: str,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    result = live(env, write_budget(tmp_path), input=farewell)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert "Live: gemini-3.1-flash-lite" in result.stdout
    assert sent == []


@pytest.mark.db
def test_a_budget_with_no_room_stops_the_session_before_any_reply(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    # A request reserves 0.038 USD (32,000 tokens in, a cap of 600 out): there is no room for one.
    budget = write_budget(tmp_path, cap_eur=0.02, usd_per_eur=1.0)

    result = live(env, budget, input=f"{QUESTION}\n{QUESTION}\nexit\n")

    assert result.exit_code == 1
    # The refused request is the turn's failure (its exception type), then the stop itself.
    assert result.stderr.splitlines() == ["model error: BudgetExhaustedError", "stopped: budget"]
    assert "Live: gemini-3.1-flash-lite" in result.stdout  # the banner came first
    assert "bot>" not in result.output  # the handoff the failed turn fell back to is not shown
    assert result.output.splitlines()[-2:] == [
        "cost: 0.0000 EUR \N{MIDDLE DOT} session 0.0000 EUR \N{MIDDLE DOT} left 0.02 EUR",
        "stopped: budget",
    ]
    assert sent == []  # nothing was sent
    assert [entry["kind"] for entry in ledger(budget)] == ["preflight"]  # the proof, no spend
    assert budget.spent_usd() == 0.0  # nothing was spent
    with budget.exclusive():  # and the budget was given back
        pass


@pytest.mark.db
def test_the_session_ends_when_the_budget_runs_out_in_the_middle(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    # A request reserves 0.038 USD and costs 0.013: the first fits under 0.039, the second does not.
    budget = write_budget(tmp_path, cap_eur=0.039, usd_per_eur=1.0)

    result = live(env, budget, input=f"{QUESTION}\n{QUESTION}\n{QUESTION}\n")

    assert result.exit_code == 1
    assert result.stderr.splitlines() == ["model error: BudgetExhaustedError", "stopped: budget"]
    replies = [line for line in result.stdout.splitlines() if line.startswith("bot> ")]
    assert len(replies) == 1  # the second turn raised because of the stop: no reply
    assert result.stdout.count("cost: 0.0130 EUR") == 1  # the first turn's
    # The second turn paid nothing, and says so before the stop.
    assert result.output.splitlines()[-2:] == [
        "cost: 0.0000 EUR \N{MIDDLE DOT} session 0.0130 EUR \N{MIDDLE DOT} left 0.03 EUR",
        "stopped: budget",
    ]
    assert sent == ["classifier"]  # the third line was never read
    assert [entry["kind"] for entry in ledger(budget)] == ["preflight", "chat"]


@pytest.mark.db
def test_an_attempt_that_may_have_been_billed_is_a_ledger_line_and_part_of_the_cost(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("frontdesk.model_gateway.metering.RETRY_WAITS", (0.0, 0.0, 0.0))
    attempts: list[str] = []

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts.append("attempt")
        if len(attempts) == 1:
            raise httpx2.ReadTimeout("no answer after the request went out")
        call = ToolCallPart(info.output_tools[0].name, PRICE)
        return ModelResponse(parts=[call], usage=USAGE)

    model = FunctionModel(flaky)
    monkeypatch.setattr(
        "frontdesk.composition.build_live_model", lambda route, *, gcp_project: model
    )
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.0)

    result = live(env, budget, input=f"{QUESTION}\nexit\n")

    assert result.exit_code == 0, result.output
    assert len(attempts) == 2  # the timeout was retried
    # The first attempt was sent and may have been billed in full: it counts at its reserve (32,000
    # tokens in, a cap of 600 out: 0.038 USD). The second counts what Google reported (0.013).
    proof, first, second = ledger(budget)
    assert proof["kind"] == "preflight"
    assert (first["kind"], first["usd"]) == ("chat-unconfirmed", pytest.approx(0.038))
    assert (second["kind"], second["usd"]) == ("chat", pytest.approx(0.013))
    assert "cost: 0.0510 EUR \N{MIDDLE DOT} session 0.0510 EUR \N{MIDDLE DOT} left 9.95 EUR" in (
        result.stdout
    )


@pytest.mark.db
def test_a_ledger_that_breaks_during_the_session_ends_it_with_an_error(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)

    def corrupting(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        budget.ledger_path.write_text("not a spend entry\n", encoding="utf-8")
        call = ToolCallPart(info.output_tools[0].name, PRICE)
        return ModelResponse(parts=[call], usage=USAGE)

    model = FunctionModel(corrupting)
    monkeypatch.setattr(
        "frontdesk.composition.build_live_model", lambda route, *, gcp_project: model
    )

    result = live(env, budget, input=f"{QUESTION}\nexit\n")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert result.stderr.splitlines() == [
        f"error: {budget.ledger_path}: line 1 is not a spend entry"
    ]
    assert "Fisioterapia: 30 EUR" in result.stdout  # the paid reply was shown first


@pytest.mark.db
def test_a_missing_project_is_an_error_and_not_a_traceback(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database, project=None)
    budget = write_budget(tmp_path)

    result = live(env, budget, input="Ciao\n")

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""  # no banner for a session that did not start
    assert result.stderr.splitlines() == [
        "error: FRONTDESK_GCP_PROJECT is required for the vertex-eu-gemini-flash-lite route"
    ]
    assert not budget.ledger_path.exists()
    with budget.exclusive():
        pass


@pytest.mark.db
def test_without_active_knowledge_the_error_says_what_to_run(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic", database)
    result = live(env, write_budget(tmp_path), input="Ciao\n")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "business 'demo-clinic' has no active knowledge version" in result.stderr
    assert f"frontdesk knowledge bootstrap-demo --env-file {env}" in result.stderr


class OtherEmbedder(HashEmbedder):
    identity = "other-v1"


@pytest.mark.db
def test_knowledge_embedded_by_another_model_is_an_error_that_says_what_to_run(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = bootstrapped(tmp_path, businesses_dir, database)
    monkeypatch.setattr("frontdesk.composition.build_embedder", lambda backend: OtherEmbedder())
    result = live(env, write_budget(tmp_path), input="Ciao\n")
    assert result.exit_code == 1
    assert result.stdout == ""
    [line] = result.stderr.splitlines()
    assert line.startswith("error: active knowledge ")
    assert "this deployment uses 'other-v1'" in line
    assert line.endswith(f"run `frontdesk knowledge reembed` --env-file {env}")


@pytest.mark.db
def test_the_chat_writes_no_transcript_and_the_ledger_holds_amounts_only(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    before = {path.name for path in tmp_path.iterdir()}
    monkeypatch.chdir(tmp_path)  # a stray file would land here

    result = live(env, budget, input=f"{QUESTION}\nexit\n")

    assert result.exit_code == 0, result.output
    created = {path.name for path in tmp_path.iterdir()} - before
    assert created == {"spend.jsonl", "spend.jsonl.lock"}
    proof, entry = ledger(budget)  # the zero-dollar proof, then the one paid request
    for line in (proof, entry):
        assert set(line) == {"at", "run", "route", "kind", "usd"}
    assert "fisioterapia" not in budget.ledger_path.read_text(encoding="utf-8").lower()


@pytest.mark.db
def test_the_runtime_is_built_for_evaluation_data_only_when_live(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[dict[str, Any], TurnRuntime]] = []
    real = cli_runtime.build_turn_runtime

    def spy(deployment: Any, **options: Any) -> TurnRuntime:
        runtime = real(deployment, **options)
        built.append((options, runtime))
        return runtime

    # Both commands build their pipeline through the one shared builder.
    monkeypatch.setattr(cli_runtime, "build_turn_runtime", spy)
    live_env = bootstrapped(tmp_path, businesses_dir, database)
    offline_env = env_file(
        tmp_path, businesses_dir / "demo-clinic", database, name="offline", route="offline-demo"
    )

    # The real builder runs here, so a real google-genai client is built; the person types
    # nothing, so nothing is ever sent to it.
    live_run = live(live_env, write_budget(tmp_path), input="exit\n")
    offline_run = runner.invoke(app, ["chat", "--env-file", str(offline_env)], input="exit\n")

    assert (live_run.exit_code, offline_run.exit_code) == (0, 0)
    [(live_options, live_runtime), (offline_options, _)] = built
    assert live_options["data_class"] is DataClass.EVALUATION
    assert live_runtime.capture_content is False  # telemetry stays metadata-only, as configured
    for agent in (live_runtime.classifier, live_runtime.responder):
        assert isinstance(agent.model, MeteredModel)
        assert agent.model.wrapped.model_name == "gemini-3.1-flash-lite"
    # Without --live the call names no data class: the default stays patient-derived.
    assert offline_options.get("data_class", DataClass.PATIENT_DERIVED) is DataClass.PATIENT_DERIVED
    assert offline_options.get("wrap_model") is None
