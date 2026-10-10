"""What the tests of `eval turns --live` on the fake Gemini share: cases, the command, and what a
run leaves behind. Nothing here calls a model or reaches the network."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from frontdesk.cli import live
from frontdesk.cli.main import app
from frontdesk.model_gateway.costs import Budget
from tests.cli.live_chat_helpers import USAGE, quote_the_first_passage, runner
from tests.conversation.helpers import first_prompt
from tests.model_gateway.budget_helpers import ROUTE

PRICE_QUESTION = "Quanto costa la fisioterapia?"
CHEST_PAIN = "Mi fa male il petto"


def write_cases(directory: Path, count: int = 2, *, name: str = "turns.yaml") -> Path:
    """`count` price questions, one message each. With the fake Gemini each costs one request, the
    classifier's: the price is read from the business's records, not written by a model."""
    path = directory / name
    path.write_text(
        "".join(
            f"- id: price-{number}\n"
            f'  message: "{PRICE_QUESTION}"\n'
            "  expect: {behaviour: answer, contains: ['30 EUR']}\n"
            for number in range(1, count + 1)
        ),
        encoding="utf-8",
    )
    return path


def write_mixed_cases(directory: Path) -> Path:
    """A price question (one request), an emergency (the rules decide: no request) and a tagged
    price question."""
    path = directory / "mixed.yaml"
    path.write_text(
        "- id: price-it\n"
        f'  message: "{PRICE_QUESTION}"\n'
        "  expect: {behaviour: answer, contains: ['30 EUR'], critical: true}\n"
        "- id: chest-it\n"
        f'  message: "{CHEST_PAIN}"\n'
        "  expect: {behaviour: emergency}\n"
        "- id: price-tagged\n"
        f'  message: "{PRICE_QUESTION}"\n'
        "  tags: [pricing]\n"
        "  expect: {behaviour: answer, contains: ['30 EUR']}\n",
        encoding="utf-8",
    )
    return path


def eval_live(
    env: Path,
    budget: Budget,
    run_dir: Path,
    cases: Path,
    *flags: str,
    route: str = ROUTE,
    yes: bool = True,
    input: str = "",
) -> Result:
    """`frontdesk eval turns --live` on `route`, paid from `budget`."""
    arguments = ["eval", "turns", "--env-file", str(env), "--live", "--route", route]
    arguments += ["--run-dir", str(run_dir), "--budget", str(budget.path), "--cases", str(cases)]
    if yes:
        arguments.append("--yes")
    return runner.invoke(app, [*arguments, *flags], input=input)


def eval_report(run_dir: Path, *flags: str) -> Result:
    return runner.invoke(app, ["eval", "report", "--run-dir", str(run_dir), *flags])


def recorded_cases(run_dir: Path) -> list[str]:
    """The case of each line of turns.jsonl, in order (a case of several turns repeats)."""
    path = run_dir / "turns.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["case_id"] for line in lines]


def one_error_line(result: Result, exit_code: int = 1) -> str:
    """The line a refusal printed on stderr: one line, no traceback, a clean exit."""
    assert result.exit_code == exit_code, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Traceback" not in result.output
    [line] = result.stderr.splitlines()
    return line


# What a request does before it is answered: given the role (classifier or responder) and the
# number of the request among that role's, it may raise, or return a coroutine to keep the request
# away (the turn limit then cuts it), or return nothing and let it be answered.
Behave = Callable[[str, int], Awaitable[None] | None]


def install_provider(
    monkeypatch: pytest.MonkeyPatch, classification: dict[str, Any], behave: Behave | None = None
) -> list[str]:
    """A fake Gemini where `build_live_model` would build the real one, like the `gemini` fixture,
    but one that can be kept away. Returns the role of each request it is sent."""
    sent: list[str] = []
    numbers: dict[str, int] = {}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        role = (
            "responder" if "claims" in tool.parameters_json_schema["properties"] else "classifier"
        )
        sent.append(role)
        numbers[role] = numbers.get(role, 0) + 1
        if behave is not None:
            outcome = behave(role, numbers[role])
            if inspect.isawaitable(outcome):
                await outcome
        args = (
            quote_the_first_passage(first_prompt(messages))
            if role == "responder"
            else classification
        )
        return ModelResponse(
            parts=[ToolCallPart(tool.name, args)], usage=USAGE, finish_reason="stop"
        )

    model = FunctionModel(respond)
    monkeypatch.setattr(
        "frontdesk.composition.build_live_model", lambda route, *, gcp_project: model
    )
    return sent


class TurnLimit:
    """The limit of a turn in seconds, which a test can lengthen again."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds


def quick_turns(monkeypatch: pytest.MonkeyPatch, seconds: float = 0.3) -> TurnLimit:
    """Shorten the limit of a turn (45 seconds), so that a request kept away is cut at once. Long
    enough for the pipeline's own work (a database read), short enough to wait for."""
    limit = TurnLimit(seconds)
    real = live.build_runtime

    def quick(*args: Any, **kwargs: Any) -> Any:
        return replace(real(*args, **kwargs), timeout_seconds=limit.seconds)

    monkeypatch.setattr("frontdesk.cli.live.build_runtime", quick)
    return limit


def failures_in(run_dir: Path) -> int:
    """How many cases of the run's report ended in a failure of the pipeline (a crash, a timeout, a
    model call that failed): the handoffs by failure that the report counts against the bot."""
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return int(report["languages"]["all"]["failed_by_pipeline"])


def rules_of(run_dir: Path) -> list[str]:
    """The rule of each recorded turn, in order."""
    path = run_dir / "turns.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [json.loads(line)["rule"] for line in lines]
