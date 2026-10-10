"""What the tests of `eval turns --live` on the fake Gemini share: cases, the command, and what a
run leaves behind. Nothing here calls a model or reaches the network."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import Result

from frontdesk.cli.main import app
from frontdesk.model_gateway.costs import Budget
from tests.cli.live_chat_helpers import runner
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
