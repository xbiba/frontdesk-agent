"""What the live chat tests share: env files, a database filled by `bootstrap-demo`, a fake Gemini
and the lines a session leaves behind. Nothing here calls a model or reaches the network."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from click.testing import Result
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from typer.testing import CliRunner

from frontdesk.cli.main import app
from frontdesk.model_gateway.costs import Budget
from tests.conversation.helpers import PASSAGE, first_prompt
from tests.model_gateway.budget_helpers import ROUTE

runner = CliRunner()
NOWHERE = "postgresql+psycopg://nobody:x@127.0.0.1:1/none"  # never connected to
# What a fake Gemini reports for each request; at the test prices (1 and 10 US dollars per million
# tokens) that is 0.013 USD, or 0.0104 EUR at 1.25 USD per EUR.
USAGE = RequestUsage(input_tokens=8_000, output_tokens=500)
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
PARKING = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Is there free parking?",
}
QUESTION = "Quanto costa la fisioterapia?"
BANNER = (
    "Live: gemini-3.1-flash-lite on Vertex AI eu, metered against {path} "
    "({left} EUR left of {cap} EUR). Test messages only: never type a real patient's data."
)
RUN_ID = re.compile(r"chat-\d{8}T\d{6}Z")


def env_file(
    tmp_path: Path,
    pack_dir: Path,
    urls: tuple[str, str] = (NOWHERE, NOWHERE),
    *,
    name: str = "live",
    route: str = ROUTE,
    project: str | None = "demo-project",
) -> Path:
    """An env file like `deploy/demo-clinic-live.env`; the project is optional, as it is there."""
    owner_url, app_url = urls
    lines = [
        f"FRONTDESK_BUSINESS_PACK={pack_dir}",
        f"FRONTDESK_DATABASE_URL={app_url}",
        f"FRONTDESK_MIGRATION_DATABASE_URL={owner_url}",
        f"FRONTDESK_MODEL_ROUTE={route}",
        "FRONTDESK_EMBEDDING_BACKEND=hash",
    ]
    if project is not None:
        lines.append(f"FRONTDESK_GCP_PROJECT={project}")
    env = tmp_path / f"{name}.env"
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def live(env: Path, budget: Budget, *flags: str, input: str = "", trace: bool = False) -> Result:
    arguments = ["chat", "--live", "--budget", str(budget.path), "--env-file", str(env)]
    arguments.append("--trace" if trace else "--no-trace")
    return runner.invoke(app, [*arguments, *flags], input=input)


def refused(result: Result) -> str:
    """The one line a refusal printed: on stderr, with no output and no traceback."""
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.stdout == ""
    assert "Traceback" not in result.output
    [line] = result.stderr.splitlines()
    return line


def fake_gemini(classification: dict[str, Any], sent: list[str]) -> FunctionModel:
    """Stands for Gemini: the classifier gets `classification`, the responder quotes its first
    passage. `sent` collects the role of each request."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        if "claims" in tool.parameters_json_schema["properties"]:
            sent.append("responder")
            args = _quote_the_first_passage(first_prompt(messages))
        else:
            sent.append("classifier")
            args = classification
        return ModelResponse(
            parts=[ToolCallPart(tool.name, args)], usage=USAGE, finish_reason="stop"
        )

    return FunctionModel(respond)


def _quote_the_first_passage(prompt: str) -> dict[str, Any]:
    found = PASSAGE.search(prompt)
    if found is None:
        return {"action": "handoff"}
    passage_id, content = found.groups()
    sentence = content.split(". ")[0].rstrip(".")
    claim = {"text": f"{sentence}.", "evidence_id": passage_id, "quote": sentence}
    return {"action": "answer", "claims": [claim]}


def bootstrapped(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    *,
    project: str | None = "demo-project",
) -> Path:
    """The live env file of the clinic, on a database that `bootstrap-demo` has filled, as the
    README does it with that same file."""
    env = env_file(tmp_path, businesses_dir / "demo-clinic", database, project=project)
    done = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", str(env)])
    assert done.exit_code == 0, done.output
    return env


def ledger(budget: Budget) -> list[dict[str, Any]]:
    text = budget.ledger_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]
