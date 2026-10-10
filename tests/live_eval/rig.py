"""A live run that a test can start again as often as it needs, on the real turn code with no
network: one run directory and one ledger, a classifier that counts its requests."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.records import RunFiles
from frontdesk.live_eval.runner import RunProgress, run_live_eval
from frontdesk.model_gateway.metering import Meter
from frontdesk.tenancy.pack import BusinessPack
from tests.conversation.helpers import document_responder, make_runtime
from tests.model_gateway.budget_helpers import ROUTE, write_budget

Activate = Callable[..., sessionmaker[Session]]
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
CAP = {"max_tokens": 600}
CASES = [
    TurnCase.model_validate(spec)
    for spec in (
        {"id": "one", "message": "Quanto costa la fisioterapia?", "expect": {}},
        {
            "id": "two",
            "messages": ["Quanto costa la fisioterapia?", "E per favore?"],
            "expect": {},
        },
        {"id": "three", "message": "Quanto costa la fisioterapia?", "expect": {}},
    )
]


class Rig:
    """One run directory and one ledger, started again as often as a test needs. Each start has a
    new meter, as a new process would; the classifier counts its requests, and `on_request` runs
    something at the n-th one (it happens while that request is away). The something may raise,
    to fail the request; it may be a coroutine, to keep the request away; and when it returns a
    classification, that is the answer to the request instead of the price."""

    def __init__(
        self,
        pack: BusinessPack,
        factory: sessionmaker[Session],
        tmp_path: Path,
        on_request: dict[int, Callable[[], Any]] | None = None,
        *,
        cap_eur: float = 10.0,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.pack = pack
        self.factory = factory
        self.budget = write_budget(tmp_path, cap_eur=cap_eur)
        self.files = RunFiles(tmp_path / "run-1")
        self.files.directory.mkdir()
        self.requests: list[str] = []
        self.meter: Meter | None = None
        self._on_request = on_request or {}
        self._timeout_seconds = timeout_seconds

    async def _classifier(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append("classifier")
        answer = PRICE
        if (action := self._on_request.get(len(self.requests))) is not None:
            outcome = action()
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if isinstance(outcome, dict):
                answer = outcome
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    async def start(
        self,
        cases: list[TurnCase] = CASES,
        *,
        limit: int | None = None,
        breaker: int | None = None,
    ) -> RunProgress:
        self.meter = meter = Meter(self.budget, route_id=ROUTE, run_id="run-1", breaker=breaker)
        runtime = make_runtime(
            self.pack,
            self.factory,
            meter.wrap(FunctionModel(self._classifier), "classifier"),
            meter.wrap(document_responder(), "responder"),
            model_settings=CAP,
            timeout_seconds=self._timeout_seconds,
        )
        return await run_live_eval(
            runtime, cases, files=self.files, run_id="run-1", meter=meter, limit=limit
        )

    def ledger_lines(self) -> list[str]:
        return self.budget.ledger_path.read_text(encoding="utf-8").splitlines()
