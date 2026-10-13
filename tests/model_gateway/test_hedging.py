"""Hedged requests (Albi, 2026-10-12): a request still away after a few seconds gets an identical
twin; the first answer wins, the other is cancelled, and a meter records the loser at the winner's
input and its output cap, as no failure."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from frontdesk.model_gateway.hedging import HedgedModel
from frontdesk.model_gateway.metering import Meter
from tests.model_gateway.budget_helpers import ROUTE, write_budget

CAP = {"max_tokens": 600}
AFTER = 0.05


def _model(
    delays: list[float], calls: list[int], *, fail: frozenset[int] = frozenset()
) -> FunctionModel:
    """Answers each request after its delay, in the order the requests come; `fail` raises."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        number = len(calls)
        calls.append(number)
        await asyncio.sleep(delays[number])
        if number in fail:
            raise RuntimeError(f"request {number} failed")
        usage = RequestUsage(input_tokens=1000, output_tokens=50)
        return ModelResponse(parts=[TextPart(f"answer {number}")], usage=usage)

    return FunctionModel(respond)


async def test_a_quick_answer_sends_no_twin() -> None:
    calls: list[int] = []
    result = await Agent(HedgedModel(_model([0.0], calls), after=AFTER)).run("hi")
    assert (result.output, calls) == ("answer 0", [0])


async def test_a_stuck_request_loses_to_its_twin() -> None:
    calls: list[int] = []
    result = await Agent(HedgedModel(_model([5.0, 0.0], calls), after=AFTER)).run("hi")
    assert (result.output, calls) == ("answer 1", [0, 1])


async def test_the_first_answer_wins_whichever_it_is() -> None:
    calls: list[int] = []
    result = await Agent(HedgedModel(_model([0.2, 5.0], calls), after=AFTER)).run("hi")
    assert result.output == "answer 0"


async def test_a_failure_waits_for_the_twin_and_two_failures_raise() -> None:
    calls: list[int] = []
    mended = HedgedModel(_model([0.1, 0.3], calls, fail=frozenset({0})), after=AFTER)
    assert (await Agent(mended).run("hi")).output == "answer 1"
    both = HedgedModel(_model([0.1, 0.2], [], fail=frozenset({0, 1})), after=AFTER)
    with pytest.raises(RuntimeError, match="request 0 failed"):
        await Agent(both).run("hi")


async def test_a_meter_records_the_lost_twin_at_its_bound_and_no_failure(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", breaker=1)
    calls: list[int] = []
    hedged = HedgedModel(meter.wrap(_model([5.0, 0.0], calls), "agent"), after=AFTER)
    result = await Agent(hedged).run("hi", model_settings=CAP)
    assert result.output == "answer 1"
    lines = [json.loads(line) for line in budget.ledger_path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["chat", "chat-hedged"]
    price = budget.price(ROUTE)
    assert lines[1]["usd"] == pytest.approx(price.cost(input_units=1000, output_units=600))
    assert meter.stop_reason is None  # a lost twin is no failure: the breaker of one did not trip
    assert [call.finish_reason for call in meter.drain()] == [None, "hedged"]
