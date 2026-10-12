"""Each live request is reserved against the budget, then recorded at its real cost."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from google.auth.exceptions import RefreshError
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from frontdesk.model_gateway.costs import BudgetError, BudgetExhaustedError
from frontdesk.model_gateway.metering import MAX_INPUT_TOKENS_PER_REQUEST, Meter
from tests.model_gateway.attempt_helpers import Sleeps
from tests.model_gateway.budget_helpers import ROUTE, write_budget


def _answering(input_tokens: int, output_tokens: int, thoughts: int = 0) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            details={"thoughts_tokens": thoughts} if thoughts else {},
        )
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=usage,
            model_name="gemini-3.1-flash-lite",
            finish_reason="stop",
        )

    return FunctionModel(model)


def _clock(*ticks: float) -> Callable[[], float]:
    remaining = list(ticks)
    return lambda: remaining.pop(0)


async def test_a_request_is_recorded_with_its_tokens_cost_and_time(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", clock=_clock(10.0, 10.5))
    fake = _answering(2_000, 300, thoughts=40)
    await Agent(meter.wrap(fake, "classifier")).run("x", model_settings={"max_tokens": 600})
    [call] = meter.drain()
    # The version the provider reported; a FunctionModel reports its own name.
    assert (call.agent, call.model_version, call.finish_reason) == (
        "classifier",
        fake.model_name,
        "stop",
    )
    assert (call.input_tokens, call.output_tokens, call.thoughts_tokens) == (2_000, 300, 40)
    assert call.usd == pytest.approx((2_000 * 1.0 + 300 * 10.0) / 1_000_000)
    assert call.seconds == pytest.approx(0.5)
    assert budget.spent_usd() == pytest.approx(call.usd)
    assert meter.drain() == ()  # drained calls are gone
    assert meter.stop_reason is None


async def test_a_request_the_budget_cannot_cover_is_never_sent(tmp_path: Path) -> None:
    # Room for the reserve of a 600-token cap, not for a 100,000-token one.
    reserve_600 = (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + 600 * 10.0) / 1_000_000
    budget = write_budget(tmp_path, cap_eur=reserve_600 * 1.5, usd_per_eur=1.0)
    sent: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append("request")
        return ModelResponse(parts=[TextPart("ok")])

    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    agent = Agent(meter.wrap(FunctionModel(model), "responder"))
    await agent.run("x", model_settings={"max_tokens": 600})
    with pytest.raises(BudgetExhaustedError):
        await agent.run("x", model_settings={"max_tokens": 100_000})
    assert sent == ["request"]
    assert meter.stop_reason == "budget"


async def test_a_rate_limit_that_outlived_the_retries_stops_the_run(tmp_path: Path) -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(429, "gemini-3.1-flash-lite", body={"error": "RESOURCE_EXHAUSTED"})

    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    with pytest.raises(ModelHTTPError):
        await Agent(meter.wrap(FunctionModel(model), "classifier")).run(
            "x", model_settings={"max_tokens": 600}
        )
    assert meter.stop_reason == "rate_limited"
    # A refused request costs nothing: each attempt is a call of no cost, with its status.
    assert {(call.finish_reason, call.usd) for call in meter.drain()} == {("error 429", 0.0)}


async def test_other_provider_errors_do_not_stop_the_run(tmp_path: Path) -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(503, "gemini-3.1-flash-lite")

    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    with pytest.raises(ModelHTTPError):
        await Agent(meter.wrap(FunctionModel(model), "classifier")).run(
            "x", model_settings={"max_tokens": 600}
        )
    assert meter.stop_reason is None


def test_a_metered_model_keeps_the_name_of_the_model_it_wraps(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    wrapped = _answering(1, 1)
    assert meter.wrap(wrapped, "classifier").model_name == wrapped.model_name


@pytest.mark.parametrize(
    ("error", "charged"),
    [
        pytest.param(TimeoutError("no answer after the request went out"), True, id="read-timeout"),
        pytest.param(asyncio.CancelledError(), True, id="cancelled-turn"),
        pytest.param(RefreshError("expired"), False, id="no-credentials"),
        pytest.param(httpx2.ConnectError("no route to host"), False, id="no-connection"),
    ],
)
async def test_a_request_without_usage_is_charged_its_reserve_once_it_may_have_been_sent(
    tmp_path: Path, error: BaseException, charged: bool
) -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise error

    budget = write_budget(tmp_path, usd_per_eur=1.0)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    with pytest.raises(type(error)):
        await Agent(meter.wrap(FunctionModel(model), "classifier")).run(
            "x", model_settings={"max_tokens": 600}
        )
    reserve = (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + 600 * 10.0) / 1_000_000
    assert budget.spent_usd() == pytest.approx(reserve if charged else 0.0)
    calls = meter.drain()
    assert [call.finish_reason for call in calls if call.usd > 0] == (
        ["unconfirmed"] if charged else []
    )
    assert meter.stop_reason is None


def _cut_off(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise TimeoutError("no answer after the request went out")


async def test_the_cap_an_agent_is_built_with_sets_the_reserve(tmp_path: Path) -> None:
    # The live agents carry their output cap as settings of the agent (`live_model_settings`),
    # not of one run: the reserve must follow that cap too.
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    agent = Agent(
        meter.wrap(FunctionModel(_cut_off), "classifier"), model_settings={"max_tokens": 600}
    )
    with pytest.raises(TimeoutError):
        await agent.run("x")
    reserve = (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + 600 * 10.0) / 1_000_000
    assert budget.spent_usd() == pytest.approx(reserve)


async def test_the_ledger_tells_a_confirmed_request_from_an_unconfirmed_one(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    await Agent(meter.wrap(_answering(10, 5), "classifier")).run(
        "x", model_settings={"max_tokens": 600}
    )
    with pytest.raises(TimeoutError):
        await Agent(meter.wrap(FunctionModel(_cut_off), "responder")).run(
            "x", model_settings={"max_tokens": 600}
        )
    entries = [
        json.loads(line) for line in budget.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(entry["kind"], entry["run"], entry["route"]) for entry in entries] == [
        ("chat", "run-1", ROUTE),
        ("chat-unconfirmed", "run-1", ROUTE),
    ]
    assert [(call.agent, call.finish_reason) for call in meter.drain()] == [
        ("classifier", "stop"),
        ("responder", "unconfirmed"),
    ]


async def test_a_metered_model_does_not_stream(tmp_path: Path) -> None:
    metered = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1").wrap(
        _answering(1, 1), "classifier"
    )
    with pytest.raises(NotImplementedError, match="does not stream"):
        async with metered.request_stream([], None, ModelRequestParameters()):
            pytest.fail("a metered model must not open a stream")


def test_a_meter_needs_a_price_for_its_route(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="no price for route 'vertex-eu-opus'"):
        Meter(write_budget(tmp_path), route_id="vertex-eu-opus", run_id="run-1")
