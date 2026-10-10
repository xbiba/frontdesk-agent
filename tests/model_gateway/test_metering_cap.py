"""What the meter lets through under the cap: a request in flight holds its reserve until it is
settled, and a request whose cost cannot be bounded is never sent (M3 design §1, §4.4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx2
import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import Tool
from pydantic_ai.usage import RequestUsage

from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError
from frontdesk.model_gateway.metering import (
    MAX_INPUT_TOKENS_PER_REQUEST,
    Meter,
    UnboundedRequestError,
)
from tests.model_gateway.attempt_helpers import Sleeps
from tests.model_gateway.budget_helpers import ROUTE, write_budget

CAP = {"max_tokens": 600}
RESERVE = (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + 600 * 10.0) / 1_000_000  # 0.038 USD
SMALL = RequestUsage(input_tokens=100, output_tokens=10)


def _slow_model(inside: asyncio.Event, leave: asyncio.Event, sent: list[str]) -> FunctionModel:
    """A model whose answer waits until `leave` is set; `inside` is set when a request arrives."""

    async def away(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append("request")
        inside.set()
        await leave.wait()
        return ModelResponse(parts=[TextPart("ok")], usage=SMALL)

    return FunctionModel(away)


async def test_requests_in_flight_hold_their_reserve_against_the_cap(tmp_path: Path) -> None:
    # Room for one reserve, not for two. A check that does not hold lets both through (probe).
    budget = write_budget(tmp_path, cap_eur=RESERVE * 1.5, usd_per_eur=1.0)
    inside, leave, sent = asyncio.Event(), asyncio.Event(), []
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    agent = Agent(meter.wrap(_slow_model(inside, leave, sent), "classifier"))

    async def second() -> object:
        await inside.wait()  # the first request is away and holds its reserve
        # It leaves soon on its own, so that a meter without a hold fails this test, not hangs it.
        asyncio.get_running_loop().call_later(0.05, leave.set)
        return await agent.run("y", model_settings=CAP)

    async with asyncio.timeout(10):
        first, other = await asyncio.gather(
            agent.run("x", model_settings=CAP), second(), return_exceptions=True
        )
    assert isinstance(other, BudgetExhaustedError)
    assert not isinstance(first, BaseException)  # the first request was served
    assert sent == ["request"]  # the second never left
    assert meter.stop_reason == "budget"
    assert budget.spent_usd() <= budget.cap_usd
    assert meter.held_usd == 0.0  # settled: nothing is held any more


async def test_requests_in_flight_that_fit_together_are_both_sent(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=RESERVE * 2.5, usd_per_eur=1.0)
    inside, leave, sent = asyncio.Event(), asyncio.Event(), []
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    agent = Agent(meter.wrap(_slow_model(inside, leave, sent), "classifier"))

    async def second() -> object:
        await inside.wait()
        asyncio.get_running_loop().call_later(0.05, leave.set)
        return await agent.run("y", model_settings=CAP)

    async with asyncio.timeout(10):
        first, other = await asyncio.gather(
            agent.run("x", model_settings=CAP), second(), return_exceptions=True
        )
    assert not isinstance(first, BaseException)
    assert not isinstance(other, BaseException)
    assert sent == ["request", "request"]
    assert meter.held_usd == 0.0
    assert meter.stop_reason is None


def _ends_with(error: BaseException | None) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if error is not None:
            raise error
        return ModelResponse(parts=[TextPart("ok")], usage=SMALL)

    return FunctionModel(model)


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param(None, id="answered"),
        pytest.param(ModelHTTPError(400, "gemini-3.1-flash-lite"), id="status"),
        pytest.param(httpx2.ConnectError("no route to host"), id="no-connection"),
        pytest.param(TimeoutError("no answer"), id="timeout"),
        pytest.param(asyncio.CancelledError(), id="cancelled"),
    ],
)
async def test_every_way_a_request_ends_gives_its_hold_back(
    tmp_path: Path, ending: BaseException | None
) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    agent = Agent(meter.wrap(_ends_with(ending), "classifier"))
    if ending is None:
        await agent.run("x", model_settings=CAP)
    else:
        with pytest.raises(type(ending)):
            await agent.run("x", model_settings=CAP)
    assert meter.held_usd == 0.0


async def test_a_request_whose_line_cannot_be_written_gives_its_hold_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_write(*args: object, **kwargs: object) -> None:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Budget, "record", cannot_write)
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    with pytest.raises(OSError, match="Read-only"):
        await Agent(meter.wrap(_ends_with(None), "classifier")).run("x", model_settings=CAP)
    assert meter.held_usd == 0.0


async def test_a_hold_is_what_one_request_could_cost_and_is_gone_when_it_is_settled(
    tmp_path: Path,
) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    assert meter.held_usd == 0.0
    reserved = meter.reserve(600)
    assert reserved == pytest.approx(RESERVE)
    assert meter.held_usd == pytest.approx(RESERVE)
    meter.release(reserved)
    assert meter.held_usd == 0.0


def _counting() -> tuple[FunctionModel, list[str]]:
    sent: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append("request")
        return ModelResponse(parts=[TextPart("ok")], usage=SMALL)

    return FunctionModel(model), sent


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(None, id="no-settings"),
        pytest.param({}, id="no-cap"),
        pytest.param({"temperature": 0.0}, id="other-settings"),
        pytest.param({"max_tokens": 0}, id="zero"),
    ],
)
async def test_a_request_without_an_output_cap_is_refused_and_never_sent(
    tmp_path: Path, settings: dict[str, float] | None
) -> None:
    # The provider's own default cap is far above anything the reserve could cover.
    budget = write_budget(tmp_path)
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(UnboundedRequestError, match="output cap"):
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=settings)  # type: ignore[arg-type]
    assert sent == []
    assert budget.spent_usd() == 0.0
    assert meter.held_usd == 0.0
    assert meter.stop_reason == "budget"  # the budget cannot bound what it would pay for


async def test_a_request_longer_than_the_reserve_assumes_is_refused_by_its_bytes(
    tmp_path: Path,
) -> None:
    # 11,000 euro signs are 33,000 bytes of UTF-8 (three each) and no tokenizer makes more tokens
    # than bytes, though they are only 11,000 characters.
    budget = write_budget(tmp_path)
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(UnboundedRequestError, match="bytes"):
        await Agent(meter.wrap(model, "classifier")).run(
            "\N{EURO SIGN}" * 11_000, model_settings=CAP
        )
    assert sent == []
    assert budget.spent_usd() == 0.0
    assert meter.stop_reason == "budget"


async def test_a_request_within_the_bound_is_sent(tmp_path: Path) -> None:
    model, sent = _counting()
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    await Agent(meter.wrap(model, "classifier")).run("\N{EURO SIGN}" * 9_000, model_settings=CAP)
    assert sent == ["request"]  # 27,000 bytes, and the JSON around them, are under 32,000


async def test_the_tools_a_request_carries_count_toward_its_bytes(tmp_path: Path) -> None:
    def look_up(question: str) -> str:
        return question

    heavy = Tool(look_up, description="x" * 33_000)
    model, sent = _counting()
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    with pytest.raises(UnboundedRequestError, match="bytes"):
        await Agent(meter.wrap(model, "classifier"), tools=[heavy]).run("x", model_settings=CAP)
    assert sent == []


def test_a_meter_prices_tokens_not_characters(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    with pytest.raises(BudgetError, match="priced per characters_1k"):
        Meter(budget, route_id="vertex-ew1-gemini-embedding-001", run_id="run-1")


async def test_a_metered_model_neither_counts_tokens_nor_compacts_messages(tmp_path: Path) -> None:
    # Both go straight to the wrapped model, past `request`: unmetered if they were allowed.
    model, _ = _counting()
    metered = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1").wrap(model, "x")
    with pytest.raises(NotImplementedError, match="count tokens"):
        await metered.count_tokens([], None, ModelRequestParameters())
    context = ModelRequestContext(
        model=metered,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    with pytest.raises(NotImplementedError, match="compact messages"):
        await metered.compact_messages(context)
