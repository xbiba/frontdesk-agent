"""The daily spend cap in production (Albi, 2026-10-13, before Instagram): past the cap of the
business's day no model is called, with an alert at 80% of it and one when it is reached."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from sqlalchemy import select

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import DailySpend
from frontdesk.model_gateway.daily_cap import CappedModel, DailyCap, SpendCapReachedError
from tests.model_gateway.budget_helpers import ROUTE, write_budget

DAY = datetime(2026, 10, 13, 10, tzinfo=UTC)


def _model() -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=100_000, output_tokens=10_000)  # 0.2 USD
        return ModelResponse(parts=[TextPart("ok")], usage=usage)

    return FunctionModel(respond)


@pytest.mark.db
async def test_past_the_daily_cap_no_model_is_called(
    app_url: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    now = [DAY]
    cap = DailyCap(
        make_session_factory(make_engine(app_url)),
        "demo-clinic",
        cap_usd=0.5,
        price=write_budget(tmp_path).price(ROUTE),
        zone=ZoneInfo("Europe/Tirane"),
        clock=lambda: now[0],
    )
    agent = Agent(CappedModel(_model(), cap))
    with caplog.at_level(logging.INFO):
        for _ in range(3):  # 0.2, 0.4 (80% of the cap), 0.6 (past it)
            await agent.run("hi")
        with pytest.raises(SpendCapReachedError):
            await agent.run("hi")
    events = [record.getMessage() for record in caplog.records]
    assert events.count("spend.alert") == 1
    assert events.count("spend.cap_reached") == 1
    now[0] = DAY + timedelta(days=1)  # the business's next day starts from nothing
    assert (await agent.run("hi")).output == "ok"


def _spent(cap: DailyCap) -> float:
    with business_session(cap.session_factory, cap.business_id) as session:
        return float(session.scalar(select(DailySpend.usd)) or 0)


@pytest.mark.db
async def test_a_request_without_an_answer_keeps_its_hold_and_an_error_status_gives_it_back(
    app_url: str, tmp_path: Path
) -> None:
    # The security review of 2026-10-13: requests that never settle must still count, and ones
    # in parallel must not all slip under the cap. Each holds its worst case before it is sent.
    cap = DailyCap(
        make_session_factory(make_engine(app_url)),
        "demo-clinic",
        cap_usd=10.0,
        price=write_budget(tmp_path).price(ROUTE),
        zone=ZoneInfo("Europe/Tirane"),
        clock=lambda: DAY,
    )

    async def refused(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(503, "m", body={})

    async def stuck(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        raise AssertionError("never answers")

    with pytest.raises(ModelHTTPError):
        await Agent(CappedModel(FunctionModel(refused), cap)).run("hi")
    assert _spent(cap) == 0.0  # an error status is not billed
    task = asyncio.create_task(Agent(CappedModel(FunctionModel(stuck), cap)).run("hi"))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    hold = 96_000 * 1.0 / 1e6 + 2_000 * 10.0 / 1e6  # the worst case of a request of the agent
    assert _spent(cap) == pytest.approx(hold)
