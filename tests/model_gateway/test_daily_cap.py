"""The daily spend cap in production (Albi, 2026-10-13, before Instagram): past the cap of the
business's day no model is called, with an alert at 80% of it and one when it is reached."""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from frontdesk.db.engine import make_engine, make_session_factory
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
