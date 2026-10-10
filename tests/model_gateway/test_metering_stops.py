"""A run stops, and stays stopped, when its ledger cannot be trusted or its budget has no room
(M3 design §6): a request that cannot be counted is a request the cap no longer covers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError
from frontdesk.model_gateway.metering import Meter, RunStoppedError
from tests.model_gateway.budget_helpers import ROUTE, write_budget

CAP = {"max_tokens": 600}


def _counting() -> tuple[FunctionModel, list[str]]:
    """A model that answers "ok" and lists each request it receives."""
    sent: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append("request")
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(model), sent


def _cut_off(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise TimeoutError("no answer after the request went out")


def _cannot_write(*args: object, **kwargs: object) -> None:
    raise OSError(30, "Read-only file system")


async def test_a_damaged_ledger_stops_the_run_before_the_request_is_sent(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text('{"usd": 1}\nnot json\n', encoding="utf-8")
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(BudgetError, match="line 2"):
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert sent == []
    assert meter.stop_reason == "ledger"


async def test_a_ledger_that_is_not_text_stops_the_run(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_bytes(b'{"usd": 1}\n\xff\xfe\n')
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(BudgetError, match="not UTF-8 text"):
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert sent == []
    assert meter.stop_reason == "ledger"


async def test_a_ledger_that_cannot_be_read_stops_the_run(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()  # a directory where the ledger should be: no one can read it
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(OSError, match="Is a directory"):
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert sent == []
    assert meter.stop_reason == "ledger"


async def test_a_ledger_that_cannot_be_written_stops_the_run_after_the_paid_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = write_budget(tmp_path)
    monkeypatch.setattr(Budget, "record", _cannot_write)
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    agent = Agent(meter.wrap(model, "classifier"))
    with pytest.raises(OSError, match="Read-only"):
        await agent.run("x", model_settings=CAP)
    assert sent == ["request"]  # the provider was asked, and its bill could not be written down
    assert meter.stop_reason == "ledger"
    with pytest.raises(RunStoppedError, match="ledger"):
        await agent.run("x", model_settings=CAP)
    assert sent == ["request"]  # one uncounted request at most: nothing else is sent


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param(TimeoutError("no answer after the request went out"), id="timeout"),
        pytest.param(asyncio.CancelledError(), id="cancelled"),
    ],
)
async def test_a_cut_off_request_whose_line_cannot_be_written_still_ends_as_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: BaseException
) -> None:
    # A cancellation must stay a cancellation: the turn's time limit and Ctrl-C depend on it.
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ending

    budget = write_budget(tmp_path)
    monkeypatch.setattr(Budget, "record", _cannot_write)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(type(ending)):
        await Agent(meter.wrap(FunctionModel(model), "classifier")).run("x", model_settings=CAP)
    assert meter.stop_reason == "ledger"


@pytest.mark.parametrize("cause", ["budget", "rate_limited", "ledger"])
async def test_once_the_run_has_stopped_no_further_request_is_sent(
    tmp_path: Path, cause: str
) -> None:
    budget = write_budget(tmp_path, cap_eur=0.001 if cause == "budget" else 10.0)
    model, sent = _counting()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    agent = Agent(meter.wrap(model, "classifier"))
    if cause == "budget":
        with pytest.raises(BudgetExhaustedError):
            await agent.run("x", model_settings=CAP)
    elif cause == "rate_limited":
        meter.rate_limited()
    else:
        budget.ledger_path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(BudgetError):
            await agent.run("x", model_settings=CAP)
    assert meter.stop_reason == cause
    budget.ledger_path.unlink(missing_ok=True)  # even a repaired ledger does not reopen the run
    with pytest.raises(RunStoppedError, match=cause):
        await agent.run("x", model_settings=CAP)
    assert sent == []
    assert meter.stop_reason == cause


async def test_the_first_cause_of_a_stop_is_the_one_kept(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    meter.rate_limited()
    with pytest.raises(RunStoppedError):
        meter.reserve(600)
    assert meter.stop_reason == "rate_limited"


async def test_a_request_already_in_flight_is_still_settled_after_a_stop(tmp_path: Path) -> None:
    # A stop refuses the next request; it does not make the one in flight vanish from the ledger.
    budget = write_budget(tmp_path)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        meter.rate_limited()  # another request met a rate limit while this one was away
        return ModelResponse(parts=[TextPart("ok")])

    await Agent(meter.wrap(FunctionModel(model), "classifier")).run("x", model_settings=CAP)
    assert budget.spent_usd() > 0
    assert [call.agent for call in meter.drain()] == ["classifier"]
