"""A stop says why (M3 Task 8, ruling 3, and the review of Task 6, N3).

A request the reserve cannot bound ends the run as a stop for the budget, and until now the cause
was only in the `turn.failed` log line. The meter keeps the cause of the first stop, so that the
command line can print it: the person who sees "stopped" learns whether the budget is spent, a
prompt is too long for the reserve, or a request was built without an output cap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError
from frontdesk.model_gateway.metering import (
    MAX_INPUT_TOKENS_PER_REQUEST,
    Meter,
    UnboundedRequestError,
)
from tests.model_gateway.attempt_helpers import ScriptedModel, Sleeps, status
from tests.model_gateway.budget_helpers import ROUTE, write_budget

CAP = {"max_tokens": 600}


def _ok(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart("ok")], usage=RequestUsage(input_tokens=100, output_tokens=10)
    )


def test_a_meter_that_is_running_has_no_reason_to_give(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    assert (meter.stop_reason, meter.stop_detail) == (None, None)


async def test_a_request_without_an_output_cap_says_so(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    with pytest.raises(UnboundedRequestError) as caught:
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run("x")  # no max_tokens
    assert meter.stop_reason == "budget"
    assert meter.stop_detail == str(caught.value)
    assert meter.stop_detail == (
        "the request has no output cap: its cost cannot be bounded, so it is not sent"
    )


async def test_a_request_longer_than_the_reserve_assumes_says_how_long_it_was(
    tmp_path: Path,
) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    long_prompt = "x" * (MAX_INPUT_TOKENS_PER_REQUEST + 1)
    with pytest.raises(UnboundedRequestError):
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run(
            long_prompt, model_settings=CAP
        )
    assert meter.stop_reason == "budget"
    detail = meter.stop_detail or ""
    assert detail.startswith("the request is ")
    assert f"more than the {MAX_INPUT_TOKENS_PER_REQUEST} tokens the reserve assumes" in detail
    assert detail.endswith("its cost cannot be bounded, so it is not sent")


async def test_a_budget_with_no_room_says_what_is_left_and_what_the_call_may_cost(
    tmp_path: Path,
) -> None:
    # A call reserves 0.038 USD (32,000 tokens in, a cap of 600 out); the budget has 0.02.
    meter = Meter(write_budget(tmp_path, cap_eur=0.02, usd_per_eur=1.0), route_id=ROUTE, run_id="r")
    with pytest.raises(BudgetExhaustedError):
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run("x", model_settings=CAP)
    assert meter.stop_reason == "budget"
    assert meter.stop_detail == (
        "the budget of 0.02 EUR has 0.0200 EUR left, and the next call may cost 0.0380 EUR"
    )


async def test_a_ledger_that_cannot_be_read_names_the_file_and_the_line(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text('{"usd": 1}\nnot json\n', encoding="utf-8")
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(BudgetError):
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run("x", model_settings=CAP)
    assert meter.stop_reason == "ledger"
    assert meter.stop_detail == f"{budget.ledger_path}: line 2 is not a spend entry"


async def test_a_ledger_that_is_a_directory_names_the_file_and_the_reason(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(OSError, match="Is a directory"):
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run("x", model_settings=CAP)
    assert meter.stop_reason == "ledger"
    assert meter.stop_detail == f"{budget.ledger_path}: cannot be read (Is a directory)"


async def test_a_ledger_that_cannot_be_appended_to_names_the_file_and_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_write(*args: object, **kwargs: object) -> None:
        raise OSError(30, "Read-only file system")

    budget = write_budget(tmp_path)
    monkeypatch.setattr(Budget, "record", cannot_write)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1")
    with pytest.raises(OSError, match="Read-only"):
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run("x", model_settings=CAP)
    assert meter.stop_reason == "ledger"
    assert (
        meter.stop_detail == f"{budget.ledger_path}: cannot be appended to (Read-only file system)"
    )


async def test_a_rate_limit_that_outlived_the_retries_says_so(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    with pytest.raises(ModelHTTPError):
        await Agent(meter.wrap(ScriptedModel(*[status(429)] * 4), "classifier")).run(
            "x", model_settings=CAP
        )
    assert meter.stop_reason == "rate_limited"
    assert (
        meter.stop_detail == "Google answered 429 (rate limited) at the last attempt of a request"
    )


async def test_the_reason_and_its_detail_are_those_of_the_first_stop(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1")
    meter.rate_limited()
    first = meter.stop_detail
    with pytest.raises(UnboundedRequestError):
        await Agent(meter.wrap(FunctionModel(_ok), "classifier")).run("x")  # no output cap
    assert meter.stop_reason == "rate_limited"
    assert meter.stop_detail == first
