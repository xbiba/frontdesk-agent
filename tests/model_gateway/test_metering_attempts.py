"""The meter settles every attempt of a request on its own (M3 design §6, review of Task 6, I2).

The engine does the retrying, so that nothing hides below the meter: before each attempt it holds
the attempt's worst case, afterwards it records what that attempt cost, which is what the provider
reported, nothing for an answer of an error status or a connection that was never made, and the
whole reserve for anything that was sent without an answer. The table follows the review's.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest
from google.auth.exceptions import RefreshError
from pydantic_ai import Agent, ModelResponse
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from frontdesk.model_gateway.costs import Budget, BudgetExhaustedError
from frontdesk.model_gateway.factory import GOOGLE_RETRIES
from frontdesk.model_gateway.metering import (
    MAX_ATTEMPTS,
    MAX_INPUT_TOKENS_PER_REQUEST,
    RETRY_STATUSES,
    RETRY_WAITS,
    Meter,
)
from tests.model_gateway.attempt_helpers import (
    ScriptedModel,
    Sleeps,
    Step,
    answer,
    broken,
    connect_timeout,
    ledger,
    read_timeout,
    refused,
    reset,
    silent,
    status,
    write_timeout,
)
from tests.model_gateway.budget_helpers import ROUTE, write_budget

CAP = {"max_tokens": 600}
RESERVE = (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + 600 * 10.0) / 1_000_000  # 0.038 USD
USED = (2_000 * 1.0 + 300 * 10.0) / 1_000_000  # what `answer()` reports: 0.005 USD
CHAT, UNCONFIRMED = "chat", "chat-unconfirmed"
BACKOFF = (2.0, 4.0, 8.0)


@dataclass(frozen=True)
class Row:
    script: tuple[Step, ...]  # how each attempt ends, in order
    ledger: tuple[tuple[str, float], ...]  # the lines the attempts leave: kind and cost
    waits: tuple[float, ...] = ()
    raises: type[BaseException] | None = None  # what the last attempt ends with, if it failed
    stop: str | None = None


def unconfirmed(count: int) -> tuple[tuple[str, float], ...]:
    return ((UNCONFIRMED, RESERVE),) * count


TABLE: dict[str, Row] = {
    "answered": Row((answer(),), ((CHAT, USED),)),
    # Sent, and nothing came back: each attempt may have been billed in full.
    "four read timeouts": Row((read_timeout,) * 4, unconfirmed(4), BACKOFF, httpx2.ReadTimeout),
    "four write timeouts": Row((write_timeout,) * 4, unconfirmed(4), BACKOFF, httpx2.WriteTimeout),
    "two timeouts then an answer": Row(
        (read_timeout, read_timeout, answer()),
        (*unconfirmed(2), (CHAT, USED)),
        (2.0, 4.0),
    ),
    "a timeout then a good answer": Row(
        (read_timeout, answer()), (*unconfirmed(1), (CHAT, USED)), (2.0,)
    ),
    "a timeout then a 200 without usage": Row((read_timeout, silent), unconfirmed(2), (2.0,)),
    # The attempts that cost nothing do not hide the one that may have cost everything.
    "a timeout then 429 three times": Row(
        (read_timeout, status(429), status(429), status(429)),
        unconfirmed(1),
        BACKOFF,
        ModelHTTPError,
        "rate_limited",
    ),
    "a timeout then 503 three times": Row(
        (read_timeout, status(503), status(503), status(503)),
        unconfirmed(1),
        BACKOFF,
        ModelHTTPError,
    ),
    "a timeout then a refused connection three times": Row(
        (read_timeout, refused, refused, refused), unconfirmed(1), BACKOFF, httpx2.ConnectError
    ),
    # Refused by status, or never made: nothing was billed.
    "429 four times": Row((status(429),) * 4, (), BACKOFF, ModelHTTPError, "rate_limited"),
    "503 four times": Row((status(503),) * 4, (), BACKOFF, ModelHTTPError),
    "refused connection four times": Row((refused,) * 4, (), BACKOFF, httpx2.ConnectError),
    "connect timeout four times": Row((connect_timeout,) * 4, (), BACKOFF, httpx2.ConnectTimeout),
    "503 then a good answer": Row((status(503), answer()), ((CHAT, USED),), (2.0,)),
    "429 then a good answer does not stop the run": Row(
        (status(429), status(429), answer()), ((CHAT, USED),), (2.0, 4.0)
    ),
    "every status the SDK retried": Row(
        (status(500), status(502), status(504), answer()), ((CHAT, USED),), BACKOFF
    ),
    # Not retried: the SDK did not retry them either.
    "a 400": Row((status(400),), (), (), ModelHTTPError),
    "a 404": Row((status(404),), (), (), ModelHTTPError),
    "a 408": Row((status(408),), (), (), ModelHTTPError),
    "credentials that cannot be refreshed": Row(
        (lambda: RefreshError("expired"),), (), (), RefreshError
    ),
    "a connection reset": Row((reset,), unconfirmed(1), (), httpx2.ReadError),
    "a broken protocol": Row((broken,), unconfirmed(1), (), httpx2.RemoteProtocolError),
    "an answer the client cannot read": Row(
        (lambda: ModelAPIError("scripted", "Failed to decode response as JSON"),),
        unconfirmed(1),
        (),
        ModelAPIError,
    ),
    # A 200 that reports nothing may still have been billed for the whole answer.
    "a 200 without usage": Row((silent,), unconfirmed(1)),
}


@pytest.mark.parametrize("row", [pytest.param(row, id=name) for name, row in TABLE.items()])
async def test_each_attempt_is_settled_on_its_own(tmp_path: Path, row: Row) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    sleeps = Sleeps()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", sleep=sleeps)
    model = ScriptedModel(*row.script)
    agent = Agent(meter.wrap(model, "classifier"))
    if row.raises is None:
        await agent.run("x", model_settings=CAP)
    else:
        with pytest.raises(row.raises):
            await agent.run("x", model_settings=CAP)
    assert model.requests == len(row.script)  # no attempt more, none fewer
    assert ledger(budget) == [(kind, pytest.approx(usd)) for kind, usd in row.ledger]
    assert sleeps.waits == list(row.waits)
    assert meter.stop_reason == row.stop
    assert meter.held_usd == 0.0
    # Each attempt that cost something is a call of the turn, in order.
    assert [call.finish_reason for call in meter.drain()] == [
        "unconfirmed" if kind == UNCONFIRMED else "stop" for kind, _ in row.ledger
    ]


async def test_the_last_attempt_ends_the_request_with_what_it_ended_with(tmp_path: Path) -> None:
    # The runner reads the 429 from the exception that comes out, as it did before.
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    model = ScriptedModel(status(503), status(503), status(503), status(429))
    with pytest.raises(ModelHTTPError) as caught:
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert caught.value.status_code == 429
    assert meter.stop_reason == "rate_limited"


async def test_a_429_that_is_not_the_last_answer_does_not_stop_the_run(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    model = ScriptedModel(status(429), status(429), status(429), status(503))
    with pytest.raises(ModelHTTPError) as caught:
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert caught.value.status_code == 503
    assert meter.stop_reason is None


async def test_an_attempt_the_budget_cannot_cover_is_not_sent(tmp_path: Path) -> None:
    # Room for one reserve: the first attempt times out and is recorded at its reserve, so the
    # second no longer fits. The run stops, with the error of the refusal.
    budget = write_budget(tmp_path, cap_eur=RESERVE * 1.5, usd_per_eur=1.0)
    sleeps = Sleeps()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", sleep=sleeps)
    model = ScriptedModel(read_timeout, answer())
    with pytest.raises(BudgetExhaustedError):
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert model.requests == 1
    assert ledger(budget) == [(UNCONFIRMED, pytest.approx(RESERVE))]
    assert meter.stop_reason == "budget"
    assert budget.spent_usd() <= budget.cap_usd


async def test_every_attempt_holds_its_reserve_while_it_is_away(tmp_path: Path) -> None:
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    held: list[float] = []

    def attempt_that_looks() -> BaseException:
        held.append(meter.held_usd)
        return read_timeout()

    def last_attempt_that_looks() -> ModelResponse:
        held.append(meter.held_usd)
        return answer()()  # type: ignore[return-value]

    model = ScriptedModel(attempt_that_looks, last_attempt_that_looks)
    await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert held == [pytest.approx(RESERVE), pytest.approx(RESERVE)]  # one at a time, not summed
    assert meter.held_usd == 0.0


async def test_a_request_cancelled_during_a_later_attempt_costs_that_attempt_a_reserve(
    tmp_path: Path,
) -> None:
    # The turn's time limit and Ctrl-C reach the attempt that is away, whichever it is.
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", sleep=Sleeps())
    away = asyncio.Event()

    async def hangs() -> ModelResponse:
        away.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    model = ScriptedModel(read_timeout, hangs)
    task = asyncio.ensure_future(
        Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    )
    async with asyncio.timeout(10):  # a meter that does not retry never gets here
        await away.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ledger(budget) == [(UNCONFIRMED, pytest.approx(RESERVE))] * 2
    assert meter.held_usd == 0.0
    assert meter.stop_reason is None


async def test_a_request_cancelled_while_it_waits_between_attempts_costs_nothing_more(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    waiting = asyncio.Event()

    async def wait_for_ever(seconds: float) -> None:
        waiting.set()
        await asyncio.Event().wait()

    meter = Meter(budget, route_id=ROUTE, run_id="run-1", sleep=wait_for_ever)
    model = ScriptedModel(read_timeout, answer())
    task = asyncio.ensure_future(
        Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    )
    async with asyncio.timeout(10):  # a meter that does not retry never waits
        await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.requests == 1
    assert ledger(budget) == [(UNCONFIRMED, pytest.approx(RESERVE))]  # the first attempt only
    assert meter.held_usd == 0.0


def test_the_engine_retries_what_the_sdk_options_still_describe() -> None:
    # google-genai makes one attempt now; the options keep the policy the engine follows.
    assert GOOGLE_RETRIES.attempts == 1
    assert set(GOOGLE_RETRIES.http_status_codes or ()) == RETRY_STATUSES
    first, longest, base = (
        GOOGLE_RETRIES.initial_delay or 0.0,
        GOOGLE_RETRIES.max_delay or 0.0,
        GOOGLE_RETRIES.exp_base or 0.0,
    )
    waits = tuple(min(first * base**step, longest) for step in range(MAX_ATTEMPTS - 1))
    assert waits == RETRY_WAITS == BACKOFF
    assert MAX_ATTEMPTS == 4


async def test_a_failure_worth_retrying_ends_the_request_at_once_when_the_run_has_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first attempt timed out and its line cannot be written: the ledger has stopped the run,
    # so no second attempt can be sent. The request ends with the timeout, without the wait.
    def cannot_write(*args: object, **kwargs: object) -> None:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Budget, "record", cannot_write)
    sleeps = Sleeps()
    meter = Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", sleep=sleeps)
    model = ScriptedModel(read_timeout, answer())
    with pytest.raises(httpx2.ReadTimeout):
        await Agent(meter.wrap(model, "classifier")).run("x", model_settings=CAP)
    assert model.requests == 1
    assert sleeps.waits == []
    assert meter.stop_reason == "ledger"
