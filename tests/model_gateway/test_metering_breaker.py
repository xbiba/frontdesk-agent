"""A run that keeps failing stops, whatever the failures are (M3 Task 8, ruling 1).

Without a breaker an outage would use up the cap on reservations: every attempt that was sent and
brought nothing back is recorded at its reserve, and the run would go on to the next case. The
breaker counts consecutive failed attempts of any kind (an error status, a request that never left,
a timeout, a reset, a cancellation, an answer without usage), and a successful attempt starts the
count again. It is judged when a request ends, so the retries of design §6 are never cut short: a
request that fails three times and then answers is a recovered request, not an outage.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NamedTuple

import httpx2
import pytest
from google.auth.exceptions import RefreshError
from pydantic_ai import Agent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError

from frontdesk.model_gateway.costs import Budget, BudgetExhaustedError
from frontdesk.model_gateway.metering import BREAKER_ATTEMPTS, Meter, RunStoppedError
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
)
from tests.model_gateway.budget_helpers import ROUTE, write_budget

CAP = {"max_tokens": 600}
UNAVAILABLE = "unavailable"


class Rig(NamedTuple):
    meter: Meter
    sleeps: Sleeps
    budget: Budget


def _rig(tmp_path: Path, *, breaker: int | None = BREAKER_ATTEMPTS) -> Rig:
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    sleeps = Sleeps()
    meter = Meter(budget, route_id=ROUTE, run_id="run-1", breaker=breaker, sleep=sleeps)
    return Rig(meter, sleeps, budget)


async def _request(agent: Agent[None, str]) -> None:
    await agent.run("x", model_settings=CAP)


async def _fails(agent: Agent[None, str]) -> None:
    with pytest.raises((httpx2.HTTPError, ModelHTTPError, RefreshError)):
        await _request(agent)


def test_the_breaker_is_three_attempts() -> None:
    assert BREAKER_ATTEMPTS == 3


def test_a_breaker_needs_at_least_one_attempt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        Meter(write_budget(tmp_path), route_id=ROUTE, run_id="run-1", breaker=0)


async def test_without_a_breaker_a_run_that_keeps_failing_goes_on(tmp_path: Path) -> None:
    # The chat and every earlier caller: nothing changes unless a run asks for the breaker.
    rig = _rig(tmp_path, breaker=None)
    agent = Agent(rig.meter.wrap(ScriptedModel(*[status(504)] * 8), "classifier"))
    await _fails(agent)
    await _fails(agent)
    assert rig.meter.stop_reason is None
    assert rig.meter.consecutive_failures == 8


# --- Failures of any kind ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "billed", "reason"),
    [
        pytest.param(status(504), 0, UNAVAILABLE, id="a-504-settles-at-zero"),
        pytest.param(status(503), 0, UNAVAILABLE, id="a-503"),
        pytest.param(read_timeout, 4, UNAVAILABLE, id="a-timeout-is-recorded-at-its-reserve"),
        pytest.param(refused, 0, UNAVAILABLE, id="a-refused-connection"),
        pytest.param(connect_timeout, 0, UNAVAILABLE, id="a-connect-timeout"),
        pytest.param(status(429), 0, "rate_limited", id="a-429-keeps-its-own-name"),
    ],
)
async def test_a_request_that_keeps_failing_ends_the_run_after_all_its_retries(
    tmp_path: Path, failure: Step, billed: int, reason: str
) -> None:
    rig = _rig(tmp_path)
    model = ScriptedModel(*[failure] * 4)
    await _fails(Agent(rig.meter.wrap(model, "classifier")))
    assert model.requests == 4  # the retries of design §6 are not cut short
    assert rig.sleeps.waits == [2.0, 4.0, 8.0]
    assert len(ledger(rig.budget)) == billed
    assert rig.meter.stop_reason == reason
    assert rig.meter.consecutive_failures == 4
    assert rig.meter.held_usd == 0.0


@pytest.mark.parametrize(
    "failures",
    [
        pytest.param((status(400), reset, lambda: RefreshError("expired")), id="status-reset-auth"),
        pytest.param((reset, reset, reset), id="three-resets"),
        pytest.param((broken, status(404), reset), id="protocol-status-reset"),
        pytest.param((status(403),) * 3, id="three-refusals"),
    ],
)
async def test_failures_of_different_kinds_in_different_requests_add_up(
    tmp_path: Path, failures: tuple[Step, ...]
) -> None:
    rig = _rig(tmp_path)
    model = ScriptedModel(*failures)
    agent = Agent(rig.meter.wrap(model, "classifier"))
    for number in range(1, 4):
        assert rig.meter.stop_reason is None, f"stopped before request {number}"
        await _fails(agent)
    assert rig.meter.stop_reason == UNAVAILABLE
    assert model.requests == 3


async def test_an_answer_without_usage_counts_as_a_failed_attempt(tmp_path: Path) -> None:
    # It may have been billed in full, and it says nothing about what it cost: three in a row.
    rig = _rig(tmp_path)
    agent = Agent(rig.meter.wrap(ScriptedModel(silent, silent, silent), "classifier"))
    await _request(agent)
    await _request(agent)
    assert rig.meter.stop_reason is None
    await _request(agent)  # this one is answered too, and it is the third
    assert rig.meter.stop_reason == UNAVAILABLE
    assert rig.meter.consecutive_failures == 3
    assert [kind for kind, _ in ledger(rig.budget)] == ["chat-unconfirmed"] * 3


async def test_a_request_cut_off_by_the_turn_limit_counts_too(tmp_path: Path) -> None:
    # A hung provider: two attempts time out, the third is still away when the turn's time limit
    # cancels it. The cancellation stays a cancellation, and it is the third failure.
    rig = _rig(tmp_path)
    away = asyncio.Event()

    async def hangs() -> ModelResponse:
        away.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    model = ScriptedModel(read_timeout, read_timeout, hangs)
    task = asyncio.ensure_future(_request(Agent(rig.meter.wrap(model, "classifier"))))
    async with asyncio.timeout(10):
        await away.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.meter.stop_reason == UNAVAILABLE
    assert [kind for kind, _ in ledger(rig.budget)] == ["chat-unconfirmed"] * 3
    assert rig.meter.held_usd == 0.0


# --- A success starts the count again ------------------------------------------------------------


async def test_a_request_that_fails_three_times_and_then_answers_is_not_an_outage(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    model = ScriptedModel(status(503), status(503), status(503), answer())
    await _request(Agent(rig.meter.wrap(model, "classifier")))
    assert model.requests == 4
    assert rig.sleeps.waits == [2.0, 4.0, 8.0]
    assert rig.meter.stop_reason is None
    assert rig.meter.consecutive_failures == 0


async def test_an_answer_between_the_failures_starts_the_count_again(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    model = ScriptedModel(
        status(400), status(400), answer(), status(400), status(400), answer(), status(400)
    )
    agent = Agent(rig.meter.wrap(model, "classifier"))
    for step in range(7):
        if step in {2, 5}:
            await _request(agent)
        else:
            await _fails(agent)
        assert rig.meter.stop_reason is None
    assert rig.meter.consecutive_failures == 1  # the last one only


async def test_the_count_runs_across_roles_and_requests(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    classifier = Agent(rig.meter.wrap(ScriptedModel(status(400), status(400)), "classifier"))
    responder = Agent(rig.meter.wrap(ScriptedModel(status(400)), "responder"))
    await _fails(classifier)
    await _fails(classifier)
    assert rig.meter.stop_reason is None
    await _fails(responder)  # a different agent on the same meter: the third in a row
    assert rig.meter.stop_reason == UNAVAILABLE


# --- What a stop is ------------------------------------------------------------------------------


async def test_once_it_has_tripped_no_further_request_is_sent(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    model = ScriptedModel(status(400), status(400), status(400), answer())
    agent = Agent(rig.meter.wrap(model, "classifier"))
    for _ in range(3):
        await _fails(agent)
    with pytest.raises(RunStoppedError, match=UNAVAILABLE):
        await _request(agent)
    assert model.requests == 3
    assert rig.meter.stop_reason == UNAVAILABLE


async def test_the_first_cause_of_a_stop_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two failed requests, then a third whose line cannot be written: it is the third failed
    # attempt, so the breaker's condition holds, and the ledger stopped the run first.
    rig = _rig(tmp_path)
    agent = Agent(rig.meter.wrap(ScriptedModel(status(400), status(400), reset), "classifier"))
    await _fails(agent)
    await _fails(agent)

    def cannot_write(*args: object, **kwargs: object) -> None:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Budget, "record", cannot_write)
    with pytest.raises(httpx2.ReadError):
        await _request(agent)
    assert rig.meter.consecutive_failures == 3
    assert rig.meter.stop_reason == "ledger"


async def test_a_request_the_budget_refused_was_never_an_attempt(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.budget.record(run_id="x", route_id=ROUTE, kind="chat", usd=10_000.0)  # far past the cap
    model = ScriptedModel(answer())
    agent = Agent(rig.meter.wrap(model, "classifier"))
    with pytest.raises(BudgetExhaustedError):
        await _request(agent)
    for _ in range(2):
        with pytest.raises(RunStoppedError, match="budget"):
            await _request(agent)
    assert model.requests == 0
    assert rig.meter.consecutive_failures == 0
    assert rig.meter.stop_reason == "budget"


@pytest.mark.parametrize(
    ("failure", "attempts", "how"),
    [
        pytest.param(status(504), 4, "ended with HTTP 504", id="a-status"),
        pytest.param(read_timeout, 4, "ended with ReadTimeout", id="a-timeout"),
        pytest.param(reset, 1, "ended with ReadError", id="a-reset"),
    ],
)
async def test_the_stop_says_how_many_attempts_failed_and_how_the_last_request_ended(
    tmp_path: Path, failure: Step, attempts: int, how: str
) -> None:
    rig = _rig(tmp_path, breaker=1 if attempts == 1 else 3)
    await _fails(Agent(rig.meter.wrap(ScriptedModel(*[failure] * attempts), "classifier")))
    assert rig.meter.stop_reason == UNAVAILABLE
    plural = "s" if attempts > 1 else ""
    assert rig.meter.stop_detail == (
        f"{attempts} attempt{plural} in a row failed; the last request {how}"
    )


async def test_the_stop_never_repeats_what_the_provider_wrote(tmp_path: Path) -> None:
    def leaky() -> BaseException:
        return ModelHTTPError(400, "gemini", body={"error": {"message": "the patient said X"}})

    rig = _rig(tmp_path, breaker=1)
    await _fails(Agent(rig.meter.wrap(ScriptedModel(leaky), "classifier")))
    assert (
        rig.meter.stop_detail == "1 attempt in a row failed; the last request ended with HTTP 400"
    )


async def test_a_usage_less_answer_is_described_as_such(tmp_path: Path) -> None:
    rig = _rig(tmp_path, breaker=1)
    await _request(Agent(rig.meter.wrap(ScriptedModel(silent), "classifier")))
    assert rig.meter.stop_detail == (
        "1 attempt in a row failed; the last request was answered without usage"
    )
