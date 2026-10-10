"""The meter on the real Gemini stack: pydantic-ai, google-genai and the client `build_live_model`
builds, answered by a transport inside this process. Nothing leaves it: the fake credentials need
no lookup, and every socket is refused while these tests run.

They show what the stack raises for a request that failed, and so what the meter charges for it
(M3 design §6): a library upgrade that changed an exception would change what the cap counts.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest
from google.auth.credentials import Credentials as AuthCredentials
from google.auth.exceptions import RefreshError
from google.genai.types import HttpRetryOptions
from google.oauth2.credentials import Credentials
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import ModelHTTPError

from frontdesk.model_gateway import factory
from frontdesk.model_gateway.costs import Budget
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.metering import MAX_INPUT_TOKENS_PER_REQUEST, Meter
from frontdesk.model_gateway.routes import DataClass, require_route
from tests.model_gateway.budget_helpers import ROUTE, write_budget

PROJECT = "clinica-mcu"
ATTEMPTS = 2  # the client's retries, shortened so that the tests do not wait
Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a wire test opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


class _ExpiredCredentials(AuthCredentials):
    """Credentials whose refresh fails, as Application Default Credentials do once they expire."""

    def refresh(self, request: object) -> None:
        raise RefreshError("the refresh token expired")


@dataclass
class Wire:
    agent: Agent[None, str]
    meter: Meter
    budget: Budget
    sent: list[httpx2.Request]
    cap: int


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., Wire]:
    """Build the route's classifier model on a transport that answers with `handler`."""
    # Only for these tests: tests forbid model requests (conftest), and only the transport answers.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    quick = HttpRetryOptions(
        attempts=ATTEMPTS,
        initial_delay=0.001,  # not zero: the client reads a zero as "the default"
        max_delay=0.001,
        exp_base=1.0,
        jitter=0.001,
        http_status_codes=[429, 500, 502, 503, 504],
    )
    monkeypatch.setattr(factory, "GOOGLE_RETRIES", quick)

    def build(handler: Handler, *, credentials: AuthCredentials | None = None) -> Wire:
        sent: list[httpx2.Request] = []

        def record_then_answer(request: httpx2.Request) -> httpx2.Response:
            sent.append(request)
            return handler(request)

        route = require_route(ROUTE, DataClass.EVALUATION)
        model = build_live_model(route, gcp_project=PROJECT)
        api = model.client._api_client
        api._async_httpx_client._transport = httpx2.MockTransport(record_then_answer)
        api._credentials = credentials or Credentials(token="fake-token")  # valid: no refresh
        budget = write_budget(tmp_path, usd_per_eur=1.0)
        meter = Meter(budget, route_id=ROUTE, run_id="run-1")
        cap = route.output_limits.classifier
        settings = live_model_settings(route, max_tokens=cap)
        agent = Agent(meter.wrap(model, "classifier"), model_settings=settings)
        return Wire(agent, meter, budget, sent, cap)

    return build


def _reserve(cap: int) -> float:
    return (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + cap * 10.0) / 1_000_000


def _fails_with(error: Exception) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error

    return handler


def _status(code: int) -> Handler:
    body = {"error": {"code": code, "message": "refused by the wire test", "status": "REFUSED"}}
    return lambda request: httpx2.Response(code, json=body)


async def test_the_usage_the_provider_reports_is_what_the_meter_records(
    wire: Callable[..., Wire],
) -> None:
    usage = {
        "promptTokenCount": 1_000,
        "candidatesTokenCount": 100,
        "thoughtsTokenCount": 40,
        "totalTokenCount": 1_140,
    }
    answer = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": usage,
        "modelVersion": "gemini-3.1-flash-lite-wire",
    }
    run = wire(lambda request: httpx2.Response(200, json=answer))
    await run.agent.run("x")
    [call] = run.meter.drain()
    assert (call.agent, call.model_version, call.finish_reason) == (
        "classifier",
        "gemini-3.1-flash-lite-wire",
        "stop",
    )
    # The provider counts thinking apart from the answer (100 and 40); it is billed as output,
    # so the stack adds it to the output tokens.
    assert (call.input_tokens, call.output_tokens, call.thoughts_tokens) == (1_000, 140, 40)
    assert run.budget.spent_usd() == pytest.approx((1_000 * 1.0 + 140 * 10.0) / 1_000_000)
    assert len(run.sent) == 1


@pytest.mark.parametrize(
    ("handler", "raises", "attempts", "charged", "stop"),
    [
        pytest.param(
            _fails_with(httpx2.ConnectError("no route to host")),
            httpx2.ConnectError,
            ATTEMPTS,
            False,
            None,
            id="no-connection",
        ),
        pytest.param(
            _fails_with(httpx2.ReadTimeout("no answer after the request went out")),
            httpx2.ReadTimeout,
            ATTEMPTS,
            True,
            None,
            id="read-timeout",
        ),
        pytest.param(
            _status(429), ModelHTTPError, ATTEMPTS, False, "rate_limited", id="rate-limited"
        ),
        pytest.param(_status(503), ModelHTTPError, ATTEMPTS, False, None, id="unavailable"),
        pytest.param(_status(400), ModelHTTPError, 1, False, None, id="refused-request"),
    ],
)
async def test_a_failed_request_is_charged_only_when_it_may_have_been_processed(
    wire: Callable[..., Wire],
    handler: Handler,
    raises: type[Exception],
    attempts: int,
    charged: bool,
    stop: str | None,
) -> None:
    run = wire(handler)
    with pytest.raises(raises):
        await run.agent.run("x")
    # The client's own retries happen below the meter: to it they are one request.
    assert len(run.sent) == attempts
    assert run.budget.spent_usd() == pytest.approx(_reserve(run.cap) if charged else 0.0)
    assert [call.finish_reason for call in run.meter.drain()] == (
        ["unconfirmed"] if charged else []
    )
    assert run.meter.stop_reason == stop


async def test_credentials_that_cannot_be_refreshed_cost_nothing_and_send_nothing(
    wire: Callable[..., Wire],
) -> None:
    never = _fails_with(AssertionError("no request may be sent without credentials"))
    run = wire(never, credentials=_ExpiredCredentials())
    with pytest.raises(RefreshError):
        await run.agent.run("x")
    assert run.sent == []
    assert run.budget.spent_usd() == 0.0
    assert run.meter.drain() == ()
    assert run.meter.stop_reason is None
