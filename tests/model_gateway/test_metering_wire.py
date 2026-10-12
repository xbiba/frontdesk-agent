"""The meter on the real Gemini stack: pydantic-ai, google-genai and the client `build_live_model`
builds, answered by a transport inside this process. Nothing leaves it: the fake credentials need
no lookup, and every socket is refused while these tests run.

They show what the stack raises for an attempt that failed, and so what the meter charges for it
(M3 design §6): a library upgrade that changed an exception would change what the cap counts. The
SDK is built as production builds it, with `GOOGLE_RETRIES` (one attempt), so every request the
transport sees is an attempt the meter settled; the waits between attempts are not slept.
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
from google.oauth2.credentials import Credentials
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, ModelHTTPError

from frontdesk.model_gateway.costs import Budget
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.metering import MAX_INPUT_TOKENS_PER_REQUEST, Meter
from frontdesk.model_gateway.routes import DataClass, require_route
from tests.model_gateway.attempt_helpers import Sleeps, ledger
from tests.model_gateway.budget_helpers import ROUTE, write_budget

PROJECT = "clinica-mcu"
BACKOFF = [2.0, 4.0, 8.0]
Step = Callable[[], Exception | httpx2.Response]
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
    sleeps: Sleeps
    cap: int


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., Wire]:
    """Build the route's classifier model on a transport that answers with `handler`."""
    # Only for these tests: tests forbid model requests (conftest), and only the transport answers.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)

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
        sleeps = Sleeps()
        meter = Meter(budget, route_id=ROUTE, run_id="run-1", sleep=sleeps)
        cap = route.output_limits.classifier
        settings = live_model_settings(route, max_tokens=cap)
        agent = Agent(meter.wrap(model, "classifier"), model_settings=settings)
        return Wire(agent, meter, budget, sent, sleeps, cap)

    return build


def _reserve(cap: int) -> float:
    return (MAX_INPUT_TOKENS_PER_REQUEST * 1.0 + cap * 10.0) / 1_000_000


def _in_order(*steps: Step) -> Handler:
    """Answers the n-th request as the n-th step says; a step gives an exception to raise or a
    response to return."""
    remaining = list(steps)

    def handler(request: httpx2.Request) -> httpx2.Response:
        outcome = remaining.pop(0)()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handler


def _status(code: int) -> Step:
    body = {"error": {"code": code, "message": "refused by the wire test", "status": "REFUSED"}}
    return lambda: httpx2.Response(code, json=body)


def _answer(*, usage: dict[str, int] | None) -> Step:
    body: dict[str, object] = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ],
        "modelVersion": "gemini-3.1-flash-lite-wire",
    }
    if usage is not None:
        body["usageMetadata"] = usage
    return lambda: httpx2.Response(200, json=body)


def _timeout() -> Exception:
    return httpx2.ReadTimeout("no answer after the request went out")


def _refused() -> Exception:
    return httpx2.ConnectError("no route to host")


def _reset() -> Exception:
    return httpx2.RemoteProtocolError("the server broke the protocol")


def _not_json() -> httpx2.Response:
    return httpx2.Response(200, content=b"<html>not json</html>")


def _blocked() -> httpx2.Response:
    # The provider read the prompt and refused it: it reports the prompt tokens it counted.
    body = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "usageMetadata": {"promptTokenCount": 40, "totalTokenCount": 40},
    }
    return httpx2.Response(200, json=body)


USAGE = {
    "promptTokenCount": 1_000,
    "candidatesTokenCount": 100,
    "thoughtsTokenCount": 40,
    "totalTokenCount": 1_140,
}
USED = (1_000 * 1.0 + 140 * 10.0) / 1_000_000  # thinking is billed as output


async def test_the_usage_the_provider_reports_is_what_the_meter_records(
    wire: Callable[..., Wire],
) -> None:
    run = wire(_in_order(_answer(usage=USAGE)))
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
    assert ledger(run.budget) == [("chat", pytest.approx(USED))]
    assert len(run.sent) == 1


@dataclass(frozen=True)
class Row:
    steps: tuple[Step, ...]  # how the transport answers each attempt
    ledger: tuple[tuple[str, float], ...]
    waits: list[float]
    raises: type[Exception] | None = None
    stop: str | None = None


def _unconfirmed(count: int, cap: int = 600) -> tuple[tuple[str, float], ...]:
    return (("chat-unconfirmed", _reserve(cap)),) * count


ROWS = {
    # The stack raises these as they are, and the engine retries them (four attempts, one request
    # each at the transport): the SDK underneath makes no attempt of its own.
    "refused connection": Row((_refused,) * 4, (), BACKOFF, httpx2.ConnectError),
    "read timeout": Row((_timeout,) * 4, _unconfirmed(4), BACKOFF, httpx2.ReadTimeout),
    "429": Row((_status(429),) * 4, (), BACKOFF, ModelHTTPError, "rate_limited"),
    "503": Row((_status(503),) * 4, (), BACKOFF, ModelHTTPError),
    # Not retried.
    "400": Row((_status(400),), (), [], ModelHTTPError),
    "connection reset": Row((_reset,), _unconfirmed(1), [], httpx2.RemoteProtocolError),
    "200 that is not JSON": Row((_not_json,), _unconfirmed(1), [], ModelAPIError),
    # Answered with usage, though the agent then refuses the empty answer: the 40 tokens are billed.
    "prompt blocked by the provider": Row(
        (_blocked,), (("chat", 40 * 1.0 / 1_000_000),), [], ContentFilterError
    ),
    # The attempts that cost nothing do not hide the one that may have cost everything.
    "timeout then 429": Row(
        (_timeout, _status(429), _status(429), _status(429)),
        _unconfirmed(1),
        BACKOFF,
        ModelHTTPError,
        "rate_limited",
    ),
    "timeout then refused connection": Row(
        (_timeout, _refused, _refused, _refused), _unconfirmed(1), BACKOFF, httpx2.ConnectError
    ),
    "two timeouts then an answer": Row(
        (_timeout, _timeout, _answer(usage=USAGE)),
        (*_unconfirmed(2), ("chat", USED)),
        [2.0, 4.0],
    ),
    "503 then an answer": Row((_status(503), _answer(usage=USAGE)), (("chat", USED),), [2.0]),
    # An answer that reports no usage may have been billed in full.
    "200 without usage": Row((_answer(usage=None),), _unconfirmed(1), []),
    "200 with all-zero usage": Row(
        (_answer(usage={"promptTokenCount": 0, "candidatesTokenCount": 0}),), _unconfirmed(1), []
    ),
}


@pytest.mark.parametrize("row", [pytest.param(row, id=name) for name, row in ROWS.items()])
async def test_each_attempt_on_the_real_stack_is_settled_on_its_own(
    wire: Callable[..., Wire], row: Row
) -> None:
    run = wire(_in_order(*row.steps))
    if row.raises is None:
        await run.agent.run("x")
    else:
        with pytest.raises(row.raises):
            await run.agent.run("x")
    assert len(run.sent) == len(row.steps)  # one request at the transport for each attempt
    assert ledger(run.budget) == [(kind, pytest.approx(usd)) for kind, usd in row.ledger]
    assert run.sleeps.waits == row.waits
    assert run.meter.stop_reason == row.stop
    assert run.meter.held_usd == 0.0


async def test_credentials_that_cannot_be_refreshed_cost_nothing_and_send_nothing(
    wire: Callable[..., Wire],
) -> None:
    def never(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request may be sent without credentials")

    run = wire(never, credentials=_ExpiredCredentials())
    with pytest.raises(RefreshError):
        await run.agent.run("x")
    assert run.sent == []
    assert ledger(run.budget) == []
    assert run.sleeps.waits == []  # credentials are not retried
    assert [(call.finish_reason, call.usd) for call in run.meter.drain()] == [("not sent", 0.0)]
    assert run.meter.stop_reason is None
