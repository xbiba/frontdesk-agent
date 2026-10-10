"""Scripted attempts for the meter's tests: a model that ends each request as the test says, and a
sleep that only writes the wait down. A step is a function, so that a failure is a new exception
each time it is raised."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx2
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from frontdesk.model_gateway.costs import Budget

Step = Callable[[], BaseException | ModelResponse | Awaitable[ModelResponse]]


class Sleeps:
    """Stands for `asyncio.sleep`: it writes each wait down and does not wait."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class ScriptedModel(Model):
    """Ends its n-th request as the n-th step says: a step that gives an exception raises it, one
    that gives a response returns it as it is, one that gives an awaitable awaits it. (A
    FunctionModel puts an estimate where a response reports no usage; this does not.)"""

    def __init__(self, *script: Step) -> None:
        super().__init__()
        self._script = list(script)
        self.requests = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    @property
    def system(self) -> str:
        return "scripted"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        step = self._script[self.requests]  # one request more than scripted is an error in the test
        self.requests += 1
        outcome = step()
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, ModelResponse):
            return outcome
        return await outcome


def answer(input_tokens: int = 2_000, output_tokens: int = 300) -> Step:
    """A good answer that reports its usage."""

    def step() -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model_name="gemini-3.1-flash-lite",
            finish_reason="stop",
        )

    return step


def silent() -> ModelResponse:
    """A 200 answer that reports no usage at all."""
    return ModelResponse(parts=[TextPart("ok")], finish_reason="stop")


def read_timeout() -> BaseException:
    return httpx2.ReadTimeout("no answer after the request went out")


def write_timeout() -> BaseException:
    return httpx2.WriteTimeout("the request was not all written")


def connect_timeout() -> BaseException:
    return httpx2.ConnectTimeout("no connection in time")


def refused() -> BaseException:
    return httpx2.ConnectError("no route to host")


def reset() -> BaseException:
    return httpx2.ReadError("connection reset by peer")


def broken() -> BaseException:
    return httpx2.RemoteProtocolError("the server broke the protocol")


def status(code: int) -> Step:
    def step() -> BaseException:
        return ModelHTTPError(code, "gemini-3.1-flash-lite", body={"error": {"code": code}})

    return step


def ledger(budget: Budget) -> list[tuple[str, float]]:
    """The kind and the cost of each line of the ledger, in order."""
    if not budget.ledger_path.exists():
        return []
    lines = budget.ledger_path.read_text(encoding="utf-8").splitlines()
    return [(entry["kind"], entry["usd"]) for entry in map(json.loads, lines)]
