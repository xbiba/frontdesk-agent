"""Metering of live model requests for the M3 live evaluation (design §4.4, §6).

`Meter.wrap` puts a `MeteredModel` around the model an agent runs on. Before each request it
reserves the most that request could cost and refuses it when the budget has no room for that;
after it, it records what the provider says the request used. A request that was sent but brought
back no usage (a read timeout, a turn cancelled by its time limit or Ctrl-C, an answer the client
could not read) may still have been billed: it is recorded at its reserve, so the ledger never
counts less than was spent. A 429 that outlived the client's retries is noted too, so the run can
stop and resume later. The runner reads the calls of each turn with `drain`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Final

import httpx2
from google.auth.exceptions import GoogleAuthError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError

# The reserve assumes this many input tokens: the prompts (under 3,000 characters), the evidence
# (at most 4,000 characters) and the schemas stay far below it, so the reserve is an upper bound.
MAX_INPUT_TOKENS_PER_REQUEST: Final = 32_000
DEFAULT_MAX_TOKENS: Final = 2_000  # a request without a cap is reserved as if it had this one
# The errors that end a request before it reaches the provider, which bills nothing for it: no
# credentials, no connection. An error status comes back as ModelHTTPError, also unbilled.
NOT_SENT: Final = (GoogleAuthError, httpx2.ConnectError, httpx2.ConnectTimeout)


class RunStoppedError(RuntimeError):
    """A request was refused because the run has already stopped (see `Meter.stop_reason`)."""


@dataclass(frozen=True)
class CallRecord:
    """One model request: who made it, what it used and cost, how long it took."""

    agent: str
    model_version: str | None
    input_tokens: int
    output_tokens: int  # thinking included
    thoughts_tokens: int
    finish_reason: str | None
    usd: float
    seconds: float


class Meter:
    """What one live run spends, request by request."""

    def __init__(
        self,
        budget: Budget,
        *,
        route_id: str,
        run_id: str,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._budget = budget
        self._price = budget.price(route_id)
        self._route_id = route_id
        self._run_id = run_id
        self.clock = clock
        self._calls: list[CallRecord] = []
        self._stop_reason: str | None = None
        self._held = 0.0

    @property
    def stop_reason(self) -> str | None:
        """Why the run must stop, once it must; None while it may go on. The first cause stays.

        "budget": the budget has no room for the next request. "rate_limited": a 429 outlived the
        retries. "ledger": the spend ledger cannot be read or written, so what a request costs can
        no longer be counted. After any of them no further request is sent.
        """
        return self._stop_reason

    def _stop(self, reason: str) -> None:
        if self._stop_reason is None:
            self._stop_reason = reason

    @property
    def held_usd(self) -> float:
        """What the requests in flight hold against the cap until they are settled."""
        return self._held

    def wrap(self, model: Model, role: str) -> Model:
        return MeteredModel(model, meter=self, role=role)

    def drain(self) -> tuple[CallRecord, ...]:
        """The requests since the last drain, oldest first."""
        calls, self._calls = tuple(self._calls), []
        return calls

    def reserve(self, max_tokens: int) -> float:
        """Refuse a request when the run has stopped or the budget has no room for it and for the
        requests in flight; otherwise hold the most it could cost, and return that.

        The hold lasts until the request is settled by `record`, `unconfirmed` or `release`."""
        if self._stop_reason is not None:
            raise RunStoppedError(
                f"the run has stopped ({self._stop_reason}): no further request is sent"
            )
        most = self._price.cost(input_units=MAX_INPUT_TOKENS_PER_REQUEST, output_units=max_tokens)
        try:
            self._budget.ensure_room(self._held + most)
        except BudgetExhaustedError:
            self._stop("budget")
            raise
        except (BudgetError, OSError):  # a ledger that cannot be read: nothing can be counted
            self._stop("ledger")
            raise
        self._held += most
        return most

    def release(self, reserved: float) -> None:
        """A request that cost nothing (the provider answered with an error status, or it never
        left) gives its hold back."""
        self._held = max(0.0, self._held - reserved)

    def _append(self, kind: str, usd: float) -> None:
        try:
            self._budget.record(run_id=self._run_id, route_id=self._route_id, kind=kind, usd=usd)
        except (BudgetError, OSError):  # a ledger that cannot be written: the next one is unpaid
            self._stop("ledger")
            raise

    def record(
        self, role: str, response: ModelResponse, seconds: float, *, reserved: float = 0.0
    ) -> None:
        """What the provider says the request used; `reserved` is the hold it gives back."""
        usage = response.usage
        usd = self._price.cost(input_units=usage.input_tokens, output_units=usage.output_tokens)
        try:
            self._append("chat", usd)
        finally:
            self.release(reserved)
        self._calls.append(
            CallRecord(
                agent=role,
                model_version=response.model_name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                thoughts_tokens=usage.details.get("thoughts_tokens", 0),
                finish_reason=response.finish_reason,
                usd=usd,
                seconds=seconds,
            )
        )

    def unconfirmed(self, role: str, usd: float, seconds: float) -> None:
        """A request that was sent but brought back no usage, recorded at its reserve `usd`, which
        is also the hold it gives back."""
        try:
            self._append("chat-unconfirmed", usd)
        finally:
            self.release(usd)
        self._calls.append(
            CallRecord(
                agent=role,
                model_version=None,
                input_tokens=0,
                output_tokens=0,
                thoughts_tokens=0,
                finish_reason="unconfirmed",
                usd=usd,
                seconds=seconds,
            )
        )

    def rate_limited(self) -> None:
        self._stop("rate_limited")


class MeteredModel(WrapperModel):
    def __init__(self, wrapped: Model, *, meter: Meter, role: str) -> None:
        super().__init__(wrapped)
        self._meter = meter
        self._role = role

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        cap = (model_settings or {}).get("max_tokens") or DEFAULT_MAX_TOKENS
        reserved = self._meter.reserve(cap)
        started = self._meter.clock()
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except ModelHTTPError as error:
            # The provider answered with an error status, which it does not bill.
            self._meter.release(reserved)
            if error.status_code == 429:
                self._meter.rate_limited()
            raise
        except NOT_SENT:
            # No credentials or no connection: the request never reached the provider.
            self._meter.release(reserved)
            raise
        except BaseException:
            # Sent, but no usage came back: the provider may have billed it. If the ledger cannot
            # take the line, the meter has stopped the run; what ended the request still goes on
            # (a cancellation must stay one).
            with suppress(BudgetError, OSError):
                self._meter.unconfirmed(self._role, reserved, self._meter.clock() - started)
            raise
        self._meter.record(self._role, response, self._meter.clock() - started, reserved=reserved)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        raise NotImplementedError("a metered model does not stream: its usage would go uncounted")
        yield  # pragma: no cover - makes this an async generator
