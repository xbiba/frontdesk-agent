"""Metering of live model requests for the M3 live evaluation (design §4.4, §6).

`Meter.wrap` puts a `MeteredModel` around the model an agent runs on. The wrapper does the retrying
itself (the SDK makes one attempt, see `GOOGLE_RETRIES`), so that nothing hides below the meter, and
settles every attempt on its own:

- a request whose cost the reserve cannot bound (no output cap, or more bytes than the input the
  reserve assumes) is not sent at all;
- before an attempt it holds the most the attempt could cost, and refuses it when the budget has no
  room for that and for the attempts still in flight;
- an attempt that was answered with usage costs what the provider reported;
- one that was answered with an error status, or that never left (no credentials, no connection),
  costs nothing;
- any other attempt that was sent (a timeout, a reset, a turn cancelled by its time limit or Ctrl-C,
  an answer that reports no usage or one the client could not read) may have been billed in full,
  and is recorded at its reserve.

A 429 that outlives the attempts, a budget with no room, and a ledger that cannot be read or written
each stop the run for good, so that it can resume later: after a stop no further request is sent.
Counted this way the ledger is an upper bound of what the provider billed, with one exception that
no code can close: a process killed between a provider's answer and its ledger line loses that
line, one request at most. The runner reads the calls of each turn with `drain`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Final, NoReturn

import httpx2
from google.auth.exceptions import GoogleAuthError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage
from pydantic_core import to_json

from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError

# The reserve assumes a request holds at most this many input tokens. Real requests stay far below
# it (the prompts are under 3,000 characters, the evidence at most 4,000), and `Meter.admit` checks
# it: no tokenizer makes more tokens than the UTF-8 bytes of its input, so a request of more bytes
# is not sent.
MAX_INPUT_TOKENS_PER_REQUEST: Final = 32_000
# The errors that end a request before it reaches the provider, which bills nothing for it: no
# credentials, no connection. An error status comes back as ModelHTTPError, also unbilled.
NOT_SENT: Final = (GoogleAuthError, httpx2.ConnectError, httpx2.ConnectTimeout)
# An answer of an error status is not billed either.
UNBILLED: Final = (ModelHTTPError, *NOT_SENT)
# What google-genai retried itself before the engine took over: 429 and 5xx answers, timeouts and
# refused connections; four attempts, with waits of 2, 4 and 8 seconds in between (M3 design §6).
# `GOOGLE_RETRIES` keeps the same policy, and a test holds the two together.
RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
RETRY_WAITS: Final = (2.0, 4.0, 8.0)
MAX_ATTEMPTS: Final = len(RETRY_WAITS) + 1


class RunStoppedError(RuntimeError):
    """A request was refused because the run has already stopped (see `Meter.stop_reason`)."""


class UnboundedRequestError(BudgetExhaustedError):
    """A request whose cost the reserve cannot bound (no output cap, too long) is not sent."""


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
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._budget = budget
        self._price = budget.price(route_id)
        if self._price.unit != "tokens_1m":
            raise BudgetError(
                f"{budget.path}: route '{route_id}' is priced per {self._price.unit}, "
                "and the meter counts tokens"
            )
        self._route_id = route_id
        self._run_id = run_id
        self.clock = clock
        self.sleep = sleep
        self._calls: list[CallRecord] = []
        self._stop_reason: str | None = None
        self._held = 0.0

    @property
    def stop_reason(self) -> str | None:
        """Why the run must stop, once it must; None while it may go on. The first cause stays.

        "budget": the budget has no room for the next request, or cannot bound what it would cost.
        "rate_limited": a 429 outlived the attempts. "ledger": the spend ledger cannot be read or written, so what a request costs can
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

    def admit(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
        model_settings: ModelSettings | None,
    ) -> int:
        """The output cap of a request, when the reserve can bound what it would cost.

        The reserve covers an output cap and at most `MAX_INPUT_TOKENS_PER_REQUEST` input tokens. A
        request without a cap would be left to the provider's own, far larger, one; a request of
        more bytes than that may hold more tokens. Either is refused before it is sent, and the
        run stops: nothing the cap rests on can be said of it.
        """
        cap = (model_settings or {}).get("max_tokens")
        if not cap:
            self._refuse("has no output cap")
        size = _request_bytes(messages, parameters)
        if size > MAX_INPUT_TOKENS_PER_REQUEST:
            self._refuse(
                f"is {size} bytes long, more than the {MAX_INPUT_TOKENS_PER_REQUEST} tokens the "
                "reserve assumes"
            )
        return cap

    def _refuse(self, why: str) -> NoReturn:
        self._stop("budget")
        raise UnboundedRequestError(
            f"the request {why}: its cost cannot be bounded, so it is not sent"
        )

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
        """What the provider says the attempt used; `reserved` is the hold it gives back. An
        answer that reports no usage may still have been billed in full: it counts its reserve."""
        usage = response.usage
        if usage.input_tokens + usage.output_tokens == 0:
            self.unconfirmed(role, reserved, seconds)
            return
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


def _request_bytes(messages: list[ModelMessage], parameters: ModelRequestParameters) -> int:
    """The UTF-8 length of what a request carries: its messages, instructions included, and the
    definitions of its tools."""
    return len(ModelMessagesTypeAdapter.dump_json(messages)) + len(
        to_json(parameters, fallback=str)
    )


def _worth_retrying(error: Exception) -> bool:
    """What google-genai used to retry: a 429 or 5xx answer, a timeout, a refused connection."""
    if isinstance(error, ModelHTTPError):
        return error.status_code in RETRY_STATUSES
    return isinstance(error, httpx2.TimeoutException | httpx2.ConnectError)


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
        cap = self._meter.admit(messages, model_request_parameters, model_settings)
        for attempt in range(MAX_ATTEMPTS):
            try:
                return await self._attempt(messages, model_settings, model_request_parameters, cap)
            except Exception as error:
                if attempt == MAX_ATTEMPTS - 1 or not _worth_retrying(error):
                    if isinstance(error, ModelHTTPError) and error.status_code == 429:
                        self._meter.rate_limited()  # it outlived the attempts
                    raise  # what the last attempt ended with, for the runner's stop logic
            await self._meter.sleep(RETRY_WAITS[attempt])
        raise AssertionError("the last attempt returns or raises")  # pragma: no cover

    async def _attempt(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        cap: int,
    ) -> ModelResponse:
        """One attempt: held while it is away, settled as soon as it ends."""
        reserved = self._meter.reserve(cap)
        started = self._meter.clock()
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except UNBILLED:
            # An error status, or a request that never left: the provider billed nothing.
            self._meter.release(reserved)
            raise
        except BaseException:
            # Sent, but no usage came back: the provider may have billed it. If the ledger cannot
            # take the line, the meter has stopped the run; what ended the attempt still goes on
            # (a cancellation must stay one).
            with suppress(BudgetError, OSError):
                self._meter.unconfirmed(self._role, reserved, self._meter.clock() - started)
            raise
        self._meter.record(self._role, response, self._meter.clock() - started, reserved=reserved)
        return response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        raise NotImplementedError(
            "a metered model does not count tokens: the call would go unmetered"
        )

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        raise NotImplementedError(
            "a metered model does not compact messages: the call would go unmetered"
        )

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
