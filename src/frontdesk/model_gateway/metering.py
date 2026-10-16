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
Each stop keeps its cause (`stop_detail`), so that the command line can say it. A run may also ask
for a breaker: when a request ends in failure and the last few attempts of its agent have all failed,
in whatever way, the run stops too. Each agent (the classifier, the responder) counts its own, so
that one that fails alone is not hidden by the other answering. Without the breaker an outage would
use up the cap on reservations, because every attempt that was sent and brought nothing back is
recorded at its reserve.
Counted this way the ledger is an upper bound of what the provider billed, with two exceptions of one
request each that no code can close: a process killed between a provider's answer and its ledger
line loses that line, and so does the first request whose line a ledger that has become unwritable
refuses (the run stops there). `check_ledger` finds a ledger that is unwritable from the start before
the first paid request. The runner reads the calls of each turn with `drain`.
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

from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError, Price
from frontdesk.model_gateway.hedging import CURRENT_TWIN

# The reserve assumes a request holds at most this many input tokens. Real requests stay far below
# it (the prompts are under 3,000 characters, the evidence at most 4,000), and `Meter.admit` checks
# it: no tokenizer makes more tokens than the UTF-8 bytes of its input, so a request of more bytes
# is not sent.
MAX_INPUT_TOKENS_PER_REQUEST: Final = 32_000
# The reception agent's requests carry the conversation, the facts, its tools' results and Gemini's
# thought signatures, which this JSON measure counts as bytes though they are not prompt text: one of
# about 2,000 input tokens measured 32,349 bytes in the run of 2026-10-12. Its role gets a larger
# bound, and its reserve holds as much.
AGENT_INPUT_TOKENS_PER_REQUEST: Final = 96_000
AGENT_ROLE: Final = "agent"
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
# The agent answers a patient who waits: a reply within 20 seconds (decision of 2026-10-12), and a
# twin goes out after 4.5 seconds anyway (frontdesk.model_gateway.hedging). The run of 2026-10-13
# lost up to 29 seconds of a turn to the waits above.
AGENT_RETRY_WAITS: Final = (1.0, 2.0, 4.0)
MAX_ATTEMPTS: Final = len(RETRY_WAITS) + 1
# Why a run stops (`Meter.stop_reason`); a command maps each to an exit code.
STOP_BUDGET: Final = "budget"
STOP_RATE_LIMITED: Final = "rate_limited"
STOP_LEDGER: Final = "ledger"
STOP_UNAVAILABLE: Final = "unavailable"
# A run that asks for the breaker (`Meter(breaker=...)`) stops once a request has ended in failure
# with this many attempts in a row failed (M3 design §6).
BREAKER_ATTEMPTS: Final = 3
# The kind of the line `check_ledger` appends: it costs nothing, so it is no paid call.
PREFLIGHT_KIND: Final = "preflight"


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
        breaker: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if breaker is not None and breaker < 1:
            raise ValueError("the breaker needs at least one failed attempt to trip")
        self._budget = budget
        self._price = _token_price(budget, route_id)
        self._route_id = route_id
        self._run_id = run_id
        self.clock = clock
        self.sleep = sleep
        self._breaker = breaker
        self._calls: list[CallRecord] = []
        self._stop_reason: str | None = None
        self._stop_detail: str | None = None
        self._held = 0.0
        self._failures: dict[str, int] = {}

    @property
    def stop_reason(self) -> str | None:
        """Why the run must stop, once it must; None while it may go on. The first cause stays.

        "budget": the budget has no room for the next request, or cannot bound what it would cost.
        "rate_limited": a 429 outlived the attempts. "ledger": the spend ledger cannot be read or
        written, so what a request costs can no longer be counted. "unavailable": the breaker
        tripped. After any of them no further request is sent.
        """
        return self._stop_reason

    @property
    def stop_detail(self) -> str | None:
        """The cause of `stop_reason` in a sentence: which limit, which file, how the last request
        ended. It is the cause of the first stop, like the reason."""
        return self._stop_detail

    @property
    def consecutive_failures(self) -> dict[str, int]:
        """For each agent that has made an attempt, the attempts in a row, up to now, that were not
        answered with usage: what the breaker counts."""
        return dict(self._failures)

    def _stop(self, reason: str, detail: str) -> None:
        if self._stop_reason is None:
            self._stop_reason, self._stop_detail = reason, detail

    def _ledger_detail(self, error: BudgetError | OSError, action: str) -> str:
        """What went wrong with the ledger, with its name: a BudgetError names it already."""
        if isinstance(error, OSError):
            return f"{self._budget.ledger_path}: cannot be {action} ({error.strerror or error})"
        return str(error)

    @property
    def held_usd(self) -> float:
        """What the requests in flight hold against the cap until they are settled."""
        return self._held

    def wrap(
        self, model: Model, role: str, *, retries: bool = True, route_id: str | None = None
    ) -> Model:
        """`model` metered for `role`. Without retries, one attempt and no stop for a rate limit:
        a fallback answers the failure (frontdesk.model_gateway.hedging). `route_id`: the route of
        a model other than the run's own (a fallback, a second try), priced at its own price."""
        self._price_of(route_id)  # a route with no price is refused before any request
        return MeteredModel(model, meter=self, role=role, retries=retries, route_id=route_id)

    def _price_of(self, route_id: str | None) -> Price:
        """The price of a route's requests: the run's own route when None."""
        if route_id is None or route_id == self._route_id:
            return self._price
        return _token_price(self._budget, route_id)

    def drain(self) -> tuple[CallRecord, ...]:
        """The requests since the last drain, oldest first."""
        calls, self._calls = tuple(self._calls), []
        return calls

    def admit(
        self,
        messages: list[ModelMessage],
        parameters: ModelRequestParameters,
        model_settings: ModelSettings | None,
        role: str = "",
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
        bound = input_bound(role)
        if size > bound:
            self._refuse(f"is {size} bytes long, more than the {bound} tokens the reserve assumes")
        return cap

    def _refuse(self, why: str) -> NoReturn:
        message = f"the request {why}: its cost cannot be bounded, so it is not sent"
        self._stop(STOP_BUDGET, message)
        raise UnboundedRequestError(message)

    def reserve(self, max_tokens: int, role: str = "", route_id: str | None = None) -> float:
        """Refuse a request when the run has stopped or the budget has no room for it and for the
        requests in flight; otherwise hold the most it could cost, and return that.

        The hold lasts until the request is settled by `record`, `unconfirmed` or `release`."""
        if self._stop_reason is not None:
            raise RunStoppedError(
                f"the run has stopped ({self._stop_reason}): no further request is sent"
            )
        price = self._price_of(route_id)
        most = price.cost(input_units=input_bound(role), output_units=max_tokens)
        try:
            self._budget.ensure_room(self._held + most)
        except BudgetExhaustedError as error:
            self._stop(STOP_BUDGET, str(error))
            raise
        except (BudgetError, OSError) as error:  # a ledger that cannot be read: nothing is counted
            self._stop(STOP_LEDGER, self._ledger_detail(error, "read"))
            raise
        self._held += most
        return most

    def release(self, reserved: float) -> None:
        """A request that cost nothing (the provider answered with an error status, or it never
        left) gives its hold back."""
        self._held = max(0.0, self._held - reserved)

    def failed(self, role: str, reserved: float, seconds: float, reason: str) -> None:
        """An attempt of `role` that cost nothing (the provider answered with an error status, or
        the request never left): it gives its hold back, it is one more failure in a row, and it is
        a call of the turn with `reason` and its time, which no record showed before."""
        self.release(reserved)
        self._failures[role] = self._failures.get(role, 0) + 1
        self._calls.append(
            CallRecord(
                agent=role,
                model_version=None,
                input_tokens=0,
                output_tokens=0,
                thoughts_tokens=0,
                finish_reason=reason,
                usd=0.0,
                seconds=seconds,
            )
        )

    def _append(self, kind: str, usd: float, route_id: str | None = None) -> None:
        route = route_id or self._route_id
        try:
            self._budget.record(run_id=self._run_id, route_id=route, kind=kind, usd=usd)
        except (BudgetError, OSError) as error:
            # A ledger that cannot be written: the next request would be paid and uncounted.
            self._stop(STOP_LEDGER, self._ledger_detail(error, "appended to"))
            raise

    def check_ledger(self) -> None:
        """Prove that the ledger can be read and appended to, before the first paid request: read
        it, so that a damaged one is found, then append a line of zero dollars, as a paid request
        will. It costs nothing and sends no request. A ledger that fails stops the run, and the
        BudgetError names the file and the reason."""
        try:
            self._budget.spent_usd()
        except BudgetError as error:
            self._stop(STOP_LEDGER, str(error))
            raise
        except OSError as error:
            detail = self._ledger_detail(error, "read")
            self._stop(STOP_LEDGER, detail)
            raise BudgetError(detail) from error
        try:
            self._append(PREFLIGHT_KIND, 0.0)
        except OSError as error:  # `_append` has stopped the run; a BudgetError passes as it is
            raise BudgetError(self._ledger_detail(error, "appended to")) from error

    def record(
        self,
        role: str,
        response: ModelResponse,
        seconds: float,
        *,
        reserved: float,
        route_id: str | None = None,
    ) -> None:
        """What the provider says the attempt used; `reserved` is the hold it gives back. An
        answer that reports no usage may still have been billed in full: it counts its reserve."""
        usage = response.usage
        if usage.input_tokens + usage.output_tokens == 0:
            self.unconfirmed(role, reserved, seconds)
            return
        price = self._price_of(route_id)
        usd = price.cost(input_units=usage.input_tokens, output_units=usage.output_tokens)
        self._failures[role] = 0  # answered with usage: whatever this agent failed is behind it
        try:
            self._append("chat", usd, route_id)
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

    def unconfirmed(
        self, role: str, usd: float, seconds: float, route_id: str | None = None
    ) -> None:
        """A request that was sent but brought back no usage, recorded at its reserve `usd`, which
        is also the hold it gives back. It is one more failure in a row for `role`."""
        self._failures[role] = self._failures.get(role, 0) + 1
        try:
            self._append("chat-unconfirmed", usd, route_id)
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

    def abandoned(
        self,
        role: str,
        reserved: float,
        winner: RequestUsage,
        cap: int,
        seconds: float,
        route_id: str | None = None,
    ) -> None:
        """A hedged twin cancelled after the other answered (frontdesk.model_gateway.hedging): the
        winner's input tokens and at most `cap` output tokens, which may have been billed. It is no
        failure in a row: its twin answered."""
        usd = self._price_of(route_id).cost(input_units=winner.input_tokens, output_units=cap)
        try:
            self._append("chat-hedged", usd, route_id)
        finally:
            self.release(reserved)
        self._calls.append(
            CallRecord(
                agent=role,
                model_version=None,
                input_tokens=winner.input_tokens,
                output_tokens=0,
                thoughts_tokens=0,
                finish_reason="hedged",
                usd=usd,
                seconds=seconds,
            )
        )

    def rate_limited(self) -> None:
        self._stop(
            STOP_RATE_LIMITED, "Google answered 429 (rate limited) at the last attempt of a request"
        )

    def request_ended(self, role: str, error: BaseException | None) -> None:
        """A request of `role` ended, with `error` or with an answer. When the breaker is on and
        the attempts in a row that failed (an answer without usage is one) have reached its limit
        for `role`, the run stops: the provider is not answering it, and each further attempt that
        is sent without an answer would be recorded at its reserve. The retries of the request that
        just ended were not cut short: a request that fails three times and then answers has no
        failure left in the count."""
        count = self._failures.get(role, 0)
        if self._breaker is None or count < self._breaker:
            return
        how = "was answered without usage" if error is None else _ending(error)
        attempts = f"{count} attempt{'s' if count != 1 else ''}"
        self._stop(
            STOP_UNAVAILABLE, f"{attempts} of the {role} in a row failed; the last request {how}"
        )


def _token_price(budget: Budget, route_id: str) -> Price:
    price = budget.price(route_id)
    if price.unit != "tokens_1m":
        raise BudgetError(
            f"{budget.path}: route '{route_id}' is priced per {price.unit}, "
            "and the meter counts tokens"
        )
    return price


def retry_waits(role: str) -> tuple[float, ...]:
    """The waits between the attempts of a request of this role."""
    return AGENT_RETRY_WAITS if role == AGENT_ROLE else RETRY_WAITS


def _unbilled(error: Exception) -> str:
    """Why an attempt cost nothing: the error status it was answered with, or that it never left."""
    return f"error {error.status_code}" if isinstance(error, ModelHTTPError) else "not sent"


def input_bound(role: str) -> int:
    """The most input tokens a request of this role may hold, and its reserve assumes."""
    return AGENT_INPUT_TOKENS_PER_REQUEST if role == AGENT_ROLE else MAX_INPUT_TOKENS_PER_REQUEST


def _request_bytes(messages: list[ModelMessage], parameters: ModelRequestParameters) -> int:
    """The UTF-8 length of what a request carries: its messages, instructions included, and the
    definitions of its tools."""
    return len(ModelMessagesTypeAdapter.dump_json(messages)) + len(
        to_json(parameters, fallback=str)
    )


def _ending(error: BaseException) -> str:
    """How a request ended, without a word of what the provider wrote: a status or a type. A
    request cancelled while it was away timed out: the turn's time limit cut it off."""
    if isinstance(error, ModelHTTPError):
        return f"ended with HTTP {error.status_code}"
    if isinstance(error, asyncio.CancelledError):
        return "timed out"
    return f"ended with {type(error).__name__}"


def _worth_retrying(error: Exception) -> bool:
    """What google-genai used to retry: a 429 or 5xx answer, a timeout, a refused connection."""
    if isinstance(error, ModelHTTPError):
        return error.status_code in RETRY_STATUSES
    return isinstance(error, httpx2.TimeoutException | httpx2.ConnectError)


class MeteredModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        *,
        meter: Meter,
        role: str,
        retries: bool = True,
        route_id: str | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._meter = meter
        self._role = role
        self._retries = retries
        self._route_id = route_id

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """The request in up to `MAX_ATTEMPTS` attempts. The last attempt ends it with what it
        ended with, which is what the runner's stop logic reads (a 429, a refusal by the budget).
        However it ends, the meter is told, for the breaker."""
        cap = self._meter.admit(messages, model_request_parameters, model_settings, self._role)
        try:
            response = await self._attempts(messages, model_settings, model_request_parameters, cap)
        except BaseException as error:
            self._meter.request_ended(self._role, error)
            raise
        self._meter.request_ended(self._role, None)
        return response

    async def _attempts(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        cap: int,
    ) -> ModelResponse:
        attempts = MAX_ATTEMPTS if self._retries else 1
        for attempt in range(attempts):
            try:
                return await self._attempt(messages, model_settings, model_request_parameters, cap)
            except Exception as error:
                last = attempt == attempts - 1
                # Once the run has stopped (a ledger that failed under this very attempt) nothing
                # more is sent, so there is nothing to wait for.
                if last or self._meter.stop_reason is not None or not _worth_retrying(error):
                    rate_limited = isinstance(error, ModelHTTPError) and error.status_code == 429
                    if rate_limited and self._retries:
                        self._meter.rate_limited()  # it outlived the attempts
                    raise
            await self._meter.sleep(retry_waits(self._role)[attempt])
        raise AssertionError("the last attempt returns or raises")  # pragma: no cover

    async def _attempt(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        cap: int,
    ) -> ModelResponse:
        """One attempt: held while it is away, settled as soon as it ends."""
        route = self._route_id
        reserved = self._meter.reserve(cap, self._role, route)
        started = self._meter.clock()
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except UNBILLED as error:
            # An error status, or a request that never left: the provider billed nothing.
            seconds = self._meter.clock() - started
            self._meter.failed(self._role, reserved, seconds, _unbilled(error))
            raise
        except BaseException:
            # Sent, but no usage came back: the provider may have billed it. If the ledger cannot
            # take the line, the meter has stopped the run; what ended the attempt still goes on
            # (a cancellation must stay one). A hedged twin that lost is an identical request to
            # the winner's: its input, at most its output cap, and no failure of the provider.
            seconds = self._meter.clock() - started
            twin = CURRENT_TWIN.get()
            with suppress(BudgetError, OSError):
                if twin is not None and twin.abandoned and twin.winner is not None:
                    self._meter.abandoned(self._role, reserved, twin.winner, cap, seconds, route)
                else:
                    self._meter.unconfirmed(self._role, reserved, seconds, route)
            raise
        seconds = self._meter.clock() - started
        self._meter.record(self._role, response, seconds, reserved=reserved, route_id=route)
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
