"""Hedged requests (Albi, 2026-10-12, live and in production): a request that has not answered
after HEDGE_AFTER_SECONDS gets an identical twin; the first answer wins and the other is cancelled.

Gemini 3.8 Flash on the eu multi-region answers in about 2 seconds at the median, but some requests
stayed stuck for 15 to 36 seconds and spent a whole turn. A twin sent after 4.5 seconds usually
answers first. The twin that loses tells a meter around it (frontdesk.model_gateway.metering) that
it was abandoned, with the winner's usage: an identical request, it is recorded at the winner's
input and at most its output cap, and it is no failure of the provider.

A fallback model (Albi, 2026-10-13: stability before Instagram). Vertex sometimes leaves the requests
of 3.8 Flash hanging until a 504, twins included, and 3.8 Flash is served in no other European
region. When the first model fails with an error status or a broken connection, the request goes to
the fallback at once, and so do the next ones for REST_SECONDS: the same stuck window is not waited
out again.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx2
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

HEDGE_AFTER_SECONDS: Final = 4.5
REST_SECONDS: Final = 120.0  # how long the first model rests after a failure
# The failures of the provider that the fallback answers: what the meter would have retried.
FALLBACK_STATUSES: Final = frozenset({429, 500, 502, 503, 504})


@dataclass
class Twin:
    """One of the two requests of a race. The loser learns it was abandoned, and the winner's usage,
    before it is cancelled."""

    abandoned: bool = False
    winner: RequestUsage | None = None


CURRENT_TWIN: Final[contextvars.ContextVar[Twin | None]] = contextvars.ContextVar(
    "frontdesk_current_twin", default=None
)


class HedgedModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        *,
        fallback: Model | None = None,
        after: float = HEDGE_AFTER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(wrapped)
        self._fallback = fallback
        self._after = after
        self._clock = clock
        self._rests_until = float("-inf")

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """The first answer of the request, of its twin, or of the fallback. An error ends the race
        only when no request is left to answer."""
        resting = self._fallback is not None and self._clock() < self._rests_until
        model = self._fallback if resting and self._fallback is not None else self.wrapped
        spare = None if resting else self._fallback
        twins: dict[asyncio.Task[ModelResponse], Twin] = {}

        def start(on: Model) -> None:
            self._start(twins, on, messages, model_settings, model_request_parameters)

        start(model)
        twinned = False
        error: BaseException | None = None
        try:
            while True:
                pending = {task for task in twins if not task.done()}
                if not pending:
                    raise error if error is not None else RuntimeError("no request answered")
                done, _ = await asyncio.wait(
                    pending,
                    timeout=None if twinned else self._after,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:  # still away: an identical twin
                    start(model)
                    twinned = True
                    continue
                for task in done:
                    failure = task.exception()
                    if failure is None:
                        return _won(twins, task)
                    error = error or failure
                    if spare is not None and _falls_back(failure):
                        self._rests_until = self._clock() + REST_SECONDS
                        start(spare)
                        spare = None
        finally:
            await _cancel_the_rest(twins)

    def _start(
        self,
        twins: dict[asyncio.Task[ModelResponse], Twin],
        model: Model,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> None:
        twin = Twin()
        context = contextvars.copy_context()
        context.run(CURRENT_TWIN.set, twin)
        request = model.request(list(messages), model_settings, model_request_parameters)
        twins[asyncio.create_task(request, context=context)] = twin


def _won(
    twins: dict[asyncio.Task[ModelResponse], Twin], winner: asyncio.Task[ModelResponse]
) -> ModelResponse:
    """The winner's answer; the requests still away learn they lost, and to whom."""
    response = winner.result()
    for task, twin in twins.items():
        if not task.done():
            twin.abandoned = True
            twin.winner = response.usage
    return response


def _falls_back(error: BaseException) -> bool:
    """A failure of the provider: an error status it would retry, a timeout, a broken connection."""
    if isinstance(error, ModelHTTPError):
        return error.status_code in FALLBACK_STATUSES
    return isinstance(error, httpx2.TimeoutException | httpx2.TransportError)


async def _cancel_the_rest(twins: dict[asyncio.Task[ModelResponse], Twin]) -> None:
    """Cancel the twins still away, and let them settle (with a meter, their ledger line)."""
    left = [task for task in twins if not task.done()]
    for task in left:
        task.cancel()
    if left:
        await asyncio.gather(*left, return_exceptions=True)
