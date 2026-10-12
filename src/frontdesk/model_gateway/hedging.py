"""Hedged requests (Albi, 2026-10-12, live and in production): a request that has not answered
after HEDGE_AFTER_SECONDS gets an identical twin; the first answer wins and the other is cancelled.

Gemini 3.8 Flash on the eu multi-region answers in about 2 seconds at the median, but some requests
stayed stuck for 15 to 36 seconds and spent a whole turn. A twin sent after 4.5 seconds usually
answers first. The twin that loses tells a meter around it (frontdesk.model_gateway.metering) that
it was abandoned, with the winner's usage: an identical request, it is recorded at the winner's
input and at most its output cap, and it is no failure of the provider.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Final

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

HEDGE_AFTER_SECONDS: Final = 4.5


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
    def __init__(self, wrapped: Model, *, after: float = HEDGE_AFTER_SECONDS) -> None:
        super().__init__(wrapped)
        self._after = after

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """The first answer of the request or of its twin. An error ends the race only when no
        twin is left to answer."""
        twins: dict[asyncio.Task[ModelResponse], Twin] = {}
        first = self._start(twins, messages, model_settings, model_request_parameters)
        try:
            done, _ = await asyncio.wait({first}, timeout=self._after)
            if done:
                return first.result()
            self._start(twins, messages, model_settings, model_request_parameters)
            return await self._first_answer(twins)
        finally:
            await _cancel_the_rest(twins)

    def _start(
        self,
        twins: dict[asyncio.Task[ModelResponse], Twin],
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> asyncio.Task[ModelResponse]:
        twin = Twin()
        context = contextvars.copy_context()
        context.run(CURRENT_TWIN.set, twin)
        request = self.wrapped.request(list(messages), model_settings, model_request_parameters)
        task = asyncio.create_task(request, context=context)
        twins[task] = twin
        return task

    async def _first_answer(self, twins: dict[asyncio.Task[ModelResponse], Twin]) -> ModelResponse:
        pending = set(twins)
        error: BaseException | None = None
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                failure = task.exception()
                if failure is None:
                    response = task.result()
                    for loser in pending:
                        twins[loser].abandoned = True
                        twins[loser].winner = response.usage
                    return response
                error = error or failure
        raise error if error is not None else RuntimeError("no twin answered")


async def _cancel_the_rest(twins: dict[asyncio.Task[ModelResponse], Twin]) -> None:
    """Cancel the twins still away, and let them settle (with a meter, their ledger line)."""
    left = [task for task in twins if not task.done()]
    for task in left:
        task.cancel()
    if left:
        await asyncio.gather(*left, return_exceptions=True)
