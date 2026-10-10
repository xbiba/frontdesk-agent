"""A request that never answers, and the turn limit that cuts it, without waiting for the limit.

The turn limit is a timer of the event loop (`asyncio.timeout`). A test that shortens it, so as not
to wait, races the work that comes before the request (a database read, a cold start): cut too
early, the turn ends as a timeout where the test wanted its request to fail some other way, and
every healthy turn of the test has the same short limit. Here the limit stays what it is (45
seconds) and the clock of the loop jumps past it when the request hangs: the timer is due at once,
the loop cuts the request as it would after the wait, and nothing depends on how long anything
really took.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from frontdesk.conversation.turn import TURN_TIMEOUT_SECONDS

NEVER = 30.0  # seconds: a request that is not cut off would still return, and fail the test


class _JumpingClock:
    """The clock of an event loop, plus the seconds it has been moved forward."""

    def __init__(self, real: Callable[[], float]) -> None:
        self._real = real
        self.moved = 0.0

    def __call__(self) -> float:
        return self._real() + self.moved


def jump_ahead(loop: asyncio.AbstractEventLoop, seconds: float) -> None:
    """Move the clock of `loop` forward without waiting: a timer set to go off within `seconds` is
    due at the next turn of the loop. Only the clock of this loop moves; the machine's does not."""
    clock = loop.time
    if not isinstance(clock, _JumpingClock):
        clock = _JumpingClock(clock)
        loop.time = clock  # type: ignore[method-assign]
    clock.moved += seconds


async def hang(limit: float = TURN_TIMEOUT_SECONDS) -> None:
    """Never answers. The turn limit (`limit` seconds) passes at once on the clock of the running
    loop, which cuts the request: the turn ends as a timeout, and the request was away."""
    jump_ahead(asyncio.get_running_loop(), limit + 1)
    await asyncio.sleep(NEVER)
