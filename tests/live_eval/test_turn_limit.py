"""The hang of the outage tests ends at the turn limit with nobody waiting for the limit."""

from __future__ import annotations

import asyncio

import pytest

from frontdesk.conversation.turn import TURN_TIMEOUT_SECONDS
from tests.live_eval.turn_limit import hang, jump_ahead


async def test_a_request_that_hangs_is_cut_by_the_turn_limit_without_waiting_for_it() -> None:
    # Were the jump not to reach the timer, the request would return after `NEVER` seconds and
    # this would fail for that, not hang for 45 seconds.
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            await hang()


async def test_each_request_that_hangs_is_cut_by_its_own_turn() -> None:
    for _ in range(3):  # three turns of one run, on one loop
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
                await hang()


async def test_a_turn_that_started_after_the_jump_has_its_whole_limit() -> None:
    jump_ahead(asyncio.get_running_loop(), 10 * TURN_TIMEOUT_SECONDS)
    async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
        await asyncio.sleep(0)  # not cut: its limit runs from the clock as it is now


async def test_the_limit_of_a_longer_turn_is_passed_when_the_hang_is_told_of_it() -> None:
    limit = 600.0
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(limit):
            await hang(limit)


async def test_only_the_loop_that_hangs_has_its_clock_moved() -> None:
    other = asyncio.new_event_loop()
    try:
        before = other.time()
        jump_ahead(asyncio.get_running_loop(), 1_000.0)
        assert other.time() - before < 1_000.0
    finally:
        other.close()
