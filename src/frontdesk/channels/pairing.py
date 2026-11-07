"""Guessing the pairing code on a channel anyone may write to (the site's chat, security review of
2026-11-07). In tester mode every message of someone who is not a tester is a try at the code: a
session that missed `per_user` times, or a process that saw `per_process` misses, pairs no one
within the window, the right code included. Counted in this instance's memory, as the address
limit is; the alert names the lock only, never a session or a text."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Final

from frontdesk.observability.logs import log_alert

logger = logging.getLogger(__name__)
MISSES_PER_SESSION: Final = 5
MISSES_PER_PROCESS: Final = 200
WINDOW_SECONDS: Final = 3600.0
_MAX_USERS: Final = 20_000


class PairingGuard:
    def __init__(
        self,
        *,
        per_user: int = MISSES_PER_SESSION,
        per_process: int = MISSES_PER_PROCESS,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_user = per_user
        self._per_process = per_process
        self._window = window_seconds
        self._clock = clock
        self._users: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()

    def locked(self, user_id: str) -> bool:
        now = self._clock()
        self._expire(self._all, now)
        if len(self._all) >= self._per_process:
            return True
        misses = self._users.get(user_id)
        if misses is None:
            return False
        self._expire(misses, now)
        return len(misses) >= self._per_user

    def miss(self, user_id: str) -> None:
        now = self._clock()
        if len(self._users) >= _MAX_USERS:
            self._forget(now)
        misses = self._users.setdefault(user_id, deque())
        self._expire(misses, now)
        self._expire(self._all, now)
        misses.append(now)
        self._all.append(now)
        if len(misses) == self._per_user:
            log_alert(logger, "channel.pairing_locked", scope="session")
        if len(self._all) == self._per_process:
            log_alert(logger, "channel.pairing_locked", scope="process")

    def _expire(self, misses: deque[float], now: float) -> None:
        while misses and misses[0] <= now - self._window:
            misses.popleft()

    def _forget(self, now: float) -> None:
        for user in [u for u, m in self._users.items() if not m or m[-1] <= now - self._window]:
            del self._users[user]
