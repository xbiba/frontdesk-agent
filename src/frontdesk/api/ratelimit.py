"""A per-address limit kept in one instance's memory (the site's chat, design of 2026-11-07 §3).

The webhook runs at most two instances, so the count is approximate, and that is enough: the
engine's own limits hold behind it. No address is kept, not even here: each is hashed with a key
drawn when the process starts, and its hits are forgotten once the window has passed.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import deque
from collections.abc import Callable
from typing import Final

from fastapi import Request

MAX_KEYS: Final = 20_000  # beyond this many addresses in one window, new ones are refused


class SlidingWindow:
    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = MAX_KEYS,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._salt = secrets.token_bytes(32)
        self._hits: dict[str, deque[float]] = {}

    def allow(self, address: str) -> bool:
        """Count a hit for `address`; False once it had `limit` hits within the window."""
        now = self._clock()
        key = hmac.new(self._salt, address.encode(), hashlib.sha256).hexdigest()
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self._max_keys:
                self._forget(now)
            if len(self._hits) >= self._max_keys:
                return False  # a flood from many addresses: refuse rather than grow
            hits = self._hits[key] = deque()
        while hits and hits[0] <= now - self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True

    def _forget(self, now: float) -> None:
        old = [
            key for key, hits in self._hits.items() if not hits or hits[-1] <= now - self._window
        ]
        for key in old:
            del self._hits[key]


def client_address(request: Request) -> str:
    """The caller's address: the last X-Forwarded-For entry, the one Cloud Run's front end adds
    (an entry the caller wrote comes before it), or the connection's peer."""
    forwarded = request.headers.get("x-forwarded-for", "")
    entries = [entry.strip() for entry in forwarded.split(",") if entry.strip()]
    if entries:
        return entries[-1]
    return request.client.host if request.client else "unknown"
