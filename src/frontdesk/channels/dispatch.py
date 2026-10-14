"""Handing a conversation to the worker (ADR-0011, M4 design D1, D12): a Cloud Tasks task calls
the private worker after a delay, with the invoker's Google identity, which Cloud Run checks."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol

import httpx2

from frontdesk.channels.inbox import valid_key

TASKS_API: Final = "https://cloudtasks.googleapis.com/v2"
_QUEUE: Final = re.compile(
    r"^projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/locations/[a-z0-9-]+/queues/[A-Za-z0-9-]{1,100}$"
)


class DispatchError(RuntimeError):
    """Cloud Tasks did not take the task: the webhook answers 500, and Meta delivers again."""


class Dispatcher(Protocol):
    async def dispatch(self, conversation_id: str, *, delay: timedelta) -> None: ...


class CloudTasksDispatcher:
    def __init__(
        self,
        *,
        queue: str,
        worker_url: str,
        invoker: str,
        token: Callable[[], str],
        client: httpx2.AsyncClient,
        now: Callable[[], datetime],
    ) -> None:
        if not _QUEUE.match(queue):
            raise ValueError(f"not a Cloud Tasks queue path: {queue!r}")
        self._url = f"{TASKS_API}/{queue}/tasks"
        self._worker = worker_url.rstrip("/")
        self._invoker = invoker
        self._token = token
        self._client = client
        self._now = now

    async def dispatch(self, conversation_id: str, *, delay: timedelta) -> None:
        if not valid_key(conversation_id):
            raise ValueError("not a conversation key")
        when = (self._now() + delay).astimezone(UTC).isoformat(timespec="milliseconds")
        task = {
            "task": {
                "scheduleTime": when.replace("+00:00", "Z"),
                "httpRequest": {
                    "httpMethod": "POST",
                    "url": f"{self._worker}/tasks/conversations/{conversation_id}",
                    "oidcToken": {"serviceAccountEmail": self._invoker, "audience": self._worker},
                },
            }
        }
        token = await asyncio.to_thread(self._token)  # a refresh may wait on the network
        try:
            response = await self._client.post(
                self._url, json=task, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx2.HTTPError as error:
            raise DispatchError(type(error).__name__) from None
        if response.status_code != 200:
            raise DispatchError(f"Cloud Tasks answered {response.status_code}")
