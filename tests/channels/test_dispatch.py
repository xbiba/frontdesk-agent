"""Handing a conversation to the worker (ADR-0011): a Cloud Tasks task that calls the private
worker after the burst window, with the invoker's identity."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest

from frontdesk.channels.dispatch import TASKS_API, CloudTasksDispatcher, DispatchError

QUEUE = "projects/demo-clinic/locations/europe-west1/queues/frontdesk-demo"
WORKER = "https://frontdesk-demo-worker-123456789.europe-west1.run.app"
INVOKER = "frontdesk-demo-invoker@demo-clinic.iam.gserviceaccount.com"
KEY = "instagram-" + "a" * 40
NOW = datetime(2026, 10, 14, 9, tzinfo=UTC)


def _dispatcher(handler: Any) -> CloudTasksDispatcher:
    return CloudTasksDispatcher(
        queue=QUEUE,
        worker_url=WORKER,
        invoker=INVOKER,
        token=lambda: "t0ken",
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        now=lambda: NOW,
    )


async def test_a_conversation_becomes_a_task_for_the_worker() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"name": f"{QUEUE}/tasks/1"})

    await _dispatcher(handler).dispatch(KEY, delay=timedelta(seconds=3))
    (request,) = seen
    assert str(request.url) == f"{TASKS_API}/{QUEUE}/tasks"
    assert request.headers["authorization"] == "Bearer t0ken"
    assert json.loads(request.content) == {
        "task": {
            "scheduleTime": "2026-10-14T09:00:03.000Z",
            "httpRequest": {
                "httpMethod": "POST",
                "url": f"{WORKER}/tasks/conversations/{KEY}",
                "oidcToken": {"serviceAccountEmail": INVOKER, "audience": WORKER},
            },
        }
    }


async def test_a_refused_task_raises() -> None:
    with pytest.raises(DispatchError, match="403"):
        await _dispatcher(lambda request: httpx2.Response(403)).dispatch(KEY, delay=timedelta(0))


async def test_an_unreachable_queue_raises() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(DispatchError, match="ConnectError"):
        await _dispatcher(handler).dispatch(KEY, delay=timedelta(0))


async def test_only_conversation_keys_are_dispatched() -> None:
    with pytest.raises(ValueError, match="conversation key"):
        await _dispatcher(lambda request: httpx2.Response(200)).dispatch(
            "../sweep", delay=timedelta(0)
        )


def test_a_queue_must_be_a_queue_path() -> None:
    with pytest.raises(ValueError, match="queue"):
        CloudTasksDispatcher(
            queue="frontdesk",
            worker_url=WORKER,
            invoker=INVOKER,
            token=lambda: "t",
            client=httpx2.AsyncClient(),
            now=lambda: NOW,
        )
