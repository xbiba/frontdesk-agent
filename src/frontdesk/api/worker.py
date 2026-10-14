"""The worker's routes (ADR-0011): Cloud Tasks calls one for each conversation, Cloud Scheduler the
others. Cloud Run lets in only the invoker's identity, so these routes check nothing themselves."""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from frontdesk.channels.inbox import valid_key
from frontdesk.channels.instagram import TokenRefreshError
from frontdesk.channels.runner import Outcome
from frontdesk.channels.wiring import Worker
from frontdesk.observability.logs import log_alert, log_event
from frontdesk.secrets import SecretUnavailableError

logger = logging.getLogger(__name__)


def worker_router(worker: Worker) -> APIRouter:
    router = APIRouter(prefix="/tasks")

    @router.post("/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> Response:
        if not valid_key(conversation_id):
            return Response(status_code=404)
        outcome = await worker.runner.process(conversation_id)
        return Response(status_code=503 if outcome is Outcome.BUSY else 200)

    @router.post("/sweep")
    async def sweep() -> JSONResponse:
        swept = await run_in_threadpool(worker.inbox.sweep)
        for key in swept.work:
            await worker.dispatcher.dispatch(key, delay=timedelta(0))
        return JSONResponse({"work": len(swept.work), "uncertain": swept.uncertain})

    @router.post("/purge")
    async def purge() -> JSONResponse:
        purged = await run_in_threadpool(worker.inbox.purge)
        holds = await run_in_threadpool(worker.purge_bookings)
        return JSONResponse(
            {"messages": purged.messages, "conversations": purged.conversations, "holds": holds}
        )

    @router.post("/refresh-token")
    async def refresh_token() -> Response:
        try:
            await worker.refresh()
        except (TokenRefreshError, SecretUnavailableError) as error:
            log_alert(logger, "channel.token_refresh_failed", reason=type(error).__name__)
            return Response(status_code=502)
        log_event(logger, "channel.token_refreshed")
        return Response(status_code=200)

    return router
