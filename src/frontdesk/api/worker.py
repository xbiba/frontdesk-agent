"""The worker's routes (ADR-0011): Cloud Tasks calls one for each conversation, Cloud Scheduler the
others. Cloud Run lets in only the invoker's identity, so these routes check nothing themselves."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Final

import httpx2
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from frontdesk.channels.dispatch import DispatchError
from frontdesk.channels.inbox import BURST, valid_key
from frontdesk.channels.instagram import INSTAGRAM, TokenRefreshError
from frontdesk.channels.runner import Outcome
from frontdesk.channels.wiring import Worker
from frontdesk.observability.logs import log_alert, log_event
from frontdesk.secrets import SecretUnavailableError

logger = logging.getLogger(__name__)
# The outcomes after which a message may still wait: younger than the burst, or come during a turn.
_GIVEN_BACK: Final = frozenset({Outcome.WAIT, Outcome.DONE})


def worker_router(worker: Worker) -> APIRouter:
    router = APIRouter(prefix="/tasks")

    @router.post("/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> Response:
        channel = conversation_id.split("-", 1)[0]  # the key names its channel
        runner = worker.runners.get(channel)
        if not valid_key(conversation_id) or runner is None:
            return Response(status_code=404)
        outcome = await runner.process(conversation_id)
        if outcome is Outcome.BUSY:
            return Response(status_code=503)
        if outcome in _GIVEN_BACK and await run_in_threadpool(
            worker.inbox[channel].waiting, conversation_id
        ):
            # A message younger than the burst, or one that came during the turn, gets its task
            # now, not at the sweep (final review of M4, M2). Should that fail, the sweep still does.
            try:
                await worker.dispatcher.dispatch(conversation_id, delay=BURST)
            except DispatchError:
                log_event(logger, "channel.dispatch_failed", conversation_id=conversation_id)
        if channel == INSTAGRAM and worker.requests is not None:
            # A patient's turn may have left a request for the clinic (W1). A staff thread whose
            # task fails here gets it from the sweep.
            try:
                await worker.requests.notify()
            except DispatchError:
                log_event(logger, "requests.dispatch_failed")
        return Response(status_code=200)

    @router.post("/sweep")
    async def sweep() -> JSONResponse:
        work, uncertain = 0, 0
        for inbox in worker.inbox.values():
            swept = await run_in_threadpool(inbox.sweep)
            for key in swept.work:
                await worker.dispatcher.dispatch(key, delay=timedelta(0))
            work, uncertain = work + len(swept.work), uncertain + swept.uncertain
        return JSONResponse({"work": work, "uncertain": uncertain})

    @router.post("/purge")
    async def purge() -> JSONResponse:
        messages, conversations = 0, 0
        for inbox in worker.inbox.values():
            purged = await run_in_threadpool(inbox.purge)
            messages, conversations = (
                messages + purged.messages,
                conversations + purged.conversations,
            )
        holds = await run_in_threadpool(worker.purge_bookings)
        return JSONResponse({"messages": messages, "conversations": conversations, "holds": holds})

    @router.post("/refresh-token")
    async def refresh_token() -> Response:
        try:
            await worker.refresh()
        except (TokenRefreshError, SecretUnavailableError, httpx2.HTTPError) as error:
            log_alert(logger, "channel.token_refresh_failed", reason=type(error).__name__)
            return Response(status_code=502)
        log_event(logger, "channel.token_refreshed")
        return Response(status_code=200)

    return router
