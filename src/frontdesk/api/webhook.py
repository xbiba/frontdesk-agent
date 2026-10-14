"""The webhook of a channel (M4 design §4 steps 1-3): Meta's signature on the raw body, then the
inbox, then a task for each conversation with work. Without the signature nothing is stored."""

from __future__ import annotations

import hmac
import logging
from typing import Final

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from frontdesk.channels.dispatch import DispatchError
from frontdesk.channels.inbox import BURST
from frontdesk.channels.wiring import Ingress
from frontdesk.observability.logs import log_alert, log_event

logger = logging.getLogger(__name__)
MAX_BODY_BYTES: Final = 1_000_000


async def read_limited(request: Request) -> bytes | None:
    """The body, or None once it passes MAX_BODY_BYTES: read in chunks, so a body without a length,
    or with a false one, is never held whole before the check."""
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
        return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            return None
    return bytes(body)


def webhook_router(ingress: Ingress) -> APIRouter:
    router = APIRouter()
    channel = ingress.webhook.name
    path = f"/webhooks/{channel}"

    @router.get(path)
    def verify(request: Request) -> Response:
        """Meta's subscription check: the challenge comes back only with the verify token."""
        params = request.query_params
        given = params.get("hub.verify_token", "").encode()
        if params.get("hub.mode") == "subscribe" and hmac.compare_digest(
            given, ingress.verify_token.encode()
        ):
            return PlainTextResponse(params.get("hub.challenge", ""))
        return Response(status_code=403)

    @router.post(path)
    async def receive(request: Request) -> Response:
        body = await read_limited(request)
        if body is None:
            return Response(status_code=413)
        if not ingress.webhook.signature_ok(body, request.headers.get("x-hub-signature-256")):
            log_event(logger, "channel.bad_signature", channel=channel)
            return Response(status_code=403)
        if ingress.off:
            return Response(status_code=200)
        try:
            events = ingress.webhook.parse(body)
        except ValueError:
            log_alert(logger, "channel.unreadable_delivery", channel=channel)
            return Response(status_code=200)  # Meta signed it: a retry would bring the same body
        accepted = await run_in_threadpool(ingress.inbox.accept, events)
        try:
            for key in accepted.work:
                await ingress.dispatcher.dispatch(key, delay=BURST)
        except DispatchError:
            log_alert(logger, "channel.dispatch_failed", channel=channel)
            return Response(status_code=500)  # Meta delivers again; the messages wait stored
        return Response(status_code=200)

    return router
