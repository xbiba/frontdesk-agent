"""The site's routes (design of 2026-11-07 §2-§3), public on the webhook service.

- POST /web/messages stores the widget's message in the inbox and gives its conversation a task.
- GET /web/replies gives the session's replies after a cursor, and whether more are coming.
- GET /web/config gives the widget its name, words, the reception's numbers and the client's
  style, if its pack has one (else the widget keeps its plain, neutral look).
- GET /web/widget.js is the widget, the same file for every business.

Only the sites the pack lists may call them from a browser: CORS answers those, and a request with
any other Origin is refused. A message needs a listed Origin. Each address may post 20 messages in
10 minutes, counted in memory; the engine's limits hold behind it.
"""

from __future__ import annotations

import json
import logging
import re
from importlib.resources import files
from typing import Any, Final

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from frontdesk.api.ratelimit import SlidingWindow, client_address
from frontdesk.api.webhook import read_limited
from frontdesk.channels.dispatch import DispatchError
from frontdesk.channels.inbox import BURST
from frontdesk.channels.web import MAX_TEXT_CHARS, inbound, valid_message_id, valid_session
from frontdesk.channels.wiring import WebIngress
from frontdesk.observability.logs import log_alert, log_event
from frontdesk.tenancy.pack import FixedTextKey

logger = logging.getLogger(__name__)
MAX_WEB_BODY_BYTES: Final = 16_384  # 2000 characters of text, escaped, and the IDs
MESSAGES_PER_ADDRESS: Final = 20
ADDRESS_WINDOW_SECONDS: Final = 600.0
CONFIG_MAX_AGE: Final = 300
_CURSOR: Final = re.compile(r"^\d{1,20}$")
_NOT_DIGITS: Final = re.compile(r"\D")


def widget_script() -> str:
    return files("frontdesk.api").joinpath("widget.js").read_text(encoding="utf-8")


def _error(status: int, code: str, headers: dict[str, str]) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status, headers=headers)


def _message_fields(body: bytes) -> tuple[str, str, str] | None:
    """The session, the message ID and the text of a well-formed message, or None."""
    try:
        data: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    session, message_id, text = data.get("session"), data.get("id"), data.get("text")
    if not (isinstance(session, str) and isinstance(message_id, str) and isinstance(text, str)):
        return None
    if not (valid_session(session) and valid_message_id(message_id) and text.strip()):
        return None
    return session, message_id, text.strip()


def web_config(web: WebIngress) -> dict[str, Any]:
    """What the widget shows. The numbers come from the pack's contacts, for tel: and wa.me."""
    pack, channel = web.pack, web.channel
    location = pack.location
    whatsapp = _NOT_DIGITS.sub("", location.whatsapp) if location.whatsapp else None
    return {
        "name": channel.name,
        "style": channel.style.model_dump() if channel.style else None,
        "default_language": pack.default_language,
        "languages": list(pack.languages),
        "texts": {lang: texts.model_dump() for lang, texts in channel.texts.items()},
        # What the widget says when no reply came in time; its header links to a person.
        "unavailable": {
            lang: pack.fixed_texts[lang][FixedTextKey.UNAVAILABLE] for lang in pack.languages
        },
        "phone": "+" + _NOT_DIGITS.sub("", location.phone),
        "whatsapp": whatsapp,
        "privacy_url": channel.privacy_url,
    }


def web_router(web: WebIngress, *, limiter: SlidingWindow | None = None) -> APIRouter:
    router = APIRouter(prefix="/web")
    per_address = limiter or SlidingWindow(MESSAGES_PER_ADDRESS, ADDRESS_WINDOW_SECONDS)
    config = web_config(web)
    script = widget_script()

    def cors(request: Request) -> dict[str, str] | None:
        """The CORS headers for a listed Origin, none without one; None for any other Origin."""
        origin = request.headers.get("origin")
        if origin is None:
            return {"Vary": "Origin"}
        if not web.channel.allows(origin):
            return None
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}

    @router.options("/{path:path}")
    def preflight(request: Request) -> Response:
        headers = cors(request)
        if headers is None or request.headers.get("origin") is None:
            return Response(status_code=403)
        headers |= {
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "600",
        }
        return Response(status_code=204, headers=headers)

    @router.post("/messages")
    async def message(request: Request) -> Response:
        headers = cors(request)
        if headers is None or request.headers.get("origin") is None:
            log_event(logger, "web.bad_origin")
            return Response(status_code=403)
        body = await read_limited(request, MAX_WEB_BODY_BYTES)
        if body is None:
            return _error(413, "too_large", headers)
        fields = _message_fields(body)
        if fields is None:
            return _error(422, "invalid", headers)
        session, message_id, text = fields
        if len(text) > MAX_TEXT_CHARS:
            return _error(413, "too_long", headers)
        if not per_address.allow(client_address(request)):
            log_event(logger, "web.address_limit")
            return _error(429, "rate", headers | {"Retry-After": "60"})
        if web.off:
            return _error(503, "off", headers)
        event = inbound(web.pack.id, session, message_id, text)
        accepted = await run_in_threadpool(web.inbox.accept, [event])
        try:
            for key in accepted.work:
                await web.dispatcher.dispatch(key, delay=BURST)
        except DispatchError:
            # The message waits stored: the widget's retry, or the sweep, gives it its task.
            log_alert(logger, "channel.dispatch_failed", channel="web")
            return _error(503, "unavailable", headers)
        return JSONResponse({"ok": True}, status_code=202, headers=headers)

    @router.get("/replies")
    async def replies(request: Request) -> Response:
        headers = cors(request)
        if headers is None:
            return Response(status_code=403)
        session = request.query_params.get("session", "")
        after = request.query_params.get("after", "0")
        if not valid_session(session) or _CURSOR.fullmatch(after) is None:
            return _error(422, "invalid", headers)
        found = await run_in_threadpool(web.replies.after, session, int(after))
        payload = {
            "replies": [{"cursor": str(r.cursor), "text": r.text} for r in found.replies],
            "waiting": found.waiting,
        }
        return JSONResponse(payload, headers=headers | {"Cache-Control": "no-store"})

    @router.get("/config")
    def configuration(request: Request) -> Response:
        headers = cors(request)
        if headers is None:
            return Response(status_code=403)
        cache = {"Cache-Control": f"public, max-age={CONFIG_MAX_AGE}"}
        return JSONResponse(config, headers=headers | cache)

    @router.get("/widget.js")
    def widget() -> Response:
        headers = {
            "Cache-Control": f"public, max-age={CONFIG_MAX_AGE}",
            "X-Content-Type-Options": "nosniff",
        }
        return Response(script, media_type="text/javascript; charset=utf-8", headers=headers)

    return router
