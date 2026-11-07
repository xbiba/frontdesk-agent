"""`frontdesk web serve`: the site's chat on this computer, to try the widget (design of 2026-11-07).

One process serves the site's routes, a test page that embeds the widget, and the worker's runs,
handed over in memory instead of through Cloud Tasks. It runs on the offline model only and
answers everyone: the messages are test messages, never a patient's. The database is the env
file's (the local Docker Postgres), migrated first; the business needs active knowledge.
"""

from __future__ import annotations

import asyncio
import html
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Final

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from frontdesk.api.app import create_app
from frontdesk.api.web import web_router
from frontdesk.channels.dispatch import InProcessDispatcher
from frontdesk.channels.inbox import BURST
from frontdesk.channels.runner import Outcome
from frontdesk.channels.wiring import build_web_ingress, build_web_worker
from frontdesk.cli.runtime import fail, without_knowledge
from frontdesk.composition import NoActiveKnowledgeError, open_deployment
from frontdesk.db.migrate import upgrade_to_head
from frontdesk.db.roles import revoke_migration_record
from frontdesk.model_gateway.routes import DataClass, UnapprovedRouteError, require_route
from frontdesk.settings import InstagramMode, Settings, load_settings

HOST: Final = "127.0.0.1"
BUSY_AGAIN: Final = timedelta(seconds=1)
PAGE: Final = """<!doctype html>
<html lang="{lang}">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test: {name}</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem">
<h1>{name}: test page</h1>
<p>The chat button is at the bottom right. Offline model, test messages only: never a patient's
data. Languages: {links}.</p>
<script src="/web/widget.js" data-lang="{lang}" defer></script>
</body>
</html>
"""


def local_app(settings: Settings) -> FastAPI:
    """The site's routes, the worker in this process and the test page at `/`."""
    route = require_route(settings.model_route, DataClass.EVALUATION)
    if route.provider != "offline":
        raise UnapprovedRouteError(
            f"web serve runs the offline model only, not '{route.route_id}': a live try goes "
            "through tester mode on Cloud Run, or `frontdesk chat --live`"
        )
    everyone = settings.model_copy(update={"web_mode": InstagramMode.LIVE})
    dispatcher = InProcessDispatcher()
    web = build_web_ingress(everyone, dispatcher=dispatcher)
    if web is None:
        raise ValueError("the pack has no web section (channels.web)")
    deployment = open_deployment(everyone)
    # A key of this process for the repeats' digests: nothing here outlives the try.
    runner, inbox = build_web_worker(
        everyone, deployment, data_class=DataClass.EVALUATION, key=secrets.token_bytes(32)
    )

    async def work(key: str) -> None:
        """What the worker's route does after a run (frontdesk.api.worker)."""
        outcome = await runner.process(key)
        if outcome is Outcome.BUSY:
            await dispatcher.dispatch(key, delay=BUSY_AGAIN)
        elif outcome in (Outcome.WAIT, Outcome.DONE) and await asyncio.to_thread(
            inbox.waiting, key
        ):
            await dispatcher.dispatch(key, delay=BURST)

    dispatcher.bind(work)
    app = create_app(everyone, public=True)
    app.include_router(web_router(web))
    pack = web.pack

    @app.get("/")
    def page(lang: str | None = None) -> HTMLResponse:
        chosen = lang if lang in pack.languages else pack.default_language
        links = ", ".join(f'<a href="/?lang={code}">{code}</a>' for code in pack.languages)
        return HTMLResponse(PAGE.format(lang=chosen, name=html.escape(pack.name), links=links))

    return app


def serve(env_file: Path, port: int) -> int:
    settings = load_settings(env_file)
    upgrade_to_head(settings.migration_database_url)
    revoke_migration_record(settings.migration_database_url, settings.database_url)
    try:
        app = local_app(settings)
    except NoActiveKnowledgeError as error:
        raise without_knowledge(error, env_file) from None
    except (UnapprovedRouteError, ValueError) as error:
        raise fail(str(error)) from None
    origin = f"http://localhost:{port}"
    pack_web = open_deployment(settings).pack.channels.web
    if pack_web is not None and not pack_web.allows(origin):
        typer.echo(f"warning: {origin} is not in the pack's web origins: the widget is refused")
    typer.echo(f"Open {origin}/ (offline model, test messages only). Ctrl-C stops.")
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
    return 0
