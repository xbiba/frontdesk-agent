"""`frontdesk web serve`: the site's chat on this computer, to try the widget (design of 2026-11-07).

One process serves the site's routes, a test page that embeds the widget, and the worker's runs,
handed over in memory instead of through Cloud Tasks. It runs on the offline model, or with
`--live` on the deployment's live route, paid from `--budget` as `chat --live` is (the same checks
before the first request, the budget held while it runs, a cost line after each reply). It
answers everyone: the messages are test messages, never a patient's. The database is the env
file's (the local Docker Postgres), migrated first; the business needs active knowledge.
"""

from __future__ import annotations

import asyncio
import html
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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
from frontdesk.cli.live_chat import LiveCosts, live_banner, live_route, model_errors_on_stderr
from frontdesk.cli.live_common import (
    check_access,
    refuse_environment_overrides,
    route_api_key,
    spent_usd,
)
from frontdesk.cli.runtime import fail, without_knowledge
from frontdesk.composition import ModelWrapper, NoActiveKnowledgeError, open_deployment
from frontdesk.db.migrate import upgrade_to_head
from frontdesk.db.roles import revoke_migration_record
from frontdesk.model_gateway.costs import BudgetError, load_budget
from frontdesk.model_gateway.metering import Meter
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
<p>The chat button is at the bottom right. {model}, test messages only: never a patient's
data. Languages: {links}.</p>
<script src="/web/widget.js" data-lang="{lang}" defer></script>
</body>
</html>
"""


def local_app(
    settings: Settings,
    *,
    wrap_model: ModelWrapper | None = None,
    after_turn: Callable[[], None] | None = None,
) -> FastAPI:
    """The site's routes, the worker in this process and the test page at `/`. A live route needs
    `wrap_model`, the meter of `--live`; `after_turn` runs after each run of the worker."""
    route = require_route(settings.model_route, DataClass.EVALUATION)
    if route.provider != "offline" and wrap_model is None:
        raise UnapprovedRouteError(
            f"web serve runs the offline model unless --live, not '{route.route_id}': "
            "`frontdesk web serve --live --budget <file>` pays for it"
        )
    everyone = settings.model_copy(update={"web_mode": InstagramMode.LIVE})
    dispatcher = InProcessDispatcher()
    web = build_web_ingress(everyone, dispatcher=dispatcher, data_class=DataClass.EVALUATION)
    if web is None:
        raise ValueError("the pack has no web section (channels.web)")
    deployment = open_deployment(everyone)
    # A key of this process for the repeats' digests: nothing here outlives the try.
    runner, inbox = build_web_worker(
        everyone,
        deployment,
        data_class=DataClass.EVALUATION,
        key=secrets.token_bytes(32),
        wrap_model=wrap_model,
    )

    async def work(key: str) -> None:
        """What the worker's route does after a run (frontdesk.api.worker)."""
        outcome = await runner.process(key)
        if after_turn is not None:
            after_turn()
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
        model = "Offline model" if wrap_model is None else "Live model, paid from the budget"
        page = PAGE.format(lang=chosen, name=html.escape(pack.name), links=links, model=model)
        return HTMLResponse(page)

    return app


def serve(env_file: Path, port: int, *, live: bool = False, budget_path: Path | None = None) -> int:
    if budget_path is not None and not live:
        raise fail("--budget only goes with --live")
    settings = load_settings(env_file)
    if live:
        return _serve_live(env_file, settings, port, budget_path)
    _migrate(settings)
    app = _app(env_file, settings)
    _run(app, settings, port, "offline model")
    return 0


def _serve_live(env_file: Path, settings: Settings, port: int, budget_path: Path | None) -> int:
    """`web serve --live`: the checks of `chat --live` before the first request, then the server,
    which holds the budget until it stops. A reply after the budget ran out is the pack's text for
    a turn that failed."""
    if budget_path is None:
        raise fail("--live needs --budget: every live request is counted against it")
    route = live_route(settings.model_route, env_file)
    refuse_environment_overrides()
    _migrate(settings)
    try:
        budget = load_budget(budget_path)
        run_id = f"web-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        meter = Meter(budget, route_id=route.route_id, run_id=run_id)
        with budget.exclusive():
            costs = LiveCosts.begin(meter, budget)
            app = _app(
                env_file,
                settings,
                wrap_model=meter.wrap,
                after_turn=lambda: typer.echo(costs.cost_line()),
            )
            check_access(route, settings.gcp_project, route_api_key(settings, route))
            meter.check_ledger()  # the last gate before the first paid request
            typer.echo(live_banner(route, budget, spent_usd=spent_usd(budget)))
            with model_errors_on_stderr():
                _run(app, settings, port, f"{route.model_name}, paid")
    except BudgetError as error:
        raise fail(str(error)) from error
    return 0


def _migrate(settings: Settings) -> None:
    upgrade_to_head(settings.migration_database_url)
    revoke_migration_record(settings.migration_database_url, settings.database_url)


def _app(
    env_file: Path,
    settings: Settings,
    *,
    wrap_model: ModelWrapper | None = None,
    after_turn: Callable[[], None] | None = None,
) -> FastAPI:
    try:
        return local_app(settings, wrap_model=wrap_model, after_turn=after_turn)
    except NoActiveKnowledgeError as error:
        raise without_knowledge(error, env_file) from None
    except (UnapprovedRouteError, ValueError) as error:
        raise fail(str(error)) from None


def _run(app: FastAPI, settings: Settings, port: int, model: str) -> None:
    origin = f"http://localhost:{port}"
    pack_web = open_deployment(settings).pack.channels.web
    if pack_web is not None and not pack_web.allows(origin):
        typer.echo(f"warning: {origin} is not in the pack's web origins: the widget is refused")
    typer.echo(f"Open {origin}/ ({model}, test messages only). Ctrl-C stops.")
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
