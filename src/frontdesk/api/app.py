"""HTTP entrypoints. `create_app` serves /health. M4 adds a channel's two services, one image
(ADR-0011): the public webhook (`create_webhook_app`) and the private worker (`create_worker_app`)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from frontdesk.db.engine import make_engine
from frontdesk.observability.logs import configure_logging, log_event
from frontdesk.settings import Settings

if TYPE_CHECKING:
    from frontdesk.channels.wiring import Ingress, Worker

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, engine: Engine | None = None) -> FastAPI:
    resolved = settings or Settings()
    database = engine or make_engine(resolved.database_url)
    app = FastAPI(title="frontdesk", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> JSONResponse:
        try:
            with database.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            log_event(logger, "health.database_unavailable")
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok"})

    return app


def create_webhook_app(
    settings: Settings | None = None,
    *,
    ingress: Ingress | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """The public service that receives a channel's webhooks (M4). Built from the environment when
    uvicorn calls it with no argument."""
    from frontdesk.api.webhook import webhook_router
    from frontdesk.channels.wiring import build_ingress

    if settings is None:
        configure_logging(Settings().log_level)
    resolved = settings or Settings()
    app = create_app(resolved, engine=engine)
    app.include_router(webhook_router(ingress or build_ingress(resolved)))
    return app


def create_worker_app(
    settings: Settings | None = None,
    *,
    worker: Worker | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    """The private service Cloud Tasks and Cloud Scheduler call (M4). Built from the environment
    when uvicorn calls it with no argument."""
    from frontdesk.api.worker import worker_router
    from frontdesk.channels.wiring import build_worker

    if settings is None:
        configure_logging(Settings().log_level)
    resolved = settings or Settings()
    app = create_app(resolved, engine=engine)
    app.include_router(worker_router(worker or build_worker(resolved)))
    return app
