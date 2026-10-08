"""HTTP entrypoint. M0 exposes /health; channel webhooks arrive in M3."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from frontdesk.db.engine import make_engine
from frontdesk.observability.logs import log_event
from frontdesk.settings import Settings

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
