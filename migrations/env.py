"""Alembic environment: online migrations; URL from the config or the settings."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from frontdesk.settings import Settings

config = context.config


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or Settings().migration_database_url


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
