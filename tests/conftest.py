from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import create_engine, make_url, text
from testcontainers.community.postgres import PostgresContainer

from frontdesk.db.migrate import upgrade_to_head

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
APP_ROLE = "frontdesk_app"
APP_PASSWORD = "frontdesk_app"


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as container:
        yield container


@pytest.fixture(scope="session")
def app_role(postgres: PostgresContainer) -> str:
    engine = create_engine(postgres.get_connection_url(), isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE ROLE {APP_ROLE} LOGIN PASSWORD '{APP_PASSWORD}' "
                "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS"
            )
        )
    engine.dispose()
    return APP_ROLE


@pytest.fixture
def new_database(
    postgres: PostgresContainer, app_role: str
) -> Iterator[Callable[[], tuple[str, str]]]:
    """Create fresh migrated databases; returns (owner_url, app_url) per call."""
    created: list[str] = []
    admin = create_engine(postgres.get_connection_url(), isolation_level="AUTOCOMMIT")

    def create() -> tuple[str, str]:
        name = f"t_{uuid.uuid4().hex[:12]}"
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
        created.append(name)
        owner = make_url(postgres.get_connection_url()).set(database=name)
        owner_url = owner.render_as_string(hide_password=False)
        upgrade_to_head(owner_url)
        app_url = owner.set(username=app_role, password=APP_PASSWORD).render_as_string(
            hide_password=False
        )
        return owner_url, app_url

    yield create
    with admin.connect() as connection:
        for name in created:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def database(new_database: Callable[[], tuple[str, str]]) -> tuple[str, str]:
    return new_database()


@pytest.fixture
def migrated_url(database: tuple[str, str]) -> str:
    return database[0]


@pytest.fixture
def app_url(database: tuple[str, str]) -> str:
    return database[1]
