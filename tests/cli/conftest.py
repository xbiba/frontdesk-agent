"""What every test module of this directory shares: the fake Gemini of the live tests, a clean
environment, and a stand-in for the access check that never reaches the network."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest
from sqlalchemy import create_engine, make_url, text
from testcontainers.community.postgres import PostgresContainer

from frontdesk.model_gateway.access import ENVIRONMENT_OVERRIDES
from frontdesk.model_gateway.routes import ModelRoute
from tests.cli.live_chat_helpers import Failure, Install, fake_gemini


@pytest.fixture(autouse=True)
def _no_proxy_or_certificate_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live command refuses to start while one is set, and a developer's shell often has one:
    a test that needs one sets it itself."""
    for name in [name for name in os.environ if name.upper() in ENVIRONMENT_OVERRIDES]:
        monkeypatch.delenv(name)


class AccessChecks:
    """Stands for `check_route_access`: it records each call, calls `on_call` (a test looks at the
    world at that moment), and raises `error` when a test sets one. It sends nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None
        self.on_call: Callable[[], None] | None = None

    async def __call__(self, route: ModelRoute, *, gcp_project: str, client: Any = None) -> None:
        self.calls.append((route.route_id, gcp_project))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error


@pytest.fixture(autouse=True)
def access_checks(monkeypatch: pytest.MonkeyPatch) -> AccessChecks:
    """No test of the commands reaches Google: the free request that proves credentials, API and
    model is replaced by this record. The check itself is tested in `tests/model_gateway`."""
    checks = AccessChecks()
    monkeypatch.setattr("frontdesk.cli.live_common.check_route_access", checks)
    return checks


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Install:
    """Puts a fake Gemini where `build_live_model` would build the real one, so that it takes the
    same `wrap_model` path. Returns the roles of the requests it is sent; `failing` makes a role
    raise (see `fake_gemini`)."""

    def install(
        classification: dict[str, Any], *, failing: Mapping[str, Failure] | None = None
    ) -> list[str]:
        sent: list[str] = []
        model = fake_gemini(classification, sent, failing)
        monkeypatch.setattr(
            "frontdesk.composition.build_live_model", lambda route, *, gcp_project: model
        )
        return sent

    return install


@pytest.fixture
def new_server_database(postgres: PostgresContainer) -> Iterator[Callable[[str], str]]:
    """Creates empty databases on the test server, named with the suffix asked for, and drops them
    after the test. Returns the URL of each, as its owner."""
    admin = create_engine(postgres.get_connection_url(), isolation_level="AUTOCOMMIT")
    created: list[str] = []

    def create(suffix: str) -> str:
        name = f"t_{uuid.uuid4().hex[:8]}{suffix}"
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
        created.append(name)
        url = make_url(postgres.get_connection_url()).set(database=name)
        return url.render_as_string(hide_password=False)

    yield create
    with admin.connect() as connection:
        for name in created:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def scratch_url(new_server_database: Callable[[str], str]) -> str:
    """A scratch database for the embedder experiment: its name ends in `_experiment`."""
    return new_server_database("_experiment")
