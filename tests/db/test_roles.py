"""The application's role on a new cloud database (M4): created from the deployment's URL, bound by
row-level security, its password updated by a second run, and refused when it could escape RLS."""

import uuid

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.exc import OperationalError

from frontdesk.db.roles import RoleError, ensure_app_role

pytestmark = pytest.mark.db


def _url(base: str, role: str, password: str) -> str:
    url = make_url(base).set(username=role, password=password)
    return url.render_as_string(hide_password=False)


def _role() -> str:
    return f"t_app_{uuid.uuid4().hex[:8]}"


def test_the_role_is_created_and_bound_by_rls(migrated_url: str) -> None:
    role = _role()
    app_url = _url(migrated_url, role, "first-pass")
    assert ensure_app_role(migrated_url, app_url) == role
    engine = create_engine(app_url)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT current_user")) == role
        attributes = "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        assert connection.scalar(text(attributes)) is False
    engine.dispose()


def test_a_second_run_sets_the_new_password(migrated_url: str) -> None:
    role = _role()
    ensure_app_role(migrated_url, _url(migrated_url, role, "first-pass"))
    ensure_app_role(migrated_url, _url(migrated_url, role, "second-pass"))
    fresh = create_engine(_url(migrated_url, role, "second-pass"))
    with fresh.connect():
        pass
    fresh.dispose()
    stale = create_engine(_url(migrated_url, role, "first-pass"))
    with pytest.raises(OperationalError):
        stale.connect()
    stale.dispose()


def test_a_role_that_could_escape_rls_is_refused(migrated_url: str) -> None:
    role = _role()
    owner = create_engine(migrated_url, isolation_level="AUTOCOMMIT")
    with owner.connect() as connection:
        connection.execute(text(f"CREATE ROLE {role} LOGIN BYPASSRLS"))
    owner.dispose()
    with pytest.raises(RoleError, match="escape row-level security"):
        ensure_app_role(migrated_url, _url(migrated_url, role, "pass"))


@pytest.mark.parametrize(
    "app_url",
    [
        "postgresql+psycopg://Bad-Role:x@localhost/db",
        "postgresql+psycopg://app@localhost/db",
    ],
)
def test_a_url_without_a_plain_role_and_a_password_is_refused(app_url: str) -> None:
    with pytest.raises(RoleError, match="lower-case role"):
        ensure_app_role("postgresql+psycopg://owner:x@localhost:1/db", app_url)
