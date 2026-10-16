"""The application's role on a new database (M4): it logs in with the password of the deployment's
URL, and row-level security binds it. Locally the database's init script creates it; on Cloud SQL,
`frontdesk db app-role` does, as the owner, before the first migration."""

from __future__ import annotations

import re
from typing import Final

from sqlalchemy import create_engine, make_url, text

_NAME: Final = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
# Escapes RLS, or can make a role that does: refused, never altered.
_POWERFUL: Final = text(
    "SELECT rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb "
    "FROM pg_roles WHERE rolname = :role"
)
_REMEMBER: Final = text(
    "SELECT set_config('frontdesk.role', :role, true), "
    "set_config('frontdesk.secret', :secret, true)"
)
# The name and the password reach the statement as settings, quoted by format(): no SQL is built
# from them in Python.
_CREATE: Final = text(
    "DO $$ BEGIN EXECUTE format('CREATE ROLE %I WITH LOGIN PASSWORD %L', "
    "current_setting('frontdesk.role'), current_setting('frontdesk.secret')); END $$"
)
_ALTER: Final = text(
    "DO $$ BEGIN EXECUTE format('ALTER ROLE %I WITH LOGIN PASSWORD %L', "
    "current_setting('frontdesk.role'), current_setting('frontdesk.secret')); END $$"
)


# The tables it reads and writes, now and after later migrations by the owner, but never the
# migrations' own record (final review of M4, M12: 0001's default privileges reach only a role
# named frontdesk_app that existed when it ran).
_GRANT: Final = text(
    "DO $$ DECLARE app text := current_setting('frontdesk.role'); BEGIN "
    "EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', app); "
    "EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %I', "
    "app); "
    "EXECUTE format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', app); "
    "EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, "
    "DELETE ON TABLES TO %I', app); "
    "EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES "
    "TO %I', app); "
    "IF to_regclass('public.alembic_version') IS NOT NULL THEN "
    "EXECUTE format('REVOKE ALL ON public.alembic_version FROM %I', app); END IF; END $$"
)


class RoleError(ValueError):
    """The application's role cannot be what row-level security needs."""


def ensure_app_role(owner_url: str, app_url: str) -> str:
    """Create the role of `app_url`, or set its password, and grant it the tables; return its
    name."""
    url = make_url(app_url)
    role, secret = url.username, url.password
    if role is None or not _NAME.match(role) or not secret:
        raise RoleError("the database URL must name a lower-case role and its password")
    engine = create_engine(owner_url)
    try:
        with engine.begin() as connection:
            powerful = connection.scalar(_POWERFUL, {"role": role})
            if powerful:
                raise RoleError(f"the role {role} can escape row-level security")
            connection.execute(_REMEMBER, {"role": role, "secret": str(secret)})
            connection.execute(_ALTER if powerful is not None else _CREATE)
            connection.execute(_GRANT)
    finally:
        engine.dispose()
    return role
