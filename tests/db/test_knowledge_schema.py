import uuid

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import KnowledgeVersion

pytestmark = pytest.mark.db

BUSINESS_TABLES = {
    "knowledge_versions",
    "documents",
    "chunks",
    "records",
    "order_lookup_attempts",
}


def _insert_version(owner_url: str, business_id: str, status: str = "draft") -> None:
    engine = create_engine(owner_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO knowledge_versions (id, business_id, status, source_digest) "
                "VALUES (:id, :business_id, :status, 'digest')"
            ),
            {"id": uuid.uuid4(), "business_id": business_id, "status": status},
        )
    engine.dispose()


def test_app_role_sees_only_its_business(database: tuple[str, str]) -> None:
    owner_url, app_url = database
    _insert_version(owner_url, "alpha")
    _insert_version(owner_url, "beta")
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, "alpha") as session:
        assert set(session.scalars(select(KnowledgeVersion.business_id))) == {"alpha"}


def test_without_business_scope_nothing_is_visible(database: tuple[str, str]) -> None:
    owner_url, app_url = database
    _insert_version(owner_url, "alpha")
    factory = make_session_factory(make_engine(app_url))
    with factory() as session:
        assert session.scalars(select(KnowledgeVersion)).all() == []


def _insert_as(factory: sessionmaker[Session], scope: str, business_id: str) -> None:
    with business_session(factory, scope) as session:
        session.add(
            KnowledgeVersion(business_id=business_id, status="draft", source_digest="digest")
        )
        session.flush()


def test_app_role_cannot_write_for_another_business(database: tuple[str, str]) -> None:
    _, app_url = database
    factory = make_session_factory(make_engine(app_url))
    with pytest.raises(DBAPIError, match="row-level security"):
        _insert_as(factory, "alpha", "beta")


def test_one_active_version_per_business(database: tuple[str, str]) -> None:
    owner_url, _ = database
    _insert_version(owner_url, "alpha", "active")
    with pytest.raises(IntegrityError):
        _insert_version(owner_url, "alpha", "active")


def test_status_is_constrained(database: tuple[str, str]) -> None:
    owner_url, _ = database
    with pytest.raises(IntegrityError):
        _insert_version(owner_url, "alpha", "published")


def test_every_business_table_has_row_level_security_enabled_and_forced(
    migrated_url: str,
) -> None:
    # Migration 0001 grants the app role DML on every future table, so a business table created
    # without RLS would be silently open. This test fails for the table that forgot it.
    engine = create_engine(migrated_url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') "
                "AND c.relname <> 'alembic_version'"
            )
        ).all()
    engine.dispose()
    protected = {name: enabled and forced for name, enabled, forced in rows}
    assert protected.keys() >= BUSINESS_TABLES, "the check must see every business table"
    assert [name for name, ok in sorted(protected.items()) if not ok] == []


def test_every_policy_fails_closed_when_the_scope_is_empty(migrated_url: str) -> None:
    # A policy that compares with the bare setting accepts '' (what a reused pooled connection
    # reads after a scoped transaction). The scope must go through NULLIF, in USING and WITH CHECK.
    engine = create_engine(migrated_url)
    with engine.connect() as connection:
        policies = connection.execute(
            text("SELECT tablename, qual, with_check FROM pg_policies WHERE schemaname = 'public'")
        ).all()
    engine.dispose()
    fails_closed = "nullif(current_setting('app.business_id'"
    assert {table for table, _, _ in policies} >= BUSINESS_TABLES
    weak = [
        table
        for table, using, check in policies
        if fails_closed not in using.lower() or fails_closed not in check.lower()
    ]
    assert weak == []
