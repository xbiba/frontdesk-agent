"""Migration 0004 on a database that already holds data, in both directions.

Databases at 0003 exist (a developer's compose volume), so the hardening is a revision of its own
rather than an edit of 0002 and 0003: those would leave such a database with the weak policies.
"""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from frontdesk.db.migrate import alembic_config
from tests.db.test_business_isolation import BUSINESS_TABLES, owner_of

pytestmark = pytest.mark.db

ADDED = {
    "knowledge_versions": {
        "ck_knowledge_versions_business_id_not_blank",
        "uq_knowledge_versions_id_business",
    },
    "documents": {
        "ck_documents_business_id_not_blank",
        "uq_documents_id_version",
        "fk_documents_version_business",
    },
    "chunks": {
        "ck_chunks_business_id_not_blank",
        "uq_chunks_document_ordinal",
        "fk_chunks_version_business",
        "fk_chunks_document_version",
    },
    "records": {"ck_records_business_id_not_blank", "fk_records_version_business"},
    "order_lookup_attempts": {"ck_order_lookup_attempts_business_id_not_blank"},
}


def _names(url: str, query: str) -> set[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return {row[0] for row in connection.execute(text(query))}
    finally:
        engine.dispose()


def _constraints(url: str) -> dict[str, set[str]]:
    """Every constraint of every business table, by name."""
    return {
        table: _names(
            url, f"SELECT conname FROM pg_constraint WHERE conrelid = 'public.{table}'::regclass"
        )
        for table in BUSINESS_TABLES
    }


def _added_by_0004(before: dict[str, set[str]], after: dict[str, set[str]]) -> dict[str, set[str]]:
    assert all(before[table] <= after[table] for table in BUSINESS_TABLES), "nothing is removed"
    return {table: after[table] - before[table] for table in BUSINESS_TABLES}


def _policies(url: str) -> dict[str, tuple[str, str]]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT tablename, qual, with_check FROM pg_policies WHERE schemaname = 'public'"
                )
            ).all()
    finally:
        engine.dispose()
    return {table: (qual, check) for table, qual, check in rows}


def _fails_closed(expression: str) -> bool:
    return "nullif(current_setting('app.business_id'" in expression.lower()


def _holds_data(owner_url: str) -> list[int]:
    with owner_of(owner_url) as owner:
        return [owner.count(table) for table in BUSINESS_TABLES]


def _populate(owner_url: str) -> None:
    with owner_of(owner_url) as owner:
        version = owner.version("alpha", status="active")
        document = owner.document("alpha", version)
        owner.chunk("alpha", version, document, ordinal=0)
        owner.chunk("alpha", version, document, ordinal=1)
        owner.record("alpha", version)
        owner.run(
            "INSERT INTO order_lookup_attempts (business_id, conversation_id, used, updated_at) "
            "VALUES ('alpha', 'c1', 1, now())"
        )


def test_0004_hardens_a_populated_database_and_downgrades_back_to_0003(
    database: tuple[str, str],
) -> None:
    owner_url, _ = database
    config = alembic_config(owner_url)

    command.downgrade(config, "0003")
    at_0003 = _constraints(owner_url)
    old_policies = _policies(owner_url)
    assert not any(
        _fails_closed(qual) or _fails_closed(check) for qual, check in old_policies.values()
    )
    _populate(owner_url)
    assert _holds_data(owner_url) == [1, 1, 2, 1, 1]

    command.upgrade(config, "0004")
    assert _added_by_0004(at_0003, _constraints(owner_url)) == ADDED
    policies = _policies(owner_url)
    assert set(policies) == set(BUSINESS_TABLES)
    assert all(_fails_closed(qual) and _fails_closed(check) for qual, check in policies.values())
    assert _holds_data(owner_url) == [1, 1, 2, 1, 1]

    command.downgrade(config, "0003")
    assert _constraints(owner_url) == at_0003
    assert _policies(owner_url) == old_policies
    assert _holds_data(owner_url) == [1, 1, 2, 1, 1]

    command.upgrade(config, "head")  # and up again: the cycle leaves nothing behind
    assert _added_by_0004(at_0003, _constraints(owner_url)) == ADDED
    assert _holds_data(owner_url) == [1, 1, 2, 1, 1]


def test_0004_stops_at_a_blank_business_id_and_changes_nothing(database: tuple[str, str]) -> None:
    owner_url, _ = database
    config = alembic_config(owner_url)
    command.downgrade(config, "0003")
    at_0003, old_policies = _constraints(owner_url), _policies(owner_url)
    with owner_of(owner_url) as owner:
        owner.version("")  # what the weak policy let an unscoped session write
    with pytest.raises(IntegrityError, match="ck_knowledge_versions_business_id_not_blank"):
        command.upgrade(config, "0004")
    # DDL is transactional: the failed revision left no half-hardened database.
    assert _constraints(owner_url) == at_0003
    assert _policies(owner_url) == old_policies


def test_0004_stops_at_a_child_that_names_another_business_version(
    database: tuple[str, str],
) -> None:
    owner_url, _ = database
    config = alembic_config(owner_url)
    command.downgrade(config, "0003")
    at_0003, old_policies = _constraints(owner_url), _policies(owner_url)
    with owner_of(owner_url) as owner:
        version = owner.version("alpha")
        owner.document("beta", version)  # what a foreign key check could not see at 0003
    with pytest.raises(IntegrityError, match="fk_documents_version_business"):
        command.upgrade(config, "0004")
    assert _constraints(owner_url) == at_0003
    assert _policies(owner_url) == old_policies
