"""The attempts table: the limit, atomicity under concurrent turns, and row-level security."""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import OrderLookupAttempts
from frontdesk.shop.attempts import MAX_ORDER_LOOKUPS, consume_order_lookup

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
WORKERS = 8


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


def _consume(
    factory: sessionmaker[Session],
    business_id: str = "demo-shop",
    conversation_id: str = "c1",
    *,
    now: datetime = NOW,
    limit: int = MAX_ORDER_LOOKUPS,
) -> bool:
    with business_session(factory, business_id) as session:
        return consume_order_lookup(
            session, business_id=business_id, conversation_id=conversation_id, now=now, limit=limit
        )


def _row(factory: sessionmaker[Session], business_id: str = "demo-shop") -> OrderLookupAttempts:
    with business_session(factory, business_id) as session:
        return session.scalars(select(OrderLookupAttempts)).one()


def test_a_refused_attempt_changes_nothing(factory: sessionmaker[Session]) -> None:
    for _ in range(MAX_ORDER_LOOKUPS):
        assert _consume(factory) is True
    assert _consume(factory, now=NOW + timedelta(minutes=5)) is False
    row = _row(factory)
    assert (row.used, row.updated_at) == (MAX_ORDER_LOOKUPS, NOW)


def test_the_limit_is_a_parameter(factory: sessionmaker[Session]) -> None:
    assert [_consume(factory, limit=1) for _ in range(3)] == [True, False, False]
    assert _row(factory).used == 1


def test_concurrent_turns_cannot_take_more_attempts_than_the_limit(
    factory: sessionmaker[Session],
) -> None:
    barrier = threading.Barrier(WORKERS)

    def attempt(_: int) -> bool:
        barrier.wait(timeout=10)  # all turns reach the statement together
        return _consume(factory)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(attempt, range(WORKERS)))
    assert results.count(True) == MAX_ORDER_LOOKUPS
    assert results.count(False) == WORKERS - MAX_ORDER_LOOKUPS
    assert _row(factory).used == MAX_ORDER_LOOKUPS


def test_a_business_cannot_use_the_attempts_of_another(factory: sessionmaker[Session]) -> None:
    with (
        pytest.raises(DBAPIError, match="row-level security"),
        business_session(factory, "demo-shop") as session,
    ):
        consume_order_lookup(session, business_id="other-shop", conversation_id="c1", now=NOW)


def test_a_business_sees_only_its_own_attempts(factory: sessionmaker[Session]) -> None:
    _consume(factory, "demo-shop")
    with business_session(factory, "other-shop") as session:
        assert session.scalars(select(OrderLookupAttempts)).all() == []
    with factory() as session:  # no business scope at all
        assert session.scalars(select(OrderLookupAttempts)).all() == []


def test_the_attempts_table_has_the_policy_of_the_knowledge_tables(migrated_url: str) -> None:
    # Migration 0003 copies migration 0002's policy. If the policy is hardened later, both
    # change together and this test keeps holding; if only one changes, it fails.
    engine = create_engine(migrated_url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT tablename, permissive, roles, cmd, qual, with_check "
                "FROM pg_policies WHERE schemaname = 'public' ORDER BY tablename, policyname"
            )
        ).all()
    engine.dispose()
    policies: dict[str, list[tuple[object, ...]]] = {}
    for table, *policy in rows:
        policies.setdefault(table, []).append(tuple(policy))
    assert len(policies["order_lookup_attempts"]) == 1
    assert policies["order_lookup_attempts"] == policies["records"]
