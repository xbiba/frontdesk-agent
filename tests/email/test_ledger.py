"""The email pass's ledger in Postgres (migration 0009): IDs, outcome, day and cost, no text; the
counts of the limits, one business at a time, and the purge after 30 days."""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import text

from frontdesk.db.engine import make_engine, make_session_factory
from frontdesk.email.ledger import Handled, MemoryLedger, PostgresLedger, sender_key

DAY = date(2026, 11, 7)
AT = datetime(2026, 11, 7, 9, tzinfo=UTC)


def test_the_sender_key_hides_the_address() -> None:
    key = sender_key("demo-clinic", "Anna <ANNA@example.com>")
    assert key == sender_key("demo-clinic", "anna@example.com")
    assert key != sender_key("demo-shop", "anna@example.com")
    assert "anna" not in key
    assert len(key) == 32


def _rows(key: str) -> list[Handled]:
    return [
        Handled("t1", "m1", "drafted", key, DAY, AT, 0.002),
        Handled("t1", "m2", "review", key, DAY, AT.replace(hour=10)),
        Handled("t2", "m3", "ignored", "other", DAY, AT),
        Handled("t3", "m4", "ignored_by_model", "other", DAY, AT),
        Handled("t4", "m5", "drafted", key, date(2026, 10, 1), AT.replace(month=10, day=1)),
    ]


def _check(ledger: PostgresLedger | MemoryLedger) -> None:
    key = sender_key("demo-clinic", "anna@example.com")
    for row in _rows(key):
        ledger.record(row)
    ledger.record(Handled("t1", "m2", "review", key, DAY, AT.replace(hour=10), 0.001))  # again
    assert ledger.last_message("t1") == "m2"
    assert ledger.last_message("t9") is None
    assert ledger.model_requests_on(DAY) == 3
    assert ledger.drafts_for(key, DAY) == 2
    assert ledger.purge(date(2026, 10, 8)) == 1
    assert ledger.last_message("t4") is None


def test_the_memory_ledger() -> None:
    _check(MemoryLedger())


@pytest.mark.db
def test_the_postgres_ledger_counts_per_business(app_url: str) -> None:
    factory = make_session_factory(make_engine(app_url))
    _check(PostgresLedger(factory, "demo-clinic"))
    other = PostgresLedger(factory, "demo-shop")
    assert other.last_message("t1") is None
    assert other.model_requests_on(DAY) == 0
    with factory() as session:
        columns = session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'email_handled'"
            )
        ).scalars()
        assert set(columns) == {
            "business_id",
            "thread_id",
            "message_id",
            "outcome",
            "sender_key",
            "day",
            "handled_at",
            "usd",
        }
