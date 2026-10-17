"""Migration 0008 (WhatsApp W1) on a database that holds requests, down to 0007 and up again."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import create_engine, text

from frontdesk.db.migrate import alembic_config

pytestmark = pytest.mark.db

COLUMNS = (
    "id, business_id, conversation_id, doctor_id, service_id, during, status, expires_at, name, "
    "request_status, language, created_at"
)
NINE = "tstzrange('2026-10-23 09:00+02', '2026-10-23 10:00+02')"
ELEVEN = "tstzrange('2026-10-23 11:00+02', '2026-10-23 12:00+02')"
# A pending request with the patient's name; a declined one, whose time another patient now holds
# (final review of W1, I4).
ROWS = (
    f"INSERT INTO booking_holds ({COLUMNS}) VALUES "
    f"(gen_random_uuid(), 'alpha', 'c1', 'dr-a', 'osteopathy', {NINE}, 'booked', "
    "'2026-10-23 10:00+02', 'Mario Rossi', 'pending', 'it', now()), "
    f"(gen_random_uuid(), 'alpha', 'c2', 'dr-a', 'osteopathy', {ELEVEN}, 'booked', "
    "'2026-10-23 12:00+02', NULL, 'declined', 'it', now()), "
    f"(gen_random_uuid(), 'alpha', 'c3', 'dr-a', 'osteopathy', {ELEVEN}, 'held', "
    "'2026-10-20 10:15+02', NULL, NULL, NULL, now())"
)


def _rows(url: str) -> list[tuple[str, str | None]]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            found = connection.execute(
                text("SELECT conversation_id, name FROM booking_holds ORDER BY conversation_id")
            )
            return [(row[0], row[1]) for row in found]
    finally:
        engine.dispose()


def test_0008_downgrades_a_database_with_requests_and_upgrades_again(
    database: tuple[str, str],
) -> None:
    owner_url, _ = database
    engine = create_engine(owner_url)
    with engine.begin() as connection:
        connection.execute(text(ROWS))
    engine.dispose()
    config = alembic_config(owner_url)

    command.downgrade(config, "0007")
    # The declined request is no appointment, and 0005 keeps no name on a booked row.
    assert _rows(owner_url) == [("c1", None), ("c3", None)]

    command.upgrade(config, "head")
    assert _rows(owner_url) == [("c1", None), ("c3", None)]
