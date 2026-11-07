"""What the email pass remembers (design §3.1, §3 "Cosa resta salvato"): for each handled message,
its IDs, the outcome, a keyed hash of the sender, the day, the cost. Never a text or an address.

The last handled message of each conversation tells a new message from one already handled: a
sender who writes again in the same conversation gets a new draft. The counts per day and per
sender are the limits of §4.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from typing import Final, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session
from frontdesk.db.models import EmailHandled
from frontdesk.tenancy.pack import address_of

# The outcomes that are model requests, which the daily limit counts.
MODEL_OUTCOMES: Final = frozenset({"drafted", "review", "ignored_by_model", "no_output"})
# The outcomes that are drafts, which the limit per sender counts.
DRAFT_OUTCOMES: Final = frozenset({"drafted", "review"})


@dataclass(frozen=True)
class Handled:
    thread_id: str
    message_id: str
    outcome: str
    sender_key: str
    day: date
    at: datetime
    usd: float = 0.0


MIN_SECRET_BYTES: Final = 16


def sender_key(secret: bytes, sender: str) -> str:
    """A keyed hash of the sender's address: the same sender gives the same key, and the key gives
    back no address. `secret` is the deployment's own (frontdesk-email-sender-key), never a public
    value such as the business ID, which would let anyone try addresses until one matched
    (security review of 2026-11-07)."""
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(f"the sender key needs a secret of at least {MIN_SECRET_BYTES} bytes")
    address = address_of(sender) or sender.strip().lower()
    return hmac.new(secret, address.encode(), hashlib.sha256).hexdigest()[:32]


class Ledger(Protocol):
    def last_message(self, thread_id: str) -> str | None: ...

    def model_requests_on(self, day: date) -> int: ...

    def drafts_for(self, key: str, day: date) -> int: ...

    def record(self, handled: Handled) -> None: ...

    def purge(self, before: date) -> int: ...


class MemoryLedger:
    """The ledger of an evaluation or a dry run: in memory, gone at the end."""

    def __init__(self) -> None:
        self.rows: list[Handled] = []

    def last_message(self, thread_id: str) -> str | None:
        found = [row for row in self.rows if row.thread_id == thread_id]
        return max(found, key=lambda row: row.at).message_id if found else None

    def model_requests_on(self, day: date) -> int:
        return sum(1 for row in self.rows if row.day == day and row.outcome in MODEL_OUTCOMES)

    def drafts_for(self, key: str, day: date) -> int:
        return sum(
            1
            for row in self.rows
            if row.day == day and row.sender_key == key and row.outcome in DRAFT_OUTCOMES
        )

    def record(self, handled: Handled) -> None:
        self.rows = [
            row
            for row in self.rows
            if (row.thread_id, row.message_id) != (handled.thread_id, handled.message_id)
        ]
        self.rows.append(replace(handled))

    def purge(self, before: date) -> int:
        kept = [row for row in self.rows if row.day >= before]
        gone, self.rows = len(self.rows) - len(kept), kept
        return gone


class PostgresLedger:
    """The ledger in the business's database (migration 0009), under its row-level security."""

    def __init__(self, session_factory: sessionmaker[Session], business_id: str) -> None:
        self._factory = session_factory
        self._business = business_id

    def last_message(self, thread_id: str) -> str | None:
        with business_session(self._factory, self._business) as session:
            return session.scalar(
                select(EmailHandled.message_id)
                .where(
                    EmailHandled.business_id == self._business,
                    EmailHandled.thread_id == thread_id,
                )
                .order_by(EmailHandled.handled_at.desc())
                .limit(1)
            )

    def model_requests_on(self, day: date) -> int:
        return self._count(day, EmailHandled.outcome.in_(MODEL_OUTCOMES))

    def drafts_for(self, key: str, day: date) -> int:
        return self._count(
            day, EmailHandled.sender_key == key, EmailHandled.outcome.in_(DRAFT_OUTCOMES)
        )

    def _count(self, day: date, *conditions: object) -> int:
        with business_session(self._factory, self._business) as session:
            found = session.scalar(
                select(func.count())
                .select_from(EmailHandled)
                .where(EmailHandled.business_id == self._business, EmailHandled.day == day)
                .where(*conditions)  # type: ignore[arg-type]
            )
        return int(found or 0)

    def record(self, handled: Handled) -> None:
        values = {
            "business_id": self._business,
            "thread_id": handled.thread_id[:128],
            "message_id": handled.message_id[:128],
            "outcome": handled.outcome,
            "sender_key": handled.sender_key,
            "day": handled.day,
            "handled_at": handled.at,
            "usd": Decimal(str(round(handled.usd, 6))),
        }
        statement = insert(EmailHandled).values(**values)
        update = {key: statement.excluded[key] for key in ("outcome", "day", "handled_at", "usd")}
        with business_session(self._factory, self._business) as session:
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=["business_id", "thread_id", "message_id"], set_=update
                )
            )

    def purge(self, before: date) -> int:
        with business_session(self._factory, self._business) as session:
            result = session.execute(
                delete(EmailHandled).where(
                    EmailHandled.business_id == self._business, EmailHandled.day < before
                )
            )
        return int(getattr(result, "rowcount", 0) or 0)
