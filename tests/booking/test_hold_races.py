"""A hold that loses a race for its time is refused like an overlap, not raised (no database needed).

Two holds inserted on one time at the same instant can wait on each other in the exclusion check, and
Postgres then cancels one of them with a deadlock error (seen in CI on
test_parallel_holds_on_one_time_leave_exactly_one). The patient whose hold was cancelled must hear
that the time is taken, not get an error.
"""

from datetime import UTC, datetime
from types import TracebackType
from typing import cast

import psycopg
import pytest
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from frontdesk.booking.holds import hold

NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)
NINE = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
TEN = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)


class _Savepoint:
    """The savepoint around the insert: leaving it flushes the row, which fails with `error`."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __enter__(self) -> "_Savepoint":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        raise self._error


class _Session:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def execute(self, statement: object) -> None:
        return None

    def begin_nested(self) -> _Savepoint:
        return _Savepoint(self._error)

    def add(self, row: object) -> None:
        return None


def _hold(error: Exception) -> object:
    return hold(
        cast(Session, _Session(error)),
        business_id="clinica-a",
        conversation_id="c1",
        doctor_id="dr-a",
        service_id="osteopathy",
        start=NINE,
        end=TEN,
        now=NOW,
    )


def test_a_hold_cancelled_by_a_deadlock_is_refused_like_an_overlap() -> None:
    deadlock = OperationalError(
        "INSERT INTO booking_holds", {}, psycopg.errors.DeadlockDetected("deadlock detected")
    )
    assert _hold(deadlock) is None


def test_an_overlap_is_still_refused() -> None:
    overlap = IntegrityError(
        "INSERT INTO booking_holds", {}, psycopg.errors.ExclusionViolation("conflicting key")
    )
    assert _hold(overlap) is None


@pytest.mark.parametrize(
    "error",
    [
        OperationalError(
            "INSERT INTO booking_holds", {}, psycopg.errors.QueryCanceled("canceling statement")
        ),
        IntegrityError(
            "INSERT INTO booking_holds",
            {},
            psycopg.errors.CheckViolation("ck_booking_holds_status"),
        ),
    ],
    ids=["a-timeout", "another-constraint"],
)
def test_any_other_database_error_still_raises(error: Exception) -> None:
    with pytest.raises(type(error)):
        _hold(error)
