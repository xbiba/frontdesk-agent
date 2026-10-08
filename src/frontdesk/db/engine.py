"""Engine, sessions and the business-scoped transaction that RLS policies read."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def business_session(factory: sessionmaker[Session], business_id: str) -> Iterator[Session]:
    """One transaction scoped to one business; commits on success, rolls back on error."""
    with factory() as session, session.begin():
        session.execute(
            text("SELECT set_config('app.business_id', :business_id, true)"),
            {"business_id": business_id},
        )
        yield session
