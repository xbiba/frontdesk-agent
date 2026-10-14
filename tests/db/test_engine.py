"""The engine (M4 review): a database error never carries the values of its statement, which can
be a person's message or their Instagram ID."""

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from frontdesk.db.engine import make_engine

pytestmark = pytest.mark.db
SENTINEL = "SENTINEL-mi-chiamo-Mario"


def _cast(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT CAST(:value AS no_such_type)"), {"value": SENTINEL})


def test_database_errors_leave_out_the_statement_s_values(app_url: str) -> None:
    engine = make_engine(app_url)
    with pytest.raises(DBAPIError) as raised:
        _cast(engine)
    assert SENTINEL not in str(raised.value)
    engine.dispose()
