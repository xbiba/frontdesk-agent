"""`frontdesk bookings purge`: expired holds go, with their names and phones."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from frontdesk.booking.holds import HOLD_LIFETIME, attach, hold
from frontdesk.cli.bookings import purge_bookings
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.settings import Settings

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 20, 8, tzinfo=UTC)


def test_purge_deletes_the_expired_holds_of_the_deployment_s_business(
    app_url: str, businesses_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, "demo-clinic") as session:
        hold_id = hold(
            session,
            business_id="demo-clinic",
            conversation_id="c1",
            doctor_id="dr-arben-demo",
            service_id="orthopedics-visit",
            start=NOW + timedelta(days=3),
            end=NOW + timedelta(days=3, minutes=30),
            now=NOW,
        )
        assert hold_id is not None
        attach(session, hold_id, name="Mario Rossi", phone="+355691234567")
    settings = Settings(business_pack=businesses_dir / "demo-clinic", database_url=app_url)
    later = NOW + HOLD_LIFETIME + timedelta(minutes=1)
    assert purge_bookings(settings, later) == 0
    assert capsys.readouterr().out == "booking holds deleted: 1\n"
