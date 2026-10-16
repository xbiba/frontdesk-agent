"""The calendar scheduler of the evaluations decides a request once, as cal.com does (W1)."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from frontdesk.booking.calendar import FixtureCalendar, RequestGoneError
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Integrations

TIRANE = ZoneInfo("Europe/Tirane")
TEN = datetime(2026, 10, 23, 10, tzinfo=TIRANE)
ELEVEN = datetime(2026, 10, 23, 11, tzinfo=TIRANE)


def test_the_fixture_decides_once(clinic_pack: BusinessPack, tmp_path: Path) -> None:
    calendar = CalendarIntegration(
        credentials_ref="adc", calendars={"dr-arben-demo": "cal-arben"}, requires_confirmation=True
    )
    pack = clinic_pack.model_copy(update={"integrations": Integrations(calendar=calendar)})
    path = tmp_path / "calendar.yaml"
    path.write_text("cal-arben: {blocks: [], busy: []}\n", encoding="utf-8")
    scheduler = CalendarScheduler(FixtureCalendar(path))
    scheduler.book(
        pack,
        "dr-arben-demo",
        "orthopedics-visit",
        TEN,
        ELEVEN,
        name="Mario Prova",
        phone="+355690000001",
        reference="hold1",
        language="it",
    )
    scheduler.decide(pack, "dr-arben-demo", TEN, ELEVEN, "hold1", accept=True)
    assert scheduler.decided == [("hold1", True)]
    with pytest.raises(RequestGoneError):  # already decided
        scheduler.decide(pack, "dr-arben-demo", TEN, ELEVEN, "hold1", accept=False)
    with pytest.raises(RequestGoneError):  # never booked here
        scheduler.decide(pack, "dr-arben-demo", TEN, ELEVEN, "hold9", accept=True)
    assert scheduler.decided == [("hold1", True)]
