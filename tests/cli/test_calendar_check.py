"""`frontdesk calendar check`: a doctor's calendar read as the bot does, without event content."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from frontdesk.cli.calendar import check_calendar
from frontdesk.cli.main import app
from frontdesk.settings import CalendarBackend, Settings
from tests.tenancy.helpers import CLINIC, copy_pack, edit_yaml

NOW = datetime(2026, 10, 22, 8, tzinfo=ZoneInfo("Europe/Tirane"))
CALENDAR = """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 10:00"}
  busy:
    - {start: "2026-10-23 09:00", end: "2026-10-23 09:30"}
"""


def _settings(
    businesses_dir: Path, tmp_path: Path, calendar_id: str = "cal-arben", **extra: object
) -> Settings:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    calendar = {"credentials_ref": "adc", "calendars": {"dr-arben-demo": calendar_id}}
    edit_yaml(pack_dir / "business.yaml", lambda d: d.update(integrations={"calendar": calendar}))
    fixture = tmp_path / "calendar.yaml"
    fixture.write_text(CALENDAR, encoding="utf-8")
    values = {
        "business_pack": pack_dir,
        "calendar_backend": CalendarBackend.FIXTURE,
        "calendar_fixture": fixture,
    }
    return Settings(**{**values, **extra})


def test_a_readable_calendar_prints_counts_and_free_times(
    businesses_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check_calendar(_settings(businesses_dir, tmp_path), "dr-arben-demo", NOW) == 0
    assert capsys.readouterr().out.splitlines() == [
        "free slots of dr-arben-demo in the next 14 days: 1",
        "free: 2026-10-23 09:30 (orthopedics-visit)",
    ]


def test_a_doctor_or_a_deployment_that_cannot_book_is_an_error(
    businesses_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check_calendar(_settings(businesses_dir, tmp_path), "dr-giulia-esempio", NOW) == 1
    assert "not a bookable doctor" in capsys.readouterr().err
    none = _settings(businesses_dir, tmp_path / "none", calendar_backend=CalendarBackend.NONE)
    assert check_calendar(none, "dr-arben-demo", NOW) == 1
    assert "FRONTDESK_CALENDAR_BACKEND is 'none'" in capsys.readouterr().err


def test_a_calendar_that_cannot_be_read_exits_2_through_the_command(
    businesses_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(businesses_dir, tmp_path, calendar_id="cal-unknown")
    env = tmp_path / "check.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={settings.business_pack}\n"
        "FRONTDESK_CALENDAR_BACKEND=fixture\n"
        f"FRONTDESK_CALENDAR_FIXTURE={settings.calendar_fixture}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["calendar", "check", "--env-file", str(env), "--doctor", "dr-arben-demo"]
    )
    assert result.exit_code == 2
    assert "calendar: cannot be read (unknown calendar)" in result.output
