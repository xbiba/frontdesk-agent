"""The booking reply: whom to book, the free slots, and their lines written by code."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from frontdesk.agent.vocab import ClarifyField
from frontdesk.booking.calendar import FixtureCalendar
from frontdesk.booking.reply import booking_target, compose_slots, find_slots
from frontdesk.booking.slots import Slot
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Integrations

TIRANE = ZoneInfo("Europe/Tirane")
BOOKABLE = {
    "dr-arben-demo": ("orthopedics-visit",),
    "ft-elira-shembull": ("physiotherapy", "osteopathy"),
}


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TIRANE)


@pytest.mark.parametrize(
    ("bookable", "doctors", "services", "target"),
    [
        pytest.param(
            {"dr-arben-demo": ("orthopedics-visit",)},
            (),
            (),
            ("dr-arben-demo", "orthopedics-visit"),
            id="the-only-doctor",
        ),
        pytest.param(
            BOOKABLE,
            ("dr-arben-demo",),
            (),
            ("dr-arben-demo", "orthopedics-visit"),
            id="a-named-doctor",
        ),
        pytest.param(
            BOOKABLE, (), ("osteopathy",), ("ft-elira-shembull", "osteopathy"), id="a-named-service"
        ),
        pytest.param(
            BOOKABLE,
            ("ft-elira-shembull",),
            ("osteopathy",),
            ("ft-elira-shembull", "osteopathy"),
            id="both",
        ),
        pytest.param(BOOKABLE, (), (), ClarifyField.DOCTOR, id="which-doctor"),
        pytest.param(
            BOOKABLE, ("ft-elira-shembull",), (), ClarifyField.SERVICE, id="which-service"
        ),
        pytest.param(BOOKABLE, ("dr-giulia-esempio",), (), None, id="a-doctor-not-bookable-online"),
        pytest.param(BOOKABLE, (), ("ultrasound",), None, id="a-service-not-bookable-online"),
        pytest.param(
            BOOKABLE,
            ("dr-arben-demo",),
            ("osteopathy",),
            None,
            id="a-service-the-doctor-does-not-do",
        ),
    ],
)
def test_whom_to_book(
    bookable: dict[str, tuple[str, ...]],
    doctors: tuple[str, ...],
    services: tuple[str, ...],
    target: object,
) -> None:
    assert booking_target(bookable, doctors, services) == target


SLOTS = (
    Slot(at(23, 9), at(23, 9, 30)),
    Slot(at(26, 10), at(26, 10, 30)),  # after the change to winter time
    Slot(at(27, 16, 30), at(27, 17)),
)


@pytest.mark.parametrize(
    ("language", "text"),
    [
        pytest.param(
            "it",
            "Primi orari liberi con Dr. Arben Demo (Visita ortopedica, 30 minuti):\n"
            "- venerdì 23 ottobre, 09:00\n- lunedì 26 ottobre, 10:00\n- martedì 27 ottobre, 16:30",
            id="it",
        ),
        pytest.param(
            "en",
            "First free times with Dr. Arben Demo (Orthopedic visit, 30 minutes):\n"
            "- Friday 23 October, 09:00\n- Monday 26 October, 10:00\n- Tuesday 27 October, 16:30",
            id="en",
        ),
        pytest.param(
            "sq",
            "Oraret e para të lira me Dr. Arben Demo (Vizitë ortopedike, 30 minuta):\n"
            "- e premte 23 tetor, 09:00\n- e hënë 26 tetor, 10:00\n- e martë 27 tetor, 16:30",
            id="sq",
        ),
    ],
)
def test_slots_are_written_by_code_in_each_language(language: str, text: str) -> None:
    names = {"it": "Visita ortopedica", "en": "Orthopedic visit", "sq": "Vizitë ortopedike"}
    composed = compose_slots(
        SLOTS, doctor="Dr. Arben Demo", service=names[language], minutes=30, language=language
    )
    assert composed == text


def test_free_slots_come_from_the_doctor_s_calendar(
    clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    path = tmp_path / "calendar.yaml"
    path.write_text(
        """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 11:00"}
  busy:
    - {start: "2026-10-23 09:30", end: "2026-10-23 10:00"}
""",
        encoding="utf-8",
    )
    calendar = CalendarIntegration(credentials_ref="adc", calendars={"dr-arben-demo": "cal-arben"})
    pack = clinic_pack.model_copy(update={"integrations": Integrations(calendar=calendar)})
    slots = find_slots(FixtureCalendar(path), pack, "dr-arben-demo", "orthopedics-visit", at(22, 8))
    assert [(slot.start, slot.end - slot.start) for slot in slots] == [
        (at(23, 9), timedelta(minutes=30)),
        (at(23, 10), timedelta(minutes=30)),
        (at(23, 10, 30), timedelta(minutes=30)),
    ]
