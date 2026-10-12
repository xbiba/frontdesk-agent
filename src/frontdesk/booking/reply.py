"""The booking reply (M4): whom to book, the free slots, and their lines, all written by code.

The model chooses the intent and may name a doctor or a service; code decides whether that can be
booked online, reads the doctor's calendar and writes every time with the fixed words below, as it
writes prices (ADR-0009). The Albanian words still need a native speaker's review (Task 16).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

from frontdesk.agent.vocab import ClarifyField
from frontdesk.booking.calendar import CalendarReader, CalendarView, opening_intervals
from frontdesk.booking.slots import Interval, Slot, free_slots
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Language

_LEAD: Final[dict[str, str]] = {
    "sq": "Oraret e para të lira me {doctor} ({service}, {minutes} minuta):",
    "it": "Primi orari liberi con {doctor} ({service}, {minutes} minuti):",
    "en": "First free times with {doctor} ({service}, {minutes} minutes):",
}
_WEEKDAYS: Final[dict[str, tuple[str, ...]]] = {
    "sq": ("e hënë", "e martë", "e mërkurë", "e enjte", "e premte", "e shtunë", "e diel"),
    "it": ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
}
_MONTHS: Final[dict[str, tuple[str, ...]]] = {
    "sq": (
        "janar",
        "shkurt",
        "mars",
        "prill",
        "maj",
        "qershor",
        "korrik",
        "gusht",
        "shtator",
        "tetor",
        "nëntor",
        "dhjetor",
    ),
    "it": (
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
    ),
    "en": (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ),
}


def booking_target(
    bookable: Mapping[str, Sequence[str]],
    doctor_ids: Sequence[str],
    service_ids: Sequence[str],
) -> tuple[str, str] | ClarifyField | None:
    """The doctor and the service to book, the question to ask when it is not clear, or None when
    what the person named cannot be booked online (reception books it)."""
    if doctor_ids:
        doctors = [item for item in doctor_ids if item in bookable]
    elif service_ids:
        doctors = [item for item, offered in bookable.items() if set(offered) & set(service_ids)]
    else:
        doctors = list(bookable)
    if not doctors:
        return None
    if len(doctors) > 1:
        return ClarifyField.DOCTOR
    offered = bookable[doctors[0]]
    services = [item for item in offered if item in service_ids] if service_ids else list(offered)
    if not services:
        return None
    if len(services) > 1:
        return ClarifyField.SERVICE
    return doctors[0], services[0]


def find_slots(
    reader: CalendarReader, pack: BusinessPack, doctor_id: str, service_id: str, now: datetime
) -> tuple[Slot, ...]:
    """The free slots of the doctor's calendar for the service. Raises CalendarUnavailableError."""
    return slots_in(read_calendar(reader, pack, doctor_id, now), pack, service_id, now)


def read_calendar(
    reader: CalendarReader, pack: BusinessPack, doctor_id: str, now: datetime
) -> CalendarView:
    """The doctor's blocks and busy time from now to the horizon. Raises CalendarUnavailableError."""
    calendar = _calendar(pack)
    zone = ZoneInfo(pack.timezone)

    def opening(day: date) -> list[Interval]:
        return opening_intervals(pack.location.opening_hours, day, zone)

    return reader.read(
        calendar.calendars[doctor_id],
        now,
        now + timedelta(days=calendar.horizon_days),
        marker=calendar.availability_marker,
        zone=zone,
        opening=opening,
    )


def slots_in(
    view: CalendarView, pack: BusinessPack, service_id: str, now: datetime
) -> tuple[Slot, ...]:
    calendar = _calendar(pack)
    service = next(item for item in pack.services if item.id == service_id)
    return free_slots(
        view.blocks,
        view.busy,
        duration=timedelta(minutes=service.duration_minutes),
        now=now,
        horizon=timedelta(days=calendar.horizon_days),
        min_notice=timedelta(minutes=calendar.min_notice_minutes),
        buffer=timedelta(minutes=calendar.buffer_minutes),
        zone=ZoneInfo(pack.timezone),
        limit=calendar.max_options,
    )


def _calendar(pack: BusinessPack) -> CalendarIntegration:
    if pack.integrations.calendar is None:
        raise ValueError(f"business '{pack.id}' has no calendar")
    return pack.integrations.calendar


def compose_slots(
    slots: Sequence[Slot], *, doctor: str, service: str, minutes: int, language: Language
) -> str:
    lead = _LEAD[language].format(doctor=doctor, service=service, minutes=minutes)
    return "\n".join([lead, *(f"- {_when(slot.start, language)}" for slot in slots)])


def _when(start: datetime, language: Language) -> str:
    weekday = _WEEKDAYS[language][start.weekday()]
    month = _MONTHS[language][start.month - 1]
    return f"{weekday} {start.day} {month}, {start:%H:%M}"
