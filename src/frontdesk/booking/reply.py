"""The booking reply (M4): whom to book, the free slots, and their lines, all written by code.

The model chooses the intent and may name a doctor or a service; code decides whether that can be
booked online, reads the doctor's calendar and writes every time with the fixed words below, as it
writes prices (ADR-0009). The Albanian words still need a native speaker's review (Task 16).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Final
from zoneinfo import ZoneInfo

from frontdesk.agent.vocab import ClarifyField, SlotPart
from frontdesk.booking.calendar import CalendarReader, CalendarView, opening_intervals
from frontdesk.booking.slots import Interval, Slot, spread
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, Language

if TYPE_CHECKING:
    from frontdesk.booking.scheduler import Scheduler

_LEADS: Final[dict[str, dict[str, str]]] = {
    "first": {
        "sq": "Oraret e para të lira me {doctor} ({service}, {minutes} minuta):",
        "it": "Primi orari liberi con {doctor} ({service}, {minutes} minuti):",
        "en": "First free times with {doctor} ({service}, {minutes} minutes):",
    },
    # The times not offered yet, when the patient asks for others.
    "more": {
        "sq": "Orare të tjera të lira me {doctor} ({service}, {minutes} minuta):",
        "it": "Altri orari liberi con {doctor} ({service}, {minutes} minuti):",
        "en": "More free times with {doctor} ({service}, {minutes} minutes):",
    },
    # The times that fit what the patient asked for: a day, from a day on, a part of the day.
    "wished": {
        "sq": "Oraret e lira me {doctor} {when} ({service}, {minutes} minuta):",
        "it": "Orari liberi con {doctor} {when} ({service}, {minutes} minuti):",
        "en": "Free times with {doctor} {when} ({service}, {minutes} minutes):",
    },
}
_FROM: Final[dict[str, str]] = {"sq": "nga {day}", "it": "dal {day}", "en": "from {day}"}
_ON: Final[dict[str, str]] = {"sq": "{day}", "it": "{day}", "en": "on {day}"}
_PARTS: Final[dict[SlotPart, dict[str, str]]] = {
    SlotPart.MORNING: {"sq": "në mëngjes", "it": "di mattina", "en": "in the morning"},
    SlotPart.AFTERNOON: {"sq": "pasdite", "it": "di pomeriggio", "en": "in the afternoon"},
}
AFTERNOON_HOUR: Final = 13
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
    scheduler: Scheduler, pack: BusinessPack, doctor_id: str, service_id: str, now: datetime
) -> tuple[Slot, ...]:
    """The first free slots to offer: the scheduler's, spread over the days. Raises
    CalendarUnavailableError."""
    every = scheduler.free_slots(pack, doctor_id, service_id, now)
    return spread(every, ZoneInfo(pack.timezone), _calendar(pack).max_options)


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


def _calendar(pack: BusinessPack) -> CalendarIntegration:
    if pack.integrations.calendar is None:
        raise ValueError(f"business '{pack.id}' has no calendar")
    return pack.integrations.calendar


def compose_slots(
    slots: Sequence[Slot],
    *,
    doctor: str,
    service: str,
    minutes: int,
    language: Language,
    numbered: bool = False,
    lead: str = "first",
    wish: SlotWish | None = None,
) -> str:
    """The lead-in and one line per slot; numbered when the patient picks one from the chat. `lead`:
    "first", "more" (the times not offered yet) or "wished" (those that fit `wish`)."""
    when = wish_words(wish, language) if wish is not None else ""
    head = _LEADS[lead][language].format(doctor=doctor, service=service, minutes=minutes, when=when)
    marks = [f"{index})" if numbered else "-" for index in range(1, len(slots) + 1)]
    lines = [
        f"{mark} {slot_time(slot.start, language)}" for mark, slot in zip(marks, slots, strict=True)
    ]
    return "\n".join([head, *lines])


def slot_time(start: datetime, language: Language) -> str:
    """A slot's start as the patient reads it: "venerdì 23 ottobre, 10:00"."""
    return f"{day_words(start.date(), language)}, {start:%H:%M}"


def day_words(day: date, language: Language) -> str:
    """A day as the patient reads it: "venerdì 23 ottobre"."""
    weekday = _WEEKDAYS[language][day.weekday()]
    month = _MONTHS[language][day.month - 1]
    return f"{weekday} {day.day} {month}"


@dataclass(frozen=True)
class SlotWish:
    """What the patient asked of the times (validated by code from the classifier's fields): one of
    the times offered, by its number; or a day, a first day and a part of the day."""

    slot: int | None = None
    day: date | None = None
    start_from: date | None = None
    part: SlotPart | None = None

    @property
    def narrows(self) -> bool:
        return self.day is not None or self.start_from is not None or self.part is not None

    def fits(self, start: datetime) -> bool:
        """`start` in the business's time zone."""
        if self.day is not None and start.date() != self.day:
            return False
        if self.start_from is not None and start.date() < self.start_from:
            return False
        afternoon = start.hour >= AFTERNOON_HOUR
        if self.part is SlotPart.MORNING and afternoon:
            return False
        return not (self.part is SlotPart.AFTERNOON and not afternoon)


def wish_words(wish: SlotWish, language: Language) -> str:
    """The wish as the patient reads it: "lunedì 26 ottobre, di pomeriggio", "dal 26 ottobre"."""
    parts = []
    if wish.day is not None:
        parts.append(_ON[language].format(day=day_words(wish.day, language)))
    elif wish.start_from is not None:
        month = _MONTHS[language][wish.start_from.month - 1]
        parts.append(_FROM[language].format(day=f"{wish.start_from.day} {month}"))
    if wish.part is not None:
        parts.append(_PARTS[wish.part][language])
    return ", ".join(parts)
