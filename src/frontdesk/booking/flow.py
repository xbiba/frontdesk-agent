"""The steps of a booking from the chat (M5): what a reply to each step may be, and the phrases.

Code reads the patient's answers, never the model: the choice of a slot (its number or its time), a
name, a phone number, yes or no. The phrases are written by code in sq, it and en (ADR-0009); the
Albanian ones still need a native speaker's review. Nothing here keeps a name or a phone: the
caller stores them on the hold.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

from frontdesk.booking.calendar import CalendarView
from frontdesk.booking.slots import free_slots

_CHOICE: Final = re.compile(
    r"^(?:(?:il|la|lo|l'|the|nr|n|no|numero|number|numri|nummri)\.?\s*)?([1-9])"
    r"(?:st|nd|rd|th|°|º)?\s*[.)!]?$",
    re.IGNORECASE,
)
_TIME: Final = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
_NAME: Final = re.compile(r"^[^\W\d_]+(?:[ '\u2019\-.]+[^\W\d_]+)*\.?$")
_PHONE_SEPARATORS: Final = re.compile(r"[\s\-./()]")
_PHONE: Final = re.compile(r"^\+?\d{8,15}$")
_YES: Final = frozenset(
    {
        "si",
        "sì",
        "sí",
        "yes",
        "y",
        "po",
        "ok",
        "okay",
        "confermo",
        "conferma",
        "va bene",
        "certo",
        "sure",
    }
)
_NO: Final = frozenset({"no", "jo", "nope", "annulla", "cancel", "stop", "anulo", "lascia stare"})
_PHRASES: Final[dict[str, dict[str, str]]] = {
    "choose": {
        "sq": "Për të rezervuar, shkruani numrin e orarit që preferoni.",
        "it": "Per prenotare, scriva il numero dell'orario che preferisce.",
        "en": "To book, write the number of the time you prefer.",
    },
    "ask_name": {
        "sq": "Mund të më shkruani emrin dhe mbiemrin tuaj?",
        "it": "Mi scrive il suo nome e cognome?",
        "en": "Could you write your first and last name?",
    },
    "ask_phone": {
        "sq": "Dhe numrin tuaj të telefonit?",
        "it": "E il suo numero di telefono?",
        "en": "And your phone number?",
    },
    "again_name": {
        "sq": "Ju lutem, shkruani vetëm emrin dhe mbiemrin.",
        "it": "Mi scriva solo nome e cognome, per favore.",
        "en": "Please write just your first and last name.",
    },
    "again_phone": {
        "sq": "Ju lutem, shkruani numrin e telefonit vetëm me shifra.",
        "it": "Mi scriva il numero di telefono con le sole cifre, per favore.",
        "en": "Please write the phone number with digits only.",
    },
    "again_yes_no": {
        "sq": "Ju lutem, përgjigjuni po ose jo.",
        "it": "Risponda sì o no, per favore.",
        "en": "Please answer yes or no.",
    },
    "summary": {
        "sq": "Përmbledhje: {when}, {service} me {doctor}.\nEmri: {name}\nTelefoni: {phone}\nTa konfirmoj rezervimin? Përgjigjuni po ose jo.",
        "it": "Riepilogo: {when}, {service} con {doctor}.\nNome: {name}\nTelefono: {phone}\nConfermo la prenotazione? Risponda sì o no.",
        "en": "Summary: {when}, {service} with {doctor}.\nName: {name}\nPhone: {phone}\nShall I confirm the booking? Please answer yes or no.",
    },
    "booked": {
        "sq": "Rezervimi u konfirmua: {when}, {service} me {doctor}. Për ta anuluar ose zhvendosur, telefononi recepsionin në {contact}.",
        "it": "Prenotazione confermata: {when}, {service} con {doctor}. Per annullarla o spostarla, chiami la reception al {contact}.",
        "en": "Booking confirmed: {when}, {service} with {doctor}. To cancel or move it, call reception on {contact}.",
    },
    "cancelled": {
        "sq": "Në rregull, nuk rezervova asgjë.",
        "it": "Va bene, non ho prenotato nulla.",
        "en": "All right, nothing has been booked.",
    },
    "taken": {
        "sq": "Ky orar nuk është më i lirë.",
        "it": "Questo orario non è più libero.",
        "en": "That time is no longer free.",
    },
    "expired": {
        "sq": "Koha për rezervimin mbaroi.",
        "it": "Il tempo per prenotare è scaduto.",
        "en": "The time to book has run out.",
    },
    "failed": {
        "sq": "Nuk arrita ta përfundoj rezervimin.",
        "it": "Non sono riuscito a completare la prenotazione.",
        "en": "I could not complete the booking.",
    },
}
# The calendar entry reception reads: in the business's default language.
_ENTRY: Final[dict[str, str]] = {
    "sq": "Telefoni: {phone}\nRezervuar nga chat-i.",
    "it": "Telefono: {phone}\nPrenotato dalla chat.",
    "en": "Phone: {phone}\nBooked from the chat.",
}


def phrase(key: str, language: str, **values: str) -> str:
    return _PHRASES[key][language].format(**values)


def entry_description(phone: str, language: str) -> str:
    return _ENTRY[language].format(phone=phone)


def parse_choice(message: str, offered: Sequence[datetime], zone: ZoneInfo) -> datetime | None:
    """The offered slot the message picks: its number ("2", "il 2") or its time ("10:00")."""
    text = message.strip()
    number = _CHOICE.match(text)
    if number is not None:
        index = int(number.group(1))
        return offered[index - 1] if index <= len(offered) else None
    local = {(item.astimezone(zone).hour, item.astimezone(zone).minute): item for item in offered}
    picked = {local.get((int(hour), int(minute))) for hour, minute in _TIME.findall(text)}
    picked.discard(None)
    return picked.pop() if len(picked) == 1 else None


def valid_name(message: str) -> str | None:
    """A name of letters, with spaces, apostrophes, hyphens or dots between its words."""
    name = " ".join(unicodedata.normalize("NFC", message).split())
    letters = sum(char.isalpha() for char in name)
    return name if 2 <= len(name) <= 60 and letters >= 2 and _NAME.match(name) else None


def valid_phone(message: str) -> str | None:
    """A phone number: an optional + and 8 to 15 digits, separators removed."""
    phone = _PHONE_SEPARATORS.sub("", message.strip())
    return phone if _PHONE.match(phone) else None


def yes_or_no(message: str) -> bool | None:
    words = " ".join(re.sub(r"[^\w\s']", " ", message.casefold()).split())
    if words in _YES:
        return True
    if words in _NO:
        return False
    return None


def still_free(
    view: CalendarView,
    start: datetime,
    *,
    duration: timedelta,
    now: datetime,
    horizon: timedelta,
    min_notice: timedelta,
    buffer: timedelta,
    zone: ZoneInfo,
) -> bool:
    """The slot is still among the free ones, by the same arithmetic that offered it."""
    every = free_slots(
        view.blocks,
        view.busy,
        duration=duration,
        now=now,
        horizon=horizon,
        min_notice=min_notice,
        buffer=buffer,
        zone=zone,
        limit=10_000,
    )
    return any(slot.start == start for slot in every)
