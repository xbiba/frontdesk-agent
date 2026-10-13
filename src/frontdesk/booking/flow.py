"""A booking from the chat (M5): what a name, a phone and a yes may be, and the phrases.

The reception agent's tools check what the patient gave: a name, a phone number, a yes to the
summary (frontdesk.conversation.reception). The phrases are
written by code in sq, it and en (ADR-0009); the Albanian ones still need a native speaker's
review. Nothing here keeps a name or a phone: the caller stores them on the hold.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_NAME: Final = re.compile(r"^[^\W\d_]+(?:[ '\u2019\-.]+[^\W\d_]+)*\.?$")
_PHONE_SEPARATORS: Final = re.compile(r"[\s\-./()]")
_PHONE: Final = re.compile(r"^\+?\d{8,15}$")
_PHRASES: Final[dict[str, dict[str, str]]] = {
    "ask_name": {
        "sq": "Mund të më shkruani emrin dhe mbiemrin tuaj?",
        "it": "Mi scrive il suo nome e cognome?",
        "en": "Could you write your first and last name?",
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


def valid_name(message: str) -> str | None:
    """A name of letters, with spaces, apostrophes, hyphens or dots between its words."""
    name = " ".join(unicodedata.normalize("NFC", message).split())
    letters = sum(char.isalpha() for char in name)
    return name if 2 <= len(name) <= 60 and letters >= 2 and _NAME.match(name) else None


def valid_phone(message: str, country_code: str = "") -> str | None:
    """A phone number in international form, "+" and 8 to 15 digits, separators removed. "00"
    stands for "+", and a national number ("069…") takes the clinic's country code."""
    phone = _PHONE_SEPARATORS.sub("", message.strip())
    if phone.startswith("00"):
        phone = "+" + phone[2:]
    elif phone.startswith("0") and country_code:
        phone = country_code + phone[1:]
    return phone if phone.startswith("+") and _PHONE.match(phone) else None


# A yes in a sentence ("sì, confermo grazie"), for the reception agent's confirm_booking: one of
# these words and none of the negations.
_AFFIRMATIVE: Final = frozenset(
    {
        "si",
        "sì",
        "sí",
        "yes",
        "yeah",
        "yep",
        "po",
        "ok",
        "okay",
        "confermo",
        "conferma",
        "confermato",
        "confirm",
        "confirmed",
        "konfirmoj",
        "certo",
        "sure",
        "perfetto",
        "esatto",
        "bene",
        "dakord",
        "mirë",
    }
)
_NEGATIVE: Final = frozenset(
    {"no", "jo", "non", "nuk", "not", "nope", "annulla", "cancel", "stop", "anulo", "aspetta"}
)


def affirms(message: str) -> bool:
    """The message says yes, and nothing in it says no."""
    words = set(re.findall(r"[^\W\d_]+", message.casefold()))
    return not words & _NEGATIVE and bool(words & _AFFIRMATIVE)
