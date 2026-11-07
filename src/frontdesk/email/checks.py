"""The checks in code on a draft (design §3.4), whose failures go to the staff as "da verificare":

- every sentence the model cites as evidence is found, word for word, in the information;
- no word of the chat channel ("chat", "bisedë");
- no sentence that says something does not exist or is not offered (Italian, English, Albanian);
- a question without evidence leaves the placeholder in the draft;
- the signature is there;
- no email address, phone number or price that the information does not hold.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

CHAT: Final = re.compile(r"\bchat\b|\bbisedë\b|\bbiseda\b", re.IGNORECASE)
ABSENCE: Final = re.compile(
    r"non\s+risult|non\s+compar|non\s+offriamo|non\s+(?:è|e')\s+(?:un\s+servizio|disponibil|previst|offert)|"
    r"non\s+(?:abbiamo|disponiamo|effettuiamo|eseguiamo|facciamo|esiste)|"
    r"we\s+do\s*n[o']t\s+(?:offer|have|provide)|we\s+don't\s+(?:offer|have|provide)|"
    r"(?:is|are)\s+not\s+(?:offered|available|among)|does\s+not\s+(?:offer|exist)|"
    r"nuk\s+(?:ofrojmë|ofrojme|kemi|është\s+(?:një\s+)?shërbim|eshte\s+sherbim|ekziston)",
    re.IGNORECASE,
)
_EMAIL: Final = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE: Final = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_DATE: Final = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
_PRICE: Final = re.compile(
    r"(?:€|eur|euro|all|lek[ëe]?)\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*(?:€|eur\b|euro|all\b|lek)",
    re.IGNORECASE,
)


class Flag(StrEnum):
    QUOTE_NOT_FOUND = "quote_not_found"
    CHAT_WORDING = "chat_wording"
    ABSENCE_CLAIM = "absence_claim"
    NO_EVIDENCE_NO_TODO = "no_evidence_no_todo"
    NO_SIGNATURE = "no_signature"
    UNKNOWN_CONTACT = "unknown_contact"
    UNKNOWN_PRICE = "unknown_price"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    pairs = (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'), ("**", ""))
    for old, new in (*pairs, ("`", "")):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in(quote: str, knowledge: str) -> bool:
    """A quote is verbatim when each of its parts (split on ellipses) is in the normalized
    information. `knowledge` is already normalized."""
    parts = [part.strip(" \"'«».,;:-") for part in re.split(r"\.\.\.|…", normalize(quote))]
    parts = [part for part in parts if part]
    return bool(parts) and all(part in knowledge for part in parts)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def contact_and_price_flags(
    draft: str, knowledge: str, *, allowed: Sequence[str] = ()
) -> list[Flag]:
    """Addresses, phone numbers and prices in the draft that the information does not hold.
    `allowed`: email addresses that may be named besides (the sender's own); never a number."""
    known = knowledge.lower()
    known_digits = {_digits(found) for found in _PHONE.findall(known)}
    addresses = {item.lower() for item in allowed}
    flags: list[Flag] = []
    emails = [found.lower() for found in _EMAIL.findall(draft) if found.lower() not in addresses]
    phones = [_digits(found) for found in _PHONE.findall(draft) if not _DATE.match(found.strip())]
    if any(item not in known for item in emails) or any(
        len(item) >= 8
        and not any(item in digits or digits in item for digits in known_digits if digits)
        for item in phones
    ):
        flags.append(Flag.UNKNOWN_CONTACT)
    known_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", known))
    prices = [first or second for first, second in _PRICE.findall(draft)]
    if any(amount not in known_numbers for amount in prices):
        flags.append(Flag.UNKNOWN_PRICE)
    return flags


def wording_flags(draft: str) -> list[Flag]:
    flags = []
    if CHAT.search(draft):
        flags.append(Flag.CHAT_WORDING)
    if ABSENCE.search(draft):
        flags.append(Flag.ABSENCE_CLAIM)
    return flags
