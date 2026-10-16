"""The WhatsApp channel's staff list (W1, spec D3): the only numbers the bot reads and writes to.

It lives in a secret, outside the repository, since it holds personal numbers: a JSON list of
`{"phone", "name", "role"}`. An error names the entry's position, never its number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat

MAX_NAME: Final = 40


class Role(StrEnum):
    RECEPTION = "reception"  # gets the booking requests (W1)
    DOCTOR = "doctor"


@dataclass(frozen=True)
class Staff:
    phone: str  # E.164
    name: str
    role: Role

    @property
    def user_id(self) -> str:
        """The WhatsApp ID: the number's digits, as webhooks write it."""
        return self.phone.removeprefix("+")


def read_staff(raw: str) -> tuple[Staff, ...]:
    try:
        entries = json.loads(raw)
    except ValueError as error:
        raise ValueError("the staff list is not JSON") from error
    if not isinstance(entries, list) or not entries:
        raise ValueError("the staff list is not a list of people")
    staff: list[Staff] = []
    for position, entry in enumerate(entries, start=1):
        person = _person(position, entry)
        if any(known.phone == person.phone for known in staff):
            raise ValueError(f"the staff list has entry {position}'s phone twice")
        staff.append(person)
    return tuple(staff)


def _person(position: int, entry: Any) -> Staff:
    where = f"entry {position} of the staff list"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} is not an object")
    phone = _phone(entry.get("phone"))
    if phone is None:
        raise ValueError(f"{where} has no valid phone in international format")
    name = entry.get("name")
    name = name.strip() if isinstance(name, str) else ""
    if not name or len(name) > MAX_NAME:
        raise ValueError(f"{where} needs a name of 1 to {MAX_NAME} characters")
    role = entry.get("role")
    if not isinstance(role, str) or role not in set(Role):
        raise ValueError(f"{where} has a role that is not one of: {', '.join(Role)}")
    return Staff(phone, name, Role(role))


def _phone(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        number = phonenumbers.parse(value, None)
    except NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, PhoneNumberFormat.E164)
