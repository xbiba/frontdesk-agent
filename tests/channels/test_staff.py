"""The WhatsApp channel's staff list (W1, spec D3): read from its secret, each person with the
WhatsApp ID of their number; a bad list is refused naming the entry, never its number."""

import json
from typing import Any

import pytest

from frontdesk.channels.staff import Role, Staff, read_staff

ALBI = {"phone": "+355 69 123 4567", "name": "Albi", "role": "reception"}


def _list(*entries: Any) -> str:
    return json.dumps(list(entries))


def test_the_staff_list_is_read_with_its_whatsapp_ids() -> None:
    doctor = {"phone": "+39 333 123 4567", "name": "Dr. Salvatore Oliverio", "role": "doctor"}
    albi, oliverio = read_staff(_list(ALBI, doctor))
    assert albi == Staff("+355691234567", "Albi", Role.RECEPTION)
    assert albi.user_id == "355691234567"
    assert (oliverio.role, oliverio.user_id) == (Role.DOCTOR, "393331234567")


@pytest.mark.parametrize(
    "second",
    [
        pytest.param("+355 69 765 4321", id="not-an-object"),
        pytest.param({"phone": "+355 12", "name": "Ana", "role": "reception"}, id="bad-phone"),
        pytest.param({"phone": "+355 69 765 4321", "name": " ", "role": "reception"}, id="no-name"),
        pytest.param(
            {"phone": "+355 69 765 4321", "name": "A" * 41, "role": "reception"}, id="long-name"
        ),
        pytest.param({"phone": "+355 69 765 4321", "name": "Ana", "role": "boss"}, id="role"),
        pytest.param({**ALBI, "name": "Albi again"}, id="phone-twice"),
    ],
)
def test_a_bad_staff_list_is_refused_naming_the_entry(second: Any) -> None:
    with pytest.raises(ValueError, match="entry 2") as refused:
        read_staff(_list(ALBI, second))
    message = str(refused.value)
    assert "765" not in message
    assert "4567" not in message


@pytest.mark.parametrize("raw", ["not json", '{"phone": "+355 69 123 4567"}', "[]"])
def test_a_staff_list_that_is_no_list_of_people_is_refused(raw: str) -> None:
    with pytest.raises(ValueError, match="staff list"):
        read_staff(raw)
