"""What an answer to each booking step may be, and the phrases (M5)."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from frontdesk.booking.flow import (
    _PHRASES,
    entry_description,
    parse_choice,
    phrase,
    valid_name,
    valid_phone,
    yes_or_no,
)

TIRANE = ZoneInfo("Europe/Tirane")
OFFERED = (
    datetime(2026, 10, 23, 9, tzinfo=TIRANE),
    datetime(2026, 10, 23, 10, tzinfo=TIRANE),
    datetime(2026, 10, 26, 12, tzinfo=TIRANE),
)


@pytest.mark.parametrize(
    ("message", "index"),
    [
        ("2", 1),
        (" 3 ", 2),
        ("il 2", 1),
        ("numero 1", 0),
        ("the 2nd", 1),
        ("1)", 0),
        ("alle 10:00", 1),
        ("10.00 va bene", 1),
        ("12:00", 2),
        ("4", None),  # three were offered
        ("Ho 2 domande", None),
        ("Quanto costa?", None),
        ("alle 11:00", None),  # not offered
        ("9:00 o 10:00", None),  # two times
    ],
)
def test_a_choice_is_a_number_or_an_offered_time(message: str, index: int | None) -> None:
    expected = None if index is None else OFFERED[index]
    assert parse_choice(message, OFFERED, TIRANE) == expected


@pytest.mark.parametrize(
    ("message", "name"),
    [
        ("Mario Rossi", "Mario Rossi"),
        ("  Ana   Marku ", "Ana Marku"),
        ("Arbër Çela", "Arbër Çela"),
        ("D'Angelo-Rossi", "D'Angelo-Rossi"),
        ("O\u2019Brien", "O\u2019Brien"),  # a typographic apostrophe
        ("M", None),
        ("Mario 3", None),
        ("mario@example.com", None),
        ("x" * 61, None),
        ("Vorrei sapere se posso portare mia figlia?", None),
    ],
)
def test_a_name_is_letters_with_spaces_or_marks_between(message: str, name: str | None) -> None:
    assert valid_name(message) == name


@pytest.mark.parametrize(
    ("message", "phone"),
    [
        ("+355 69 123 4567", "+355691234567"),
        ("069-123-4567", "+355691234567"),  # a national number takes the clinic's country code
        ("0039 333 123 4567", "+393331234567"),
        ("(+39) 333.123.4567", "+393331234567"),
        ("3331234567", None),  # neither a + nor a national 0
        ("+1234567", None),  # too short
        ("+1234567890123456", None),  # too long
        ("il mio numero è 0691234567", None),
    ],
)
def test_a_phone_is_international_with_8_to_15_digits(message: str, phone: str | None) -> None:
    assert valid_phone(message, "+355") == phone


@pytest.mark.parametrize(
    ("message", "answer"),
    [
        ("Sì!", True),
        ("po", True),
        ("Yes", True),
        ("ok", True),
        ("No.", False),
        ("jo", False),
        ("annulla", False),
        ("forse", None),
    ],
)
def test_yes_or_no_in_the_three_languages(message: str, answer: bool | None) -> None:
    assert yes_or_no(message) is answer


def test_every_phrase_exists_in_every_language_and_the_entry_too() -> None:
    assert all(set(texts) == {"sq", "it", "en"} for texts in _PHRASES.values())
    assert phrase("choose", "it") == "Quale preferisce? Può scrivere il numero o il giorno."
    assert (
        entry_description("+355691234567", "it") == "Telefono: +355691234567\nPrenotato dalla chat."
    )
