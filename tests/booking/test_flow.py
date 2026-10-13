"""What a name, a phone and a yes may be, and the phrases of a booking (M5)."""

import pytest

from frontdesk.booking.flow import (
    _PHRASES,
    affirms,
    entry_description,
    phrase,
    valid_name,
    valid_phone,
)


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
        ("sì, confermo grazie", True),
        ("po", True),
        ("Yes please", True),
        ("va bene", True),
        ("No.", False),
        ("non ancora", False),
        ("jo", False),
        ("aspetta un attimo", False),
        ("quanto costa?", False),
    ],
)
def test_a_yes_is_one_of_its_words_and_no_negation(message: str, answer: bool) -> None:
    assert affirms(message) is answer


def test_every_phrase_exists_in_every_language_and_the_entry_too() -> None:
    assert all(set(texts) == {"sq", "it", "en"} for texts in _PHRASES.values())
    assert phrase("taken", "it") == "Questo orario non è più libero."
    assert (
        entry_description("+355691234567", "it") == "Telefono: +355691234567\nPrenotato dalla chat."
    )
