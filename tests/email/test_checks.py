"""The checks in code on a draft, and the information it is checked against."""

from datetime import date
from pathlib import Path

import pytest

from frontdesk.email.checks import (
    Flag,
    contact_and_price_flags,
    normalize,
    quote_in,
    wording_flags,
)
from frontdesk.email.knowledge import index_text, knowledge_text
from frontdesk.tenancy.loader import load_pack

KNOWLEDGE = normalize(
    "Il trattamento costa 100 EUR a seduta. Telefono: +355 69 000 0000. info@example.com"
)


def test_a_quote_is_found_word_for_word() -> None:
    assert quote_in("Il trattamento costa 100 EUR a seduta.", KNOWLEDGE)
    assert quote_in("«il  trattamento costa» … a seduta", KNOWLEDGE)
    assert not quote_in("Il trattamento costa 80 EUR", KNOWLEDGE)
    assert not quote_in("...", KNOWLEDGE)


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        ("Gentile Anna, la fisioterapia c'è. [FIRMA]", []),
        ("Ci scriva qui in chat", [Flag.CHAT_WORDING]),
        ("Purtroppo non offriamo la dialisi.", [Flag.ABSENCE_CLAIM]),
        ("We don't offer dialysis.", [Flag.ABSENCE_CLAIM]),
        ("Nuk ofrojmë këtë shërbim.", [Flag.ABSENCE_CLAIM]),
    ],
)
def test_wording(draft: str, expected: list[Flag]) -> None:
    assert wording_flags(draft) == expected


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        ("Chiami il +355 69 000 0000 o scriva a info@example.com: costa 100 EUR.", []),
        ("Scriva a segreteria@example.com", [Flag.UNKNOWN_CONTACT]),
        ("Chiami il +355 69 111 2222", [Flag.UNKNOWN_CONTACT]),
        ("Costa 80 EUR a seduta", [Flag.UNKNOWN_PRICE]),
        ("Costa € 80", [Flag.UNKNOWN_PRICE]),
        ("Le rispondo a anna@example.com", []),
        ("Giovedì 19 alle 10", []),
        ("Il 19.11.2026 alle 10", []),
    ],
)
def test_contacts_and_prices(draft: str, expected: list[Flag]) -> None:
    assert contact_and_price_flags(draft, KNOWLEDGE, allowed=["anna@example.com"]) == expected


def test_the_information_is_written_from_the_pack(businesses_dir: Path) -> None:
    pack = load_pack(businesses_dir / "demo-clinic")
    text = knowledge_text(pack, date(2026, 10, 7))
    assert pack.location.phone in text
    assert pack.fixed_texts["it"]["emergency"] in text
    assert pack.fixed_texts["it"]["disclosure"] not in text
    index = index_text(pack)
    assert index.startswith("INDEX")
    assert pack.services[0].names["it"] in index
    documents = [item for item in pack.documents if item.valid_until is not None]
    for item in documents:
        assert (item.body in knowledge_text(pack, date(2000, 1, 1))) is (
            item.valid_from is None or item.valid_from <= date(2000, 1, 1)
        )
