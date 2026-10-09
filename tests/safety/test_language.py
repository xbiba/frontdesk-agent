import pytest

from frontdesk.safety.language import detect_language
from tests.safety.test_rules import EMERGENCIES

ALL = ["sq", "it", "en"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sa kushton një vizitë?", "sq"),
        ("sa kushton nje vizite?", "sq"),
        ("Përshëndetje, ku ndodhet klinika?", "sq"),
        ("Quanto costa una visita?", "it"),
        ("how much is a visit?", "en"),
        ("pershendetje, quanto costa la visita? grazie", "it"),
    ],
)
def test_detects_the_language(text: str, expected: str) -> None:
    assert detect_language(text, current=None, supported=ALL, default="sq") == expected


@pytest.mark.parametrize("reply", ["ok", "OK!", "grazie", "faleminderit", "po", "si", "👍"])
def test_short_replies_keep_the_current_language(reply: str) -> None:
    assert detect_language(reply, current="it", supported=ALL, default="sq") == "it"


def test_unsupported_language_falls_back() -> None:
    assert (
        detect_language("Sa kushton dergesa?", current=None, supported=["it", "en"], default="it")
        == "it"
    )


def test_ambiguous_text_keeps_the_current_language() -> None:
    assert detect_language("ORD-10427 20121", current="en", supported=ALL, default="sq") == "en"


@pytest.mark.parametrize("default", ["sq", "it"])
@pytest.mark.parametrize(
    ("text", "language"),
    [(text, language) for language, texts in EMERGENCIES.items() for text in texts],
)
def test_emergencies_are_answered_in_their_language(text: str, language: str, default: str) -> None:
    assert detect_language(text, current=None, supported=ALL, default=default) == language
