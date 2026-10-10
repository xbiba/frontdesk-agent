"""The negations a text holds, in the words of its language (the review of document answers).

A reply that adds a negation the expected answer lacks ("not at night", "s'ka parkim") is sent to the
reviewer. The words and the two contractions are those the quote check refuses a quote after
(`test_quote_matching.py`), so a contraction cannot slip past the review.
"""

import pytest

from frontdesk.agent.answer_check import negations_in
from frontdesk.tenancy.pack import Language

APOSTROPHE = chr(0x2019)
SOFT_HYPHEN = chr(0x00AD)
ZERO_WIDTH_SPACE = chr(0x200B)


def _words(text: str) -> list[str]:
    return text.split()


WORDS: dict[Language, list[str]] = {
    "en": _words("not no non never without none nothing nobody neither nor cannot"),
    "it": _words(
        "non no né nè mai senza nessun nessuno nessuna niente nulla neanche nemmeno neppure"
    ),
    "sq": _words("nuk mos jo pa asnjë asnje askush asgjë asgje kurrë kurre"),
}
WORD_CASES = [
    pytest.param(language, word, id=f"{language}-{word}")
    for language, words in WORDS.items()
    for word in words
]
OTHER_LANGUAGE_CASES = [
    pytest.param(language, word, id=f"{language}-not-{other}-{word}")
    for language in WORDS
    for other in WORDS
    if other != language
    for word in WORDS[other]
    if word not in WORDS[language]
]


@pytest.mark.parametrize(("language", "word"), WORD_CASES)
def test_every_negation_word_of_a_language_is_found(language: Language, word: str) -> None:
    assert negations_in(f"The clinic {word} is open", language) == {word}


@pytest.mark.parametrize(("language", "word"), OTHER_LANGUAGE_CASES)
def test_a_word_that_negates_in_another_language_is_none_here(
    language: Language, word: str
) -> None:
    assert negations_in(f"The clinic {word} is open", language) == set()


def test_case_does_not_matter_and_each_negation_is_returned_once() -> None:
    assert negations_in("NON è aperto. Non di notte, mai!", "it") == {"non", "mai"}


def test_a_text_without_a_negation_holds_none() -> None:
    assert negations_in("Ci sono sei posti auto.", "it") == set()
    assert negations_in("", "en") == set()


@pytest.mark.parametrize("word", _words("ne nota note now nuke pasta jolly moscow nonna nobel"))
@pytest.mark.parametrize("language", ["en", "it", "sq"])
def test_words_that_only_look_like_a_negation_are_not_found(language: Language, word: str) -> None:
    assert negations_in(f"The clinic {word} is open", language) == set()


@pytest.mark.parametrize(
    ("text", "found"),
    [
        pytest.param("We don't take cards", {"don't"}, id="dont"),
        pytest.param("It isn't free and we can't promise", {"isn't", "can't"}, id="two"),
        pytest.param(f"Sorry, we can{APOSTROPHE}t take cards", {"can't"}, id="typographic"),
        pytest.param("We won't open on Sunday", {"won't"}, id="wont"),
    ],
)
def test_the_english_contraction_is_a_negation_in_english_text_only(
    text: str, found: set[str]
) -> None:
    assert negations_in(text, "en") == found
    assert negations_in(text, "it") == set()
    assert negations_in(text, "sq") == set()


@pytest.mark.parametrize(
    ("text", "found"),
    [
        pytest.param("S'ka parkim falas për pacientët", {"s'ka"}, id="s-ka"),
        pytest.param(f"S{APOSTROPHE}ka parkim falas", {"s'ka"}, id="typographic"),
        pytest.param("Nuk ka parkim dhe s'ka vend", {"nuk", "s'ka"}, id="with-nuk"),
    ],
)
def test_the_albanian_contraction_is_a_negation_in_albanian_text_only(
    text: str, found: set[str]
) -> None:
    assert negations_in(text, "sq") == found
    assert negations_in(text, "en") == set()


def test_the_italian_elision_is_not_an_albanian_negation() -> None:
    # "s'intende" is Italian ("is meant"); only text in Albanian reads "s'" as a negation.
    assert negations_in("Per visita s'intende un controllo", "it") == set()


def test_an_invisible_character_does_not_hide_a_negation() -> None:
    assert negations_in(f"{SOFT_HYPHEN}non è aperto", "it") == {"non"}
    assert negations_in(f"no{ZERO_WIDTH_SPACE}n è aperto", "it") == {"non"}
