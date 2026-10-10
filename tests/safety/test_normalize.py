import re

from hypothesis import given
from hypothesis import strategies as st

from frontdesk.safety.normalize import fold, normalize, strip_diacritics, tokens


def test_albanian_and_italian_diacritics_are_removed() -> None:
    assert strip_diacritics("Vizitë çmimi è perché") == "Vizite cmimi e perche"


def test_normalize_lowers_and_drops_punctuation() -> None:
    assert normalize("  Sa kushton një VIZITË?!  ") == "sa kushton nje vizite"


def test_tokens_split_on_spaces() -> None:
    assert tokens("Dr. Arben-Demo, 30€") == ["dr", "arben", "demo", "30"]


def test_runs_of_separators_collapse_to_one_space() -> None:
    assert normalize("a  -  b\t\nc") == "a b c"


def test_compatibility_forms_are_folded() -> None:
    ligature = "\N{LATIN SMALL LIGATURE FI}"
    fullwidth_30 = "\N{FULLWIDTH DIGIT THREE}\N{FULLWIDTH DIGIT ZERO}"
    assert normalize(f"{ligature}sio {fullwidth_30}") == "fisio 30"


@given(st.text())
def test_normalize_output_is_clean_and_stable(text: str) -> None:
    result = normalize(text)
    assert re.fullmatch(r"(?:[a-z0-9]+(?: [a-z0-9]+)*)?", result)
    assert normalize(result) == result
    assert tokens(text) == result.split()


INVISIBLE = [
    "\N{SOFT HYPHEN}",
    "\N{ZERO WIDTH SPACE}",
    "\N{ZERO WIDTH NON-JOINER}",
    "\N{ZERO WIDTH JOINER}",
    "\N{WORD JOINER}",
    "\N{ZERO WIDTH NO-BREAK SPACE}",
    "\N{LEFT-TO-RIGHT MARK}",
]


def test_invisible_characters_inside_a_word_keep_the_word_whole() -> None:
    assert normalize("che\N{ZERO WIDTH SPACE}st pa\N{SOFT HYPHEN}in") == "chest pain"
    assert tokens("fizio\N{WORD JOINER}terapia") == ["fizioterapia"]


def test_fold_turns_fullwidth_forms_into_ascii_and_drops_invisible_characters() -> None:
    assert fold("mario\N{FULLWIDTH COMMERCIAL AT}example.com") == "mario@example.com"
    assert fold("333\N{ZERO WIDTH SPACE}123\N{ZERO WIDTH SPACE}4567") == "3331234567"


@given(st.text(), st.integers(min_value=0), st.sampled_from(INVISIBLE))
def test_an_invisible_character_anywhere_changes_nothing(
    text: str, position: int, invisible: str
) -> None:
    at = position % (len(text) + 1)
    assert normalize(text[:at] + invisible + text[at:]) == normalize(text)
