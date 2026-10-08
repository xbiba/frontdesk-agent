import re

from hypothesis import given
from hypothesis import strategies as st

from frontdesk.safety.normalize import normalize, strip_diacritics, tokens


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
