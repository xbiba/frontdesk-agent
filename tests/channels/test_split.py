"""Replies cut to a channel's size (M4 design D9): parts of at most the channel's bytes, cut
between lines, sentences, words and, last, characters."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from frontdesk.channels.split import split_message


def _bytes(part: str) -> int:
    return len(part.encode("utf-8"))


def test_a_short_reply_is_one_part() -> None:
    assert split_message("Ciao! Come posso aiutarla?", 1000) == ["Ciao! Come posso aiutarla?"]


def test_a_blank_reply_has_no_part() -> None:
    assert split_message(" \n \n", 1000) == []


def test_parts_break_between_lines_first() -> None:
    lines = [f"Riga {n}: " + "x" * 40 for n in range(30)]
    parts = split_message("\n".join(lines), 200)
    assert all(_bytes(part) <= 200 for part in parts)
    assert "\n".join(parts).split("\n") == lines


def test_a_long_sentence_breaks_between_words() -> None:
    text = " ".join(["parola"] * 100)
    parts = split_message(text, 100)
    assert all(_bytes(part) <= 100 for part in parts)
    assert " ".join(parts).split() == text.split()


def test_a_word_longer_than_a_part_breaks_between_characters() -> None:
    url = "https://www.clinicamcu.com/" + "a" * 300
    parts = split_message(url, 100)
    assert "".join(parts) == url
    assert all(_bytes(part) <= 100 for part in parts)


def test_characters_of_two_bytes_count_as_two() -> None:
    parts = split_message("è" * 600, 1000)
    assert [_bytes(part) for part in parts] == [1000, 200]


def test_a_part_must_hold_the_longest_character() -> None:
    with pytest.raises(ValueError, match="at least 4 bytes"):
        split_message("ciao", 3)


@given(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=3000),
    st.integers(min_value=4, max_value=1200),
)
def test_no_part_is_too_long_and_nothing_is_lost(text: str, max_bytes: int) -> None:
    parts = split_message(text, max_bytes)
    assert all(0 < _bytes(part) <= max_bytes for part in parts)
    assert "".join("".join(parts).split()) == "".join(text.split())
