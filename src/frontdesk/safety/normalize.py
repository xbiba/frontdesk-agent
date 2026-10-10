"""Text normalization shared by safety rules, language detection and full-text search."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def strip_invisible(text: str) -> str:
    """`text` without format characters (Unicode category Cf).

    Soft hyphens, zero-width spaces and joiners, word joiners, direction marks and byte-order marks
    print nothing, yet copied or obfuscated text carries them inside words. Left in, they would
    split "pain" with a soft hyphen inside into two words that no rule reads (final review P34).
    """
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def space_invisible(text: str) -> str:
    """`text` with each format character (Unicode category Cf) read as a space.

    The other reading of an invisible character. Removed (`strip_invisible`), it keeps a word whole;
    but when it is the only separator between two words ("chest", a zero-width space, "pain"),
    removing it joins them into one word that no rule reads. `normalize` kept such words apart
    until it began to remove these characters. The rules read both forms.
    """
    return "".join(" " if unicodedata.category(char) == "Cf" else char for char in text)


def fold(text: str) -> str:
    """`text` as the patterns that read raw text see it: invisible characters removed, then
    compatibility forms folded (NFKC), so that a fullwidth at sign is an "@" (final review P57)."""
    return unicodedata.normalize("NFKC", strip_invisible(text))


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize(text: str) -> str:
    """Lower-case, no diacritics, no invisible characters, punctuation to spaces:
    'Vizitë?' -> 'vizite'."""
    return _NON_ALNUM.sub(" ", strip_diacritics(strip_invisible(text)).lower()).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()
