"""Text normalization shared by safety rules, language detection and full-text search."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize(text: str) -> str:
    """Lower-case, no diacritics, punctuation to spaces: 'Vizitë?' -> 'vizite'."""
    return _NON_ALNUM.sub(" ", strip_diacritics(text).lower()).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()
