"""Replies cut to a channel's size (M4 design D9): Instagram takes at most 1000 bytes of UTF-8 in
one message. Parts break between lines, then sentences, then words; a word longer than a part is
cut between characters. Nothing is lost and nothing is reordered."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final

MIN_PART_BYTES: Final = 4  # the longest UTF-8 character
_SPLITTERS: Final = (re.compile(r"(?<=[.!?])\s+"), re.compile(r"\s+"))


def split_message(text: str, max_bytes: int) -> list[str]:
    """`text` in parts of at most `max_bytes` bytes. A text of whitespace has no part."""
    if max_bytes < MIN_PART_BYTES:
        raise ValueError(f"a part must hold at least {MIN_PART_BYTES} bytes")
    parts: list[str] = []
    current = ""
    for separator, atom in _atoms(text.strip(), max_bytes):
        candidate = f"{current}{separator}{atom}" if current else atom
        if _size(candidate) <= max_bytes:
            current = candidate
        else:
            parts.append(current)
            current = atom
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _atoms(text: str, max_bytes: int) -> Iterator[tuple[str, str]]:
    """The pieces to pack, each with the separator that comes before it."""
    if not text:
        return
    for index, line in enumerate(text.split("\n")):
        yield from _fitted("\n" if index else "", line, max_bytes, 0)


def _fitted(separator: str, text: str, max_bytes: int, level: int) -> Iterator[tuple[str, str]]:
    if _size(text) <= max_bytes:
        yield separator, text
    elif level < len(_SPLITTERS):
        pieces = [piece for piece in _SPLITTERS[level].split(text) if piece]
        for index, piece in enumerate(pieces):
            yield from _fitted(separator if index == 0 else " ", piece, max_bytes, level + 1)
    else:
        yield from _characters(separator, text, max_bytes)


def _characters(separator: str, text: str, max_bytes: int) -> Iterator[tuple[str, str]]:
    chunk = ""
    for char in text:
        if chunk and _size(chunk + char) > max_bytes:
            yield separator, chunk
            separator, chunk = "", char
        else:
            chunk += char
    if chunk:
        yield separator, chunk
