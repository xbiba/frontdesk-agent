"""Split Markdown knowledge documents into heading-aware chunks (spec §7)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    heading: str
    content: str


def chunk_markdown(title: str, body: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """One chunk per section; long sections split on paragraph boundaries.

    Each chunk keeps its heading path ("Title > Section"), so a chunk read alone
    still says what it is about.
    """
    chunks: list[Chunk] = []
    for heading, text in _sections(body):
        path = f"{title} > {heading}" if heading else title
        for piece in _split(text, max_chars):
            chunks.append(Chunk(ordinal=len(chunks), heading=path, content=piece))
    return chunks


def _sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append((match.group(2), []))
        else:
            sections[-1][1].append(line)
    joined = [(heading, "\n".join(lines).strip()) for heading, lines in sections]
    return [(heading, text) for heading, text in joined if text]


def _split(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in _PARAGRAPH_BREAK.split(text)):
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            pieces.extend(_hard_split(paragraph, max_chars))
            current = ""
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def _hard_split(paragraph: str, max_chars: int) -> list[str]:
    parts = (paragraph[i : i + max_chars].strip() for i in range(0, len(paragraph), max_chars))
    return [part for part in parts if part]
