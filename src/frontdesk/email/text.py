"""What the model reads of an email: the sender's new text without the quoted messages, cut at
MAX_CHARS, and the quoted part apart, as context (design §3.3).

A reply quotes the conversation below a line such as "On ... wrote:" or with lines that start with
">". The new text is what the sender wrote above them; the rest is context, never the knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

MAX_CHARS: Final = 6000  # the sender's text the model reads; longer is cut and flagged (§4)
CONTEXT_CHARS: Final = 1500  # the quoted or earlier messages it reads, in short

# The line a mail client writes above the message it quotes, in the engine's languages.
_QUOTE_HEADER: Final = re.compile(
    r"^\s*(?:"
    r"On\s.{1,200}\bwrote:"
    r"|Il\s.{1,200}\bha\s+scritto:"
    r"|Në\s.{1,200}\bshkroi:"
    r"|Ne\s.{1,200}\bshkroi:"
    r"|-{2,}\s*(?:Original Message|Messaggio originale|Mesazhi origjinal)\s*-{2,}"
    r"|_{10,}"
    r"|(?:From|Da|Nga):\s.+"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EmailText:
    new: str  # what the sender wrote, at most MAX_CHARS
    quoted: str  # the messages it quotes, at most CONTEXT_CHARS
    truncated: bool  # the new text was longer than MAX_CHARS


def split_quotes(body: str) -> EmailText:
    """The sender's new text and the quoted part of `body`."""
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    new: list[str] = []
    quoted: list[str] = []
    in_quote = False
    for line in lines:
        if not in_quote and _QUOTE_HEADER.match(line):
            in_quote = True
        if in_quote or line.lstrip().startswith(">"):
            quoted.append(line.lstrip().lstrip(">").strip())
            continue
        new.append(line)
    text = _tidy("\n".join(new))
    return EmailText(
        new=text[:MAX_CHARS],
        quoted=_tidy("\n".join(quoted))[:CONTEXT_CHARS],
        truncated=len(text) > MAX_CHARS,
    )


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def shorten(text: str, limit: int) -> str:
    """At most `limit` characters, cut at a word and marked when cut."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"
