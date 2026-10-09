"""Checks on every document answer before it can be sent (ADR-0009).

Each claim must cite a passage that was provided, quote it word for word, take
every number from the quote, name no price, avoid diagnostic language and stay
close in meaning to its quote. Facts from records never come through here: code
writes them (Task 20).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from frontdesk.agent.responder import AnswerAction, Claim, ResponderOutput
from frontdesk.knowledge.embedders import Embedder, cosine
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.safety.rules import contains_diagnostic_language

MAX_ANSWER_CHARS: Final = 600

_TIME: Final = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
_NUMBER: Final = re.compile(r"\d+(?:[.,]\d+)?")
_CURRENCY: Final = re.compile(
    r"[€$£]|\b(?:eur|euro|euros|lek|leke|lekë|usd|dollars?)\b", re.IGNORECASE
)
# Case-sensitive and after a number: "ALL our services" is not a price in lek.
_LEK_CODE: Final = re.compile(r"\d\s?ALL\b")
_MARKDOWN: Final = re.compile(r"\*\*|__|`|^[ \t]*#+[ \t]*", re.MULTILINE)
_BLANK_LINES: Final = re.compile(r"\n{3,}")
# Typographic quotes and the angle quotes that render_evidence uses in place of < and >.
_QUOTE_MARKS: Final = str.maketrans(
    {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2039": "<", "\u203a": ">"}
)


@dataclass(frozen=True)
class AnswerCheck:
    ok: bool
    text: str
    cited: tuple[Evidence, ...]
    reasons: tuple[str, ...]
    invalid_ids: int


def numbers_in(text: str) -> set[str]:
    found = {f"{int(hour):02d}:{minute}" for hour, minute in _TIME.findall(text)}
    found |= {_canonical(number) for number in _NUMBER.findall(_TIME.sub(" ", text))}
    return found


def mentions_price(text: str) -> bool:
    return bool(_CURRENCY.search(text) or _LEK_CODE.search(text))


def quote_in(quote: str, source: str) -> bool:
    flat = _flat(quote)
    # An empty string is "in" every string, so a quote with nothing in it would pass for any passage.
    return bool(flat) and flat in _flat(source)


def to_plain_text(text: str) -> str:
    return _BLANK_LINES.sub("\n\n", _MARKDOWN.sub("", text)).strip()


def check_answer(
    output: ResponderOutput,
    evidence: Sequence[Evidence],
    *,
    embedder: Embedder,
    min_similarity: float,
) -> AnswerCheck:
    passages = {item.evidence_id: item for item in evidence if item.kind is EvidenceKind.DOCUMENT}
    known = [claim for claim in output.claims if claim.evidence_id in passages]
    invalid = len(output.claims) - len(known)
    reasons: list[str] = ["unknown_evidence"] if invalid else []
    if output.action is not AnswerAction.ANSWER:
        if output.action is AnswerAction.CLARIFY and output.clarify is None:
            reasons.append("missing_clarify_field")
        return AnswerCheck(not reasons, "", (), tuple(reasons), invalid)
    if not known:
        reasons.append("no_claims")
    for claim in known:
        reasons += _claim_problems(claim, passages[claim.evidence_id])
    if known:
        reasons += _support_problems(known, embedder, min_similarity)
    text = " ".join(to_plain_text(claim.text) for claim in known)
    if len(text) > MAX_ANSWER_CHARS:
        reasons.append("too_long")
    cited = tuple({claim.evidence_id: passages[claim.evidence_id] for claim in known}.values())
    unique = tuple(dict.fromkeys(reasons))
    return AnswerCheck(not unique, text, cited, unique, invalid)


def _claim_problems(claim: Claim, passage: Evidence) -> list[str]:
    text = to_plain_text(claim.text)
    problems: list[str] = []
    if not quote_in(claim.quote, passage.content):
        problems.append("quote_not_in_source")
    if numbers_in(text) - numbers_in(claim.quote):
        problems.append("number_not_in_quote")
    if mentions_price(text):
        problems.append("price_in_answer")
    if contains_diagnostic_language(text):
        problems.append("diagnostic_language")
    return problems


def _support_problems(
    claims: Sequence[Claim], embedder: Embedder, min_similarity: float
) -> list[str]:
    vectors = embedder.embed([claim.text for claim in claims] + [claim.quote for claim in claims])
    count = len(claims)
    weak = any(
        not _is_close(cosine(vectors[i], vectors[count + i]), min_similarity) for i in range(count)
    )
    return ["unsupported_claim"] if weak else []


def _is_close(similarity: float, min_similarity: float) -> bool:
    # `similarity < min_similarity` alone lets through what cannot be compared: every comparison
    # with NaN is False, and a threshold of 0 accepts the 0.0 that a zero vector (text without
    # tokens) gives. A zero or NaN similarity supports nothing, whatever the threshold.
    return similarity > 0 and similarity >= min_similarity


def _flat(text: str) -> str:
    return " ".join(text.translate(_QUOTE_MARKS).split())


def _canonical(number: str) -> str:
    normalized = number.replace(",", ".")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized.lstrip("0") or "0"
