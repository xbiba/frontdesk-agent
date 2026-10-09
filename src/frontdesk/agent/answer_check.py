"""Checks on every document answer before it can be sent (ADR-0009).

Each claim must cite a passage that was provided, quote it word for word, take
every number and every link or address from the quote, name no price, avoid
diagnostic language and stay close in meaning to its quote. Facts from records
never come through here: code writes them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

from frontdesk.agent.number_words import spoken_numbers
from frontdesk.agent.responder import AnswerAction, Claim, ResponderOutput
from frontdesk.knowledge.embedders import Embedder, cosine
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.safety.rules import contains_diagnostic_language
from frontdesk.tenancy.pack import Language

MAX_ANSWER_CHARS: Final = 600

_TIME: Final = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
# A thousands group ("1.000", "2,500", "1.000.000") comes first: the dot is a thousands separator in
# Italian and Albanian and a decimal point in English, and the comma the reverse, so it is compared
# as written and never taken for the number before the separator.
_GROUPED: Final = re.compile(r"[1-9]\d{0,2}(?:[.,]\d{3})+")
_NUMBER: Final = re.compile(rf"{_GROUPED.pattern}(?!\d)|\d+(?:[.,]\d+)?")
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

# Soft hyphens and zero-width characters, which copied text carries inside words.
_INVISIBLE: Final = str.maketrans(dict.fromkeys((0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)))
# A word, or a number with its separators ("14", "10:30", "1.000"), or words joined by a hyphen
# ("non-refundable", "e-mail"): a quote may not start or end inside one of these.
_TOKEN: Final = re.compile(r"[^\W_]+(?:(?:[-\U00002013]|(?<=\d)[.,:/](?=\d))[^\W_]+)*")
# Where a negation stops reaching the quote: a sentence, clause, bracket or list item ends.
_CLAUSE_BREAK: Final = re.compile(r"[.!?;:,()\U00002022]| [-\U00002013\U00002014*] ")
_LEAD_WORD: Final = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*'?")
# Lower case with accents, so that Italian "né" (nor) is not the pronoun "ne". The words belong to a
# language: a passage is read with the negation words of its own. Two forms are rules rather than
# words: Albanian "s'" ("s'ka": there is not), which is also the Italian elision ("s'intende"), and
# English "n't". Each applies to its language only.
_NEGATIONS: Final[dict[Language, str]] = {
    "en": "not no never without none nothing nobody neither nor cannot",
    "it": "non né nè mai senza nessun nessuno nessuna niente nulla neanche nemmeno neppure",
    "sq": "nuk mos jo pa asnjë asnje askush asgjë asgje kurrë kurre",
}
_NEGATION_WORDS: Final = {
    language: frozenset(words.split()) for language, words in _NEGATIONS.items()
}
# "no" abbreviates "number" after a word that takes one and before the number ("Order no 12"). Only
# there it is not a negation; "no 24-hour service" and "no 5 spaces" still are.
_NUMBER_LABELS: Final = frozenset(
    {
        "order",
        "invoice",
        "receipt",
        "reference",
        "ref",
        "item",
        "article",
        "ticket",
        "tracking",
        "parcel",
        "shipment",
        "account",
        "customer",
        "booking",
        "appointment",
        "room",
        "tel",
        "phone",
    }
)
_NUMBER_AHEAD: Final = re.compile(r"\s*\d")
# A web address (any scheme, or www.), an email address, or the "](" of a markdown link. An address
# ends at a space or a closing bracket, so the brackets of a markdown link are told apart from it.
_LINK: Final = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|www\.)[^\s\]<>\"]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\]\(", re.IGNORECASE
)
_AFTER_LINK: Final = ".,;:!?)}'\"\u2026\u00bb"  # punctuation that follows a link, not part of it
NEGATION_REACH: Final = 4  # words before the quote, in its clause, that are searched for one


@dataclass(frozen=True)
class AnswerCheck:
    ok: bool
    text: str
    cited: tuple[Evidence, ...]
    reasons: tuple[str, ...]
    invalid_ids: int


def numbers_in(text: str) -> set[str]:
    """Every number in `text`: digits (times as HH:MM) and words, "fourteen" and "14" alike."""
    found = {f"{int(hour):02d}:{minute}" for hour, minute in _TIME.findall(text)}
    found |= {_canonical(number) for number in _NUMBER.findall(_TIME.sub(" ", text))}
    return found | spoken_numbers(text)


def mentions_price(text: str) -> bool:
    return bool(_CURRENCY.search(text) or _LEK_CODE.search(text))


def links_in(text: str) -> set[str]:
    """The web addresses, email addresses and markdown links ("](") in `text`, compared by `==`.

    Contacts are written by code, never by the model, so a claim may only repeat the ones its quote
    holds. Case does not matter; the punctuation after an address is not part of it.
    """
    return {
        found.rstrip(_AFTER_LINK).casefold() if found != "](" else found
        for found in _LINK.findall(_flat(text))
    }


def quote_in(quote: str, source: str, language: Language | None = None) -> bool:
    """The quote is in the source word for word, and was not cut out of a word or a negation.

    Spacing and quote marks do not matter. The quote may not start or end inside a word or a number
    ("4 giorni" is not in "14 giorni", "available" is not in "unavailable"), and the words just
    before it may not hold a negation ("accept cards" is not in "we do not accept cards"). A quote
    found more than once must pass at every place. The negation words are those of the source's
    `language`; when it is not known, the words of all three languages count.
    """
    flat = _flat(quote)
    # An empty string is "in" every string, so a quote with nothing in it would pass for any passage.
    if not flat:
        return False
    haystack = _flat(source)
    starts = _occurrences(haystack, flat)
    languages = tuple(_NEGATION_WORDS) if language is None else (language,)
    return bool(starts) and not any(
        _cuts_a_token(haystack, start, start + len(flat))
        or _negated_before(haystack, start, languages)
        for start in starts
    )


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
    if not quote_in(claim.quote, passage.content, passage.language):
        problems.append("quote_not_in_source")
    if numbers_in(text) - numbers_in(claim.quote):
        problems.append("number_not_in_quote")
    if links_in(text) - links_in(claim.quote):
        problems.append("link_not_in_quote")
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
    composed = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    return " ".join(composed.translate(_QUOTE_MARKS).split())


def _occurrences(haystack: str, needle: str) -> list[int]:
    found: list[int] = []
    position = haystack.find(needle)
    while position != -1:
        found.append(position)
        position = haystack.find(needle, position + 1)
    return found


def _cuts_a_token(haystack: str, start: int, end: int) -> bool:
    return any(
        token.start() < cut < token.end()
        for token in _TOKEN.finditer(haystack)
        for cut in (start, end)
    )


def _negated_before(haystack: str, start: int, languages: Collection[Language]) -> bool:
    clause = _CLAUSE_BREAK.split(haystack[:start])[-1]
    clause_start = start - len(clause)
    words = list(_LEAD_WORD.finditer(clause))
    return any(
        _is_negation(words[index].group().lower(), languages)
        and not _abbreviates_number(haystack, clause_start, words, index)
        for index in range(max(0, len(words) - NEGATION_REACH), len(words))
    )


def _abbreviates_number(
    haystack: str, clause_start: int, words: Sequence[re.Match[str]], index: int
) -> bool:
    """Whether `words[index]` is the "no" of "Order no 12": after a word that takes a number, and
    followed by one. Anywhere else "no" negates."""
    if index == 0 or words[index].group().lower() != "no":
        return False
    label = words[index - 1].group().lower()
    ahead = _NUMBER_AHEAD.match(haystack, clause_start + words[index].end())
    return label in _NUMBER_LABELS and ahead is not None


def _is_negation(word: str, languages: Collection[Language]) -> bool:
    return any(
        word in _NEGATION_WORDS[language]
        or (language == "sq" and word.startswith("s'"))
        or (language == "en" and word.endswith("n't"))
        for language in languages
    )


def _canonical(number: str) -> str:
    normalized = number.replace(",", ".")
    if _GROUPED.fullmatch(number):
        return normalized
    integer, _, fraction = normalized.partition(".")
    integer = integer.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{integer}.{fraction}" if fraction else integer
