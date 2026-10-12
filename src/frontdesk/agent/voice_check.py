"""Checks on the voice's rewrite of a reply that code wrote (Albi's live chat, 2026-10-12).

The rewrite may change words, never facts. It must keep every number, link, doctor and service of
the draft and add none. It may not name a price the draft does not name, nor add health words, a
diagnosis or a promise. When the draft states facts from records, it may not add or drop a negation,
which could turn a fact around. Any problem keeps the code's text.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Final

from frontdesk.agent.answer_check import (
    links_in,
    mentions_price,
    negations_in,
    numbers_in,
    to_plain_text,
)
from frontdesk.safety.rules import (
    contains_clinical_terms,
    contains_diagnostic_language,
    promise_phrase,
)
from frontdesk.tenancy.pack import Language

MIN_ROOM: Final = 240  # characters a rewrite may add to a short draft


def voice_problems(
    draft: str, voiced: str, *, language: Language, names: Collection[str], facts: bool
) -> tuple[str, ...]:
    """Why `voiced` may not replace `draft`; empty when it may. `names`: the doctors and services
    the business has, in the reply's language. `facts`: the draft states facts from records."""
    text = to_plain_text(voiced)
    if not text:
        return ("empty",)
    checks = {
        "too_long": len(text) > len(draft) + max(MIN_ROOM, len(draft) // 2),
        "numbers": numbers_in(text) != numbers_in(draft),
        "links": links_in(text) != links_in(draft),
        "price": mentions_price(text) and not mentions_price(draft),
        "clinical": contains_clinical_terms(text) and not contains_clinical_terms(draft),
        "diagnostic": contains_diagnostic_language(text)
        and not contains_diagnostic_language(draft),
        "promise": promise_phrase(text) is not None and promise_phrase(draft) is None,
        "names": _names(draft, text, names),
        "negation": facts
        and bool(negations_in(text, language)) != bool(negations_in(draft, language)),
    }
    return tuple(reason for reason, failed in checks.items() if failed)


def _names(draft: str, text: str, names: Collection[str]) -> bool:
    """A doctor or a service the draft names is missing, or one it does not name was added."""
    before, after = draft.casefold(), text.casefold()
    return any((name.casefold() in before) != (name.casefold() in after) for name in names)
