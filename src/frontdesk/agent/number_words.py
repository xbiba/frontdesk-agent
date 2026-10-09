"""Numbers written as words, read as digits (ADR-0009: every number in a claim comes from its quote).

One list for English, Italian and Albanian is applied to every text, whatever its language, because
a claim may translate its quote: "six", "gjashtë" and "6" all read as "6". A compound reads whole
("forty-five", "dyzet e pesë", "quarantacinque" and "45" are one number), so "forty" cannot hide
inside "forty-five".

Words that are also articles, pronouns or verbs count only where a number is certain: after a tens
word ("twenty-one", "njëzet e një") or before a unit of time ("one week", "un'ora", "sei giorni",
"një javë"). On their own these never count: English "one", Italian "un", "uno", "una" and "sei"
(you are), Albanian "një" (a). Italian "due" stays in the list although English "due" exists: an
English claim that says "due to" when its quote does not is refused, which fails closed, while
leaving "due" out would let "due anni" pass for "tre anni".

Not read: ordinals ("second floor"), fractions ("half"), and hundreds or thousands written in
several words ("two hundred" reads as 2 and 100; "duecento" and "dyqind" read as 200).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Final

from frontdesk.safety.normalize import strip_diacritics

# Each list holds its words in the order of their values, in lower case without diacritics.
_UNDER_TWENTY: Final = {
    "en": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen",
    "it": "zero uno due tre quattro cinque sei sette otto nove dieci undici dodici tredici "
    "quattordici quindici sedici diciassette diciotto diciannove",
    "sq": "zero nje dy tre kater pese gjashte shtate tete nente dhjete njembedhjete dymbedhjete "
    "trembedhjete katermbedhjete pesembedhjete gjashtembedhjete shtatembedhjete tetembedhjete "
    "nentembedhjete",
}
_TENS: Final = {
    "en": "twenty thirty forty fifty sixty seventy eighty ninety",
    "it": "venti trenta quaranta cinquanta sessanta settanta ottanta novanta",
    "sq": "njezet tridhjete dyzet pesedhjete gjashtedhjete shtatedhjete tetedhjete nentedhjete",
}
_ALBANIAN_HUNDREDS: Final = (
    "njeqind dyqind treqind katerqind peseqind gjashteqind shtateqind teteqind nenteqind"
)
_BIG: Final = {
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "cento": 100, "mille": 1000, "milione": 1_000_000,
    "mije": 1000, "milion": 1_000_000,
}  # fmt: skip
# Count only after a tens word or before a unit of time.
_AMBIGUOUS: Final = {"one": 1, "un": 1, "uno": 1, "una": 1, "sei": 6, "nje": 1}
_TIME_UNITS: Final = {
    "en": "day days week weeks month months year years hour hours minute minutes",
    "it": "giorno giorni settimana settimane mese mesi anno anni ora ore minuto minuti",
    "sq": "dite jave muaj vit vjet ore minute minuta",
}
# Typographic apostrophes become "'"; soft hyphens and zero-width characters, which copied text
# carries inside words, are dropped so that they cannot hide a number.
_CLEAN: Final = str.maketrans(
    {0x2019: "'", 0x2018: "'", 0x02BC: "'"}
    | dict.fromkeys((0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
)
_PERCENT: Final = re.compile(r"\bper\s*cento\b")  # "per cento" is not a hundred


def _by_value(words: str, first: int, step: int = 1) -> dict[str, int]:
    return {word: first + step * index for index, word in enumerate(words.split())}


def _italian_compounds() -> dict[str, int]:
    """Italian writes 21 to 99, 200 to 900 and 2000 to 9000 as one word: ventuno, duemila."""
    units = _by_value(_UNDER_TWENTY["it"], 0)
    found: dict[str, int] = {}
    for tens, ten in _by_value(_TENS["it"], 20, 10).items():
        for unit, value in units.items():
            if 1 <= value <= 9:
                stem = tens[:-1] if unit in {"uno", "otto"} else tens  # ventuno, ventotto
                found[stem + unit] = ten + value
    for unit, value in units.items():
        if 2 <= value <= 9:
            found[f"{unit}cento"] = value * 100
            found[f"{unit}mila"] = value * 1000
    return found


def _numbers() -> tuple[dict[str, int], dict[str, int]]:
    """The words that count anywhere, and the tens words."""
    plain: dict[str, int] = {}
    tens: dict[str, int] = {}
    for language in ("en", "it", "sq"):
        plain |= _by_value(_UNDER_TWENTY[language], 0)
        tens |= _by_value(_TENS[language], 20, 10)
    plain |= tens | _by_value(_ALBANIAN_HUNDREDS, 100, 100) | _BIG | _italian_compounds()
    return {word: value for word, value in plain.items() if word not in _AMBIGUOUS}, tens


def _any_of(words: Iterable[str]) -> str:
    return "|".join(sorted(map(re.escape, words), key=len, reverse=True))


_PLAIN, _TENS_WORDS = _numbers()
_TIME_UNIT_WORDS: Final = frozenset(
    word for words in _TIME_UNITS.values() for word in words.split()
)
_AFTER_TENS: Final = {
    word: value for word, value in (_PLAIN | _AMBIGUOUS).items() if 1 <= value <= 9
}
_WORD: Final = re.compile(rf"\b(?:{_any_of(_PLAIN)})\b")
# "twenty-one", "twenty one", "dyzet e pesë": a tens word, then its units, apart from a hyphen,
# spaces, or "e" or "and" between them; any other punctuation ends the number.
_COMPOUND: Final = re.compile(
    rf"\b({_any_of(_TENS_WORDS)})(?:\s*-\s*|\s+(?:e\s+|and\s+)?)({_any_of(_AFTER_TENS)})\b"
)
_AMBIGUOUS_NUMBER: Final = re.compile(
    rf"\b({_any_of(_AMBIGUOUS)})[\s'-]+(?:{_any_of(_TIME_UNIT_WORDS)})\b"
)


def spoken_numbers(text: str) -> set[str]:
    """The numbers the words of `text` spell, as digits: {"45"} for "forty-five"."""
    rest = _PERCENT.sub(" ", strip_diacritics(text).lower().translate(_CLEAN))
    found: set[int] = set()

    def keep(read: Callable[[re.Match[str]], int]) -> Callable[[re.Match[str]], str]:
        def replace(match: re.Match[str]) -> str:
            found.add(read(match))
            return " "  # what was read is not read again

        return replace

    rest = _COMPOUND.sub(keep(lambda m: _TENS_WORDS[m[1]] + _AFTER_TENS[m[2]]), rest)
    rest = _AMBIGUOUS_NUMBER.sub(keep(lambda m: _AMBIGUOUS[m[1]]), rest)
    found.update(_PLAIN[word] for word in _WORD.findall(rest))
    return {str(value) for value in found}
