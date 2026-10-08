"""Deterministic rules that run before any model call (spec §6, first level).

Emergencies, explicit requests for medical advice and personal data stop here.
Emergencies match as phrases and as word stems close to each other ("dolore
forte al petto"), so a few words in between do not hide them; the classifier is
a second net for what the rules miss (ADR-0006). Symptom mentions alone pass on
to the classifier, which may route them to orientation; advice patterns stay
narrow so practical requests ("can I take an appointment?") are not blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from frontdesk.safety.normalize import normalize


class Verdict(StrEnum):
    EMERGENCY = "emergency"
    CLINICAL_ADVICE = "clinical_advice"
    PERSONAL_DATA = "personal_data"
    PASS = "pass"  # noqa: S105 - a verdict, not a password


class PersonalField(StrEnum):
    PHONE = "phone"
    EMAIL = "email"
    NAME = "name"
    IDENTITY_DOCUMENT = "identity_document"
    BIRTH_DATE = "birth_date"


@dataclass(frozen=True)
class RuleResult:
    verdict: Verdict
    rule: str = "none"


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")")


def _near(first: str, second: str, *, gap: int = 3) -> str:
    """Two word stems at most `gap` words apart, in either order."""
    between = rf"\w*(?: \w+){{0,{gap}}} "
    return rf"(?:{first}){between}(?:{second})|(?:{second}){between}(?:{first})"


_MEDICINES = (
    r"(?:pill|tablet|medicine|medication|antibiotic|painkiller|ibuprofen|paracetamol"
    r"|farmac|medicin|antidolorific|antinfiammatori|pastiglia|compressa|tachipirina"
    r"|ilac|bar|antibiotik)"
)

# Word stems in normalized text (lower case, no diacritics, one space between words).
# A stem matches the start of a word; `\b` makes it a whole word, as for the
# Albanian negation "s" in "s'po marr frymë".
_PAIN = r"pain|hurt|ache|aching|tight|pressure|dolor|male\b|fitt|oppress|dhimbj|dhemb|shtreng"
_CHEST = r"chest|petto\b|torace|gjoks"
_BREATH = r"breath|respir|fiato|frym"
_CANNOT = (
    r"can ?t\b|cannot|can not|could ?n ?t\b|could not|unable|hard to|struggl|trouble|difficult"
    r"|short|gasp|non riesc|non riusc|fatic|difficolt|manca|affann|nuk\b|s\b|veshtires|zen|mezi"
)
_BLOOD = r"bleed|blood(?! test| pressure)|sangu|gjak(?!ut)"
_MUCH = r"heav|lots?\b|much|severe|stop|profus|molto|tanto|forte|abbondant|ferma|shume|ndal|rrjedh"

EMERGENCY: Final = _any(
    _near(_PAIN, _CHEST),
    _near(_CANNOT, _BREATH),
    _near(_BLOOD, _MUCH),
    r"not breathing",
    r"non respira",
    r"heart attack",
    r"cardiac arrest",
    r"arresto cardiaco",
    r"infart",
    r"infarkt",
    r"stroke",
    r"ictus",
    r"goditje ne tru",
    r"paraliz",
    r"unconscious",
    r"passed out",
    r"faint",
    r"collaps",
    r"unresponsive",
    r"not responding",
    r"sven",
    r"svien",
    r"incoscient",
    r"(?:perso|perde) (?:i sensi|conoscenza)",
    r"pa ndjenja",
    r"humbi ndjenjat",
    r"te fiket",
    r"seizure",
    r"convuls",
    r"epilept",
    r"epilett",
    r"konvuls",
    r"anaphyla",
    r"anafila",
    r"emorragia",
    r"gjakderdhje",
    r"overdose",
    r"poison",
    r"avvelen",
    r"intossic",
    r"helmim",
    r"helmua",
    r"too many pills",
    r"(?:troppe|troppi) (?:pastiglie|compresse|pillole|farmaci)",
    r"shume (?:tableta|hape|ilace)",
    r"suicid",
    r"kill myself",
    r"end my life",
    r"want to die",
    r"self harm",
    r"togliermi la vita",
    r"farla finita",
    r"voglio morire",
    r"vetevras",
    r"vras veten",
    r"dua te vdes",
    r"emergency",
    r"emergenza",
    r"urgjenc",
    r"emergjenc",
    r"pronto soccorso",
    r"ambulan",
)

CLINICAL_ADVICE: Final = _any(
    r"what do i have",
    r"is it serious",
    rf"(?:can|should) i (?:take|use) (?:a |an |the |some )?{_MEDICINES}",
    r"(?:what|which) (?:medicine|medication|pills?|drug)",
    r"dosage",
    r"how many (?:pills|mg)",
    r"my (?:test |lab |blood )?(?:test )?results",
    r"interpret",
    r"diagnos",
    r"che cosa ho(?! bisogno)",
    r"cosa potrebbe essere",
    r"e grave",
    rf"(?:devo|posso) prendere (?:un |una |il |la |dei |delle )?{_MEDICINES}",
    r"(?:che|quale) (?:farmaco|medicina|medicinale)",
    r"dosaggio",
    r"quante (?:pastiglie|compresse)",
    r"miei (?:esami|risultati|referti)",
    r"interpretar",
    r"cfare kam(?! nevoj)",
    r"cfare mund te jete",
    r"eshte e rende",
    r"eshte serioz",
    rf"(?:duhet|mund) te marr (?:nje )?{_MEDICINES}",
    r"(?:cfare|cilin|cili) ilac",
    r"sa doza",
    r"rezultatet e mia",
    r"analizat e mia",
    r"diagnoz",
)

CLINICAL_TERMS: Final = _any(
    r"pain",
    r"ache",
    r"fever",
    r"cough",
    r"rash",
    r"infection",
    r"bleed",
    r"swollen",
    r"swelling",
    r"nausea",
    r"vomit",
    r"dizz",
    r"cancer",
    r"tumou?r",
    r"symptom",
    r"injur",
    r"illness",
    r"disease",
    r"sick",
    r"hurt",
    r"sore",
    r"faint",
    r"dolor",
    r"mal di",
    r"fa male",
    r"male (?:al|alla|allo|ai|agli|alle)\b",
    r"sven",
    r"febbre",
    r"tosse",
    r"eruzion",
    r"infezion",
    r"sanguin",
    r"gonfi",
    r"vertigin",
    r"tumor",
    r"sintom",
    r"ferit",
    r"malatti",
    r"dhimbj",
    r"dhemb",
    r"ethe",
    r"temperatur",
    r"kolle",
    r"skuqj",
    r"infeksion",
    r"gjakderdh",
    r"enjtj",
    r"marramendj",
    r"kancer",
    r"simptom",
    r"semund",
)

DIAGNOSTIC_LANGUAGE: Final = _any(
    r"you (?:may|might|probably) have",
    r"it (?:could|might|may) be (?:a|an)",
    r"sounds like (?:a|an)",
    r"you should take",
    r"take \d+ ?mg",
    r"potrebbe trattarsi",
    r"probabilmente hai",
    r"dovresti prendere",
    r"si tratta probabilmente",
    r"ndoshta keni",
    r"duket si nje",
    r"duhet te merrni (?:ilac|bar|antibiotik)",
    r"ka shume mundesi te jete",
)

_PHONE: Final = re.compile(r"(?<![\w-])(?:\+|00)?\d(?:[\s().-]?\d){7,}(?!\w)")
_ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_EMAIL: Final = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_NAME: Final = _any(r"my name is", r"mi chiamo", r"il mio nome e", r"quhem", r"emri im")
_IDENTITY: Final = _any(
    r"passport",
    r"passaport",
    r"pasaport",
    r"carta d identita",
    r"codice fiscale",
    r"leternjoftim",
    r"numri personal",
    r"id card",
)
_BIRTH: Final = _any(
    r"date of birth", r"data di nascita", r"datelindj", r"born on", r"nat[oa] il", r"lindur me"
)


def personal_fields(text: str) -> frozenset[PersonalField]:
    normalized = normalize(text)
    found: set[PersonalField] = set()
    if any(not _ISO_DATE.fullmatch(match.group(0).strip()) for match in _PHONE.finditer(text)):
        found.add(PersonalField.PHONE)
    if _EMAIL.search(text):
        found.add(PersonalField.EMAIL)
    if _NAME.search(normalized):
        found.add(PersonalField.NAME)
    if _IDENTITY.search(normalized):
        found.add(PersonalField.IDENTITY_DOCUMENT)
    if _BIRTH.search(normalized):
        found.add(PersonalField.BIRTH_DATE)
    return frozenset(found)


def evaluate(text: str, *, expected: frozenset[PersonalField] = frozenset()) -> RuleResult:
    normalized = normalize(text)
    if EMERGENCY.search(normalized):
        return RuleResult(Verdict.EMERGENCY, "emergency")
    if CLINICAL_ADVICE.search(normalized):
        return RuleResult(Verdict.CLINICAL_ADVICE, "clinical_advice")
    unexpected = personal_fields(text) - expected
    if unexpected:
        return RuleResult(Verdict.PERSONAL_DATA, sorted(unexpected)[0].value)
    return RuleResult(Verdict.PASS)


def contains_clinical_terms(text: str) -> bool:
    normalized = normalize(text)
    return bool(CLINICAL_TERMS.search(normalized) or CLINICAL_ADVICE.search(normalized))


def contains_diagnostic_language(text: str) -> bool:
    return DIAGNOSTIC_LANGUAGE.search(normalize(text)) is not None
