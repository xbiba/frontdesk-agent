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
    r"|short|gasp|non riesc|non riusc|fatic|difficolt|manca|affann|nuk\b|s(?= (?:po|mund|marr|merr|arrij)\b)|veshtires|zen|mezi"
)
# Blood tests are not bleeding: "analisi del sangue" never counts as a blood word.
_BLOOD = (
    r"bleed|blood(?! test| pressure)|gjak(?!ut)"
    r"|(?<!analisi del )(?<!analisi di )(?<!esami del )(?<!esame del )(?<!prelievo del )(?<!prelievo di )sangu"
)
# "how much" and "sa shume" ask for a price or an amount; they do not describe bleeding.
_MUCH = (
    r"heav|lots?\b|(?<!how )much|severe|stop|profus|molto|tanto|forte|abbondant|ferma"
    r"|(?<!sa )shume|ndal|rrjedh"
)

EMERGENCY: Final = _any(
    _near(_PAIN, _CHEST),
    _near(_CANNOT, _BREATH),
    _near(_BLOOD, _MUCH),
    r"(?:not|isn t|wasn t|aren t) breathing",
    r"non respir",
    r"heart attack",
    r"cardiac arrest",
    r"arresto cardiaco",
    r"infart",
    r"infarkt",
    r"stroke",
    r"ictus",
    r"goditje ne tru",
    r"paralis",
    r"paralys",
    r"paraliz",
    r"unconscious",
    r"passed out",
    r"faint",
    r"collaps",
    r"unresponsive",
    r"(?:he|she|they|baby|child|son|daughter|father|mother|friend|patient)(?: s| is| are)? not responding",
    r"sven(?!d)",
    r"svien",
    r"incoscient",
    r"(?:perso|perde) (?:i sensi|conoscenza)",
    r"pa ndjenja",
    r"humbi ndjenjat",
    r"te fiket",
    r"seizure",
    r"convuls",
    r"epileps",
    r"epiless",
    r"epilept",
    r"epilett",
    r"konvuls",
    r"anaphyla",
    r"chok(?:ing|ed|es)",
    r"soffoc(?:o|a|ando|ato|ata|amento)\b",
    r"throat (?:(?:is|s|was|keeps|feels like it s|feels like it is) )?"
    r"(?:clos(?:ing|es|ed)|swell(?:ing|s)|tighten)",
    r"gola (?:che )?(?:mi )?(?:si )?(?:sta )?(?:chiud|serr)",
    r"(?:chiude|chiudendo|serra) la gola",
    r"anafila",
    r"emorragia",
    r"gjakderdhje",
    r"overdose",
    r"overdoz",
    r"poison",
    r"avvelen",
    r"intossic",
    r"helmim",
    r"helmua",
    r"too many (?:pills|tablets|meds|medicines|medications|painkillers|drugs)",
    r"(?:troppe|troppi) (?:pastiglie|compresse|pillole|farmaci|medicine|medicinali|pasticche)",
    r"(?:troppe|troppi) (?:sonniferi|antidolorifici|psicofarmaci)",
    r"shume (?:tableta|hape\b|ilace|pilula|barna\b)",
    r"suicid",
    r"kill myself",
    r"end my life",
    r"want to die",
    r"self ?harm",
    # Intent, not accident: "I hurt myself skiing" and "I don't want to hurt myself again" pass.
    r"(?<!n t )(?<!not )(?<!never )(?:want(?:ed)? to|wanna|planning to|thinking (?:of|about)"
    r"|feel(?:ing)? like|urge to) (?:hurt|harm|injur|cut)\w* (?:myself|my self)",
    r"end it all",
    r"take my (?:own )?life",
    r"better off dead",
    r"wish i (?:was|were) dead",
    r"want to (?:live|be alive) (?:any ?more|no more)",
    r"no longer want to (?:live(?! in\b| here| there| with| near| alone| abroad)|be alive)",
    r"togliermi la vita",
    r"farla finita",
    r"voglio morire",
    r"(?<!non )(?:mi )?(?:voglio|vorrei|ho voglia di) (?:farmi (?:del )?male|uccidermi|ammazzarmi|suicidarmi)",
    r"(?<!non )mi (?:voglio|vorrei) (?:uccidere|ammazzare|suicidare|far(?:e)? (?:del )?male)",
    r"(?:penso|pensato|pensando|intenzione) (?:di|a) (?:uccidermi|ammazzarmi|suicidarmi|farmi del male)",
    r"(?<!non )vorrei (?:essere mort[oa]|morire)",
    r"non voglio piu vivere(?! in\b| qui| qua| vicino| lontano)",
    r"non ho piu voglia di vivere",
    r"autolesion",
    r"(?:tagliarmi|mi taglio) le vene",
    r"vetevr",
    r"fund jetes",
    r"nuk dua te jetoj me",
    r"vras veten",
    r"dua te vdes",
    r"emergency",
    r"emergenza",
    r"urgjenc",
    r"emergjenc",
    r"pronto soccorso",
    r"ambulan[cz]",
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
# Dates (2026-10-10, 10.10.2026, 10-10-2026) and time ranges (09.00-17.00) are digits joined by
# separators like a phone number; they are blanked out before the phone search. The parts are
# range-checked so that "12-34-5678" is still read as a phone number.
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH = r"(?:0?[1-9]|1[0-2])"
_YEAR = r"(?:19|20)\d{2}"
_CLOCK = r"(?:[01]?\d|2[0-3])[.:][0-5]\d"
_NOT_PHONE: Final = re.compile(
    rf"(?<![\d.-])(?:{_YEAR}[.-]{_MONTH}[.-]{_DAY}|{_DAY}[.-]{_MONTH}[.-]{_YEAR}|{_CLOCK}-{_CLOCK})(?!\d)"
)
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
    if _PHONE.search(_NOT_PHONE.sub(" ", text)):
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


def emergency_rule(text: str) -> str | None:
    """The rule that finds an emergency in `text`, or None. Only the emergency patterns run.

    They take time linear in the length of the text, which the personal-data patterns do not (a
    long word makes the email pattern quadratic), so a message too long for `evaluate` is still
    checked for an emergency with this alone.
    """
    return "emergency" if EMERGENCY.search(normalize(text)) else None


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
