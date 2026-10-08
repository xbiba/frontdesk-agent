import pytest
from hypothesis import given
from hypothesis import strategies as st

from frontdesk.safety.rules import (
    PersonalField,
    Verdict,
    contains_clinical_terms,
    contains_diagnostic_language,
    evaluate,
)

EMERGENCIES: dict[str, list[str]] = {
    "sq": [
        "kam dhimbje ne gjoks",
        "kam dhimbje të fortë në gjoks",
        "më dhemb gjoksi shumë",
        "dhimbje gjoksi",
        "nuk marr frymë",
        "nuk mund të marr frymë",
        "s'po marr dot frymë",
        "vështirësi në frymëmarrje",
        "babai ra pa ndjenja",
        "humbi ndjenjat",
        "po më rrjedh shumë gjak",
        "më ka zënë fryma",
    ],
    "it": [
        "dolore al petto",
        "ho un forte dolore al petto",
        "dolore forte al petto",
        "mi fa male il petto",
        "ho male al petto",
        "non riesco a respirare",
        "non riesco più a respirare",
        "faccio fatica a respirare",
        "è svenuto",
        "perde molto sangue",
        "credo sia un infarto",
        "mio figlio ha le convulsioni",
        "voglio farla finita",
        "ha preso troppe pastiglie",
    ],
    "en": [
        "chest pain",
        "severe chest pain",
        "pain in my chest",
        "my chest hurts",
        "I can't breathe",
        "I can not breathe",
        "struggling to breathe",
        "shortness of breath",
        "he passed out",
        "bleeding a lot",
        "I think it's a heart attack",
        "she had a seizure",
        "my chest feels tight",
    ],
}
NOT_EMERGENCIES = [
    "Quanto costa la fisioterapia?",
    "Sa kushton fizioterapia?",
    "Is there parking?",
    "Devo fare le analisi del sangue, serve il digiuno?",
    "Sa kushton analiza e gjakut?",
    "Can I book a chest X-ray?",
    "Do you check blood pressure?",
    "Non riesco a trovare l'indirizzo",
    "Nuk e gjej adresën",
    "A mund të marr një takim të shtunën?",
    "Ho mal di schiena, da chi devo andare?",
]


@pytest.mark.parametrize("text", [text for texts in EMERGENCIES.values() for text in texts])
def test_emergencies_are_recognised_with_words_in_between(text: str) -> None:
    assert evaluate(text).verdict is Verdict.EMERGENCY


@pytest.mark.parametrize("text", NOT_EMERGENCIES)
def test_practical_messages_are_not_emergencies(text: str) -> None:
    assert evaluate(text).verdict is not Verdict.EMERGENCY


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("I can't breathe", Verdict.EMERGENCY),
        ("Kam dhimbje ne gjoks qe nga mengjesi", Verdict.EMERGENCY),
        ("Kam dhimbje në gjoks që nga mëngjesi", Verdict.EMERGENCY),
        ("non riesco a respirare bene", Verdict.EMERGENCY),
        ("Çfarë ilaçi duhet të marr për shpinën?", Verdict.CLINICAL_ADVICE),
        ("cfare ilaci duhet te marr per shpinen", Verdict.CLINICAL_ADVICE),
        ("posso prendere un antinfiammatorio?", Verdict.CLINICAL_ADVICE),
        ("can you interpret my blood test results?", Verdict.CLINICAL_ADVICE),
        ("Il mio numero è 069 123 4567", Verdict.PERSONAL_DATA),
        ("call me +355691234567", Verdict.PERSONAL_DATA),
        ("+39 333 123 4567", Verdict.PERSONAL_DATA),
        ("scrivetemi a mario.rossi@example.com", Verdict.PERSONAL_DATA),
        ("Mi chiamo Mario Rossi", Verdict.PERSONAL_DATA),
        ("Quanto costa la fisioterapia?", Verdict.PASS),
        ("Sa kushton fizioterapia?", Verdict.PASS),
        ("Posso prendere un appuntamento?", Verdict.PASS),
        ("A mund të marr një takim të shtunën?", Verdict.PASS),
        ("Ho mal di schiena, da quale medico devo andare?", Verdict.PASS),
        ("Çfarë kam nevojë të sjell në vizitën e parë?", Verdict.PASS),
        ("Di cosa ho bisogno per la prima visita?", Verdict.PASS),
        ("Vorrei venire il 2026-10-10", Verdict.PASS),
        ("Ordine ORD-10427, CAP 20121", Verdict.PASS),
    ],
)
def test_rule_verdicts(text: str, verdict: Verdict) -> None:
    assert evaluate(text).verdict is verdict


def test_emergency_wins_over_everything() -> None:
    result = evaluate("chest pain, my number is +355691234567")
    assert result.verdict is Verdict.EMERGENCY


def test_expected_fields_do_not_trip_the_rule() -> None:
    assert (
        evaluate("+355691234567", expected=frozenset({PersonalField.PHONE})).verdict is Verdict.PASS
    )
    assert evaluate("+355691234567", expected=frozenset({PersonalField.EMAIL})).verdict is (
        Verdict.PERSONAL_DATA
    )


def test_clinical_terms_and_diagnostic_language() -> None:
    assert contains_clinical_terms("visita per dolore lombare")
    assert contains_clinical_terms("kam dhimbje")
    assert contains_clinical_terms("tell me what illness I have")
    assert contains_clinical_terms("ho male alla schiena")
    assert not contains_clinical_terms("Is there parking?")
    assert contains_diagnostic_language("It could be a tendinitis, you might have strained it")
    assert contains_diagnostic_language("Potrebbe trattarsi di un'infiammazione")
    assert not contains_diagnostic_language("La fisioterapia costa 30 EUR a seduta.")


@given(st.text(max_size=500))
def test_rules_never_raise(text: str) -> None:
    assert evaluate(text).verdict in set(Verdict)
