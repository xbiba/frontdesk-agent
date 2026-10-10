import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from frontdesk.safety.rules import (
    PersonalField,
    Verdict,
    contains_clinical_terms,
    contains_diagnostic_language,
    emergency_rule,
    evaluate,
    personal_fields,
)
from tests.timing import LINEAR_TIME_LIMIT_SECONDS

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
        "dua t'i jap fund jetës",
        "nuk dua të jetoj më",
        "mendoj për vetëvrasje",
        "mora shumë pilula",
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
        "voglio farmi del male",
        "mi voglio uccidere",
        "non voglio più vivere",
        "ho preso troppe medicine",
        "mi si chiude la gola",
        "autolesionismo",
        "non respiro bene",
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
        "I want to hurt myself",
        "I'm thinking about hurting myself",
        "I want to end it all",
        "I took too many tablets",
        "my throat is closing",
        "self-harm",
        "he isn't breathing",
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
    "mi sono fatto male al ginocchio",
    "I hurt myself skiing, can I book physio?",
    "kam vrarë këmbën",
    "uccidere i batteri",
    "I don't want to hurt myself again, how do I restart running?",
    "non voglio farmi del male mentre mi alleno",
    "ho paura di farmi male",
    "This pain is killing me",
    "I have a sore throat, what time do you close?",
    "How much does a blood draw cost?",
    "How much is a blood sugar test?",
    "Sa shumë kushton një analizë gjaku?",
    "Ka shumë hapësirë për parkim?",
    "Avete una svendita?",
    "Why are you not responding to my messages?",
    "What's the price of respiratory physiotherapy?",
    "Il prelievo del sangue è molto doloroso?",
    "Le analisi del sangue richiedono molto tempo?",
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
        ("Il mio numero è +355 00 111 2222", Verdict.PERSONAL_DATA),
        ("call me +355001112222", Verdict.PERSONAL_DATA),
        ("+39 00 1111 2222", Verdict.PERSONAL_DATA),
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
        ("Vorrei venire il 10.10.2026", Verdict.PASS),
        ("Vorrei venire il 10-10-2026", Verdict.PASS),
        ("Siete aperti dalle 09.00-17.00?", Verdict.PASS),
        ("Il mio numero è 0011112222", Verdict.PERSONAL_DATA),
        ("Il mio ordine è 1234567890", Verdict.PERSONAL_DATA),
        ("chiamami al 0011112222 il 10.10.2026", Verdict.PERSONAL_DATA),
        ("Il mio numero è 12-34-5678", Verdict.PERSONAL_DATA),
    ],
)
def test_rule_verdicts(text: str, verdict: Verdict) -> None:
    assert evaluate(text).verdict is verdict


def test_emergency_wins_over_everything() -> None:
    result = evaluate("chest pain, my number is +355001112222")
    assert result.verdict is Verdict.EMERGENCY


def test_expected_fields_do_not_trip_the_rule() -> None:
    assert (
        evaluate("+355001112222", expected=frozenset({PersonalField.PHONE})).verdict is Verdict.PASS
    )
    assert evaluate("+355001112222", expected=frozenset({PersonalField.EMAIL})).verdict is (
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


# The emergency patterns alone: the net for a message too long for `evaluate`, whose personal-data
# patterns take time that grows with the square of the length of a long word.


@pytest.mark.parametrize("text", [text for texts in EMERGENCIES.values() for text in texts])
def test_the_emergency_net_alone_names_the_same_rule_as_evaluate(text: str) -> None:
    assert emergency_rule(text) == evaluate(text).rule == "emergency"


@pytest.mark.parametrize("text", NOT_EMERGENCIES)
def test_the_emergency_net_alone_leaves_practical_messages_alone(text: str) -> None:
    assert emergency_rule(text) is None


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("Che farmaco devo prendere per la schiena?", Verdict.CLINICAL_ADVICE),
        ("Il mio numero è +355 00 111 2222", Verdict.PERSONAL_DATA),
        ("scrivetemi a mario.rossi@example.com", Verdict.PERSONAL_DATA),
    ],
)
def test_the_emergency_net_alone_does_not_run_the_other_rules(text: str, verdict: Verdict) -> None:
    assert evaluate(text).verdict is verdict
    assert emergency_rule(text) is None


@given(st.text(max_size=500))
def test_the_emergency_net_alone_agrees_with_evaluate_on_emergencies(text: str) -> None:
    assert (emergency_rule(text) is not None) == (evaluate(text).verdict is Verdict.EMERGENCY)


def test_the_emergency_net_reads_all_of_a_long_text() -> None:
    calm = "ciao come stai " * 7_000
    assert len(calm) > 100_000
    assert emergency_rule(calm) is None
    assert emergency_rule("mi fa male il petto " + calm) == "emergency"
    assert emergency_rule(calm + "nuk marr frymë") == "emergency"
    assert emergency_rule(calm + "he isn't breathing " + calm) == "emergency"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a" * 100_000, id="one-long-word"),
        pytest.param("pain " * 20_000, id="repeated-stem"),
        pytest.param("1 " * 50_000, id="digits"),
        pytest.param("a.b@" * 25_000, id="at-signs"),
        # Each of these characters becomes 18 once normalized (NFKD).
        pytest.param(chr(0xFDFA) * 100_000, id="characters-that-expand"),
        pytest.param("ciao come stai " * 7_000, id="prose"),
    ],
)
def test_the_emergency_net_alone_takes_time_linear_in_the_length(text: str) -> None:
    started = time.perf_counter()
    assert emergency_rule(text) is None
    assert time.perf_counter() - started < LINEAR_TIME_LIMIT_SECONDS


@pytest.mark.parametrize(
    "message",
    [
        "My che\N{SOFT HYPHEN}st hurts",
        "mi fa male il pet\N{ZERO WIDTH SPACE}to",
        "kam dhimbje ne gjo\N{WORD JOINER}ks",
        "\N{ZERO WIDTH NO-BREAK SPACE}non riesco a respi\N{ZERO WIDTH NON-JOINER}rare",
    ],
)
def test_invisible_characters_do_not_hide_an_emergency(message: str) -> None:
    assert evaluate(message).verdict is Verdict.EMERGENCY


@pytest.mark.parametrize(
    ("message", "field"),
    [
        ("scrivimi a mario\N{FULLWIDTH COMMERCIAL AT}example.com", PersonalField.EMAIL),
        ("il mio numero è 333\N{ZERO WIDTH SPACE}123\N{ZERO WIDTH SPACE}4567", PersonalField.PHONE),
        ("my number is \N{FULLWIDTH PLUS SIGN}355 69 123 4567", PersonalField.PHONE),
    ],
)
def test_fullwidth_and_invisible_characters_do_not_hide_personal_data(
    message: str, field: PersonalField
) -> None:
    assert field in personal_fields(message)
