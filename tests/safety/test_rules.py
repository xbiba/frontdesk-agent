import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from frontdesk.safety.normalize import fold, normalize, space_invisible
from frontdesk.safety.rules import (
    PersonalField,
    Verdict,
    contains_clinical_terms,
    contains_diagnostic_language,
    emergency_rule,
    evaluate,
    personal_fields,
    rule_text_length,
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


# A phone number whose only boundary with a word beside it is an invisible character. With the
# character removed the digits are glued to the letters, and the pattern refuses digits that follow
# a letter; read as it was written, each of the first four was found (review of the Task 2 change).
GLUED_PHONE_NUMBERS = [
    "Tel\N{LEFT-TO-RIGHT MARK}355691234567",
    "call\N{ZERO WIDTH SPACE}12345678",
    "12345678\N{ZERO WIDTH SPACE}now",
    "12345678\N{SOFT HYPHEN}now",
    "call\N{ZERO WIDTH SPACE}333\N{SOFT HYPHEN}123\N{SOFT HYPHEN}4567",
]

# A phone number beside a symbol that folding turns into a letter, a hyphen or an underscore. As
# written the symbol is none of those, so each was found before folding (re-review of Task 2).
PHONE_NUMBERS_BESIDE_FOLDING_SYMBOLS = [
    "12345678\N{CIRCLED LATIN SMALL LETTER A}",
    "\N{CIRCLED LATIN SMALL LETTER A}12345678",
    "\N{SMALL HYPHEN-MINUS}12345678",
    "12345678\N{FULLWIDTH LOW LINE}",
]

# A phone number and an address written with fullwidth or invisible characters, and the field the
# rules find in each. The tests of trace redaction read the same messages.
DISGUISED_PERSONAL_DATA: list[tuple[str, PersonalField]] = [
    ("scrivimi a mario\N{FULLWIDTH COMMERCIAL AT}example.com", PersonalField.EMAIL),
    ("il mio numero è 333\N{ZERO WIDTH SPACE}123\N{ZERO WIDTH SPACE}4567", PersonalField.PHONE),
    ("my number is \N{FULLWIDTH PLUS SIGN}355 69 123 4567", PersonalField.PHONE),
    *[(message, PersonalField.PHONE) for message in GLUED_PHONE_NUMBERS],
    *[(message, PersonalField.PHONE) for message in PHONE_NUMBERS_BESIDE_FOLDING_SYMBOLS],
    # Two invisible characters in a row are two spaces when read as spaces, which splits the
    # number in two: only the fold with the invisible characters removed joins it.
    ("my number is 1234\N{ZERO WIDTH SPACE}\N{SOFT HYPHEN}5678", PersonalField.PHONE),
    # A soft hyphen in the domain splits it when read as a space: only the removed fold finds it.
    ("write to mario@exa\N{SOFT HYPHEN}mple.com", PersonalField.EMAIL),
]


@pytest.mark.parametrize(("message", "field"), DISGUISED_PERSONAL_DATA)
def test_fullwidth_and_invisible_characters_do_not_hide_personal_data(
    message: str, field: PersonalField
) -> None:
    assert field in personal_fields(message)


@pytest.mark.parametrize("message", GLUED_PHONE_NUMBERS)
def test_a_phone_number_glued_to_a_word_by_an_invisible_character_is_still_found(
    message: str,
) -> None:
    assert evaluate(message).verdict is Verdict.PERSONAL_DATA


# An invisible character that is the only separator between two words has to count as a space, as it
# did before such characters were removed to keep words whole (P34): "chest", a zero-width space
# and "pain" are two words, while "che", a zero-width space and "st" are one.


@pytest.mark.parametrize(
    "message",
    [
        "dolore al\N{ZERO WIDTH SPACE}petto",
        "nuk marr\N{ZERO WIDTH SPACE}frymë",
        "chest\N{ZERO WIDTH SPACE}pain",
        "he isn't\N{WORD JOINER}breathing",
    ],
)
def test_an_invisible_character_between_two_words_does_not_hide_an_emergency(
    message: str,
) -> None:
    assert evaluate(message).verdict is Verdict.EMERGENCY
    assert emergency_rule(message) == "emergency"


def test_an_invisible_character_between_two_words_does_not_hide_clinical_advice() -> None:
    message = "Che\N{ZERO WIDTH SPACE}farmaco devo prendere per la schiena?"
    assert evaluate(message).verdict is Verdict.CLINICAL_ADVICE


def test_an_emergency_found_in_either_form_wins_over_advice_found_in_the_other() -> None:
    # With the invisible character removed only the question is found; read as a space, the
    # emergency is too, and an emergency still outranks the question.
    message = "what do i have, chest\N{ZERO WIDTH SPACE}pain"
    assert evaluate(message).verdict is Verdict.EMERGENCY


@pytest.mark.parametrize(
    ("message", "field"),
    [
        ("date of\N{ZERO WIDTH SPACE}birth", PersonalField.BIRTH_DATE),
        ("mi\N{ZERO WIDTH SPACE}chiamo Mario", PersonalField.NAME),
        ("il mio codice\N{ZERO WIDTH SPACE}fiscale", PersonalField.IDENTITY_DOCUMENT),
    ],
)
def test_an_invisible_character_between_two_words_does_not_hide_a_personal_field(
    message: str, field: PersonalField
) -> None:
    assert field in personal_fields(message)


def test_an_invisible_character_between_two_words_does_not_hide_clinical_terms() -> None:
    assert contains_clinical_terms("mal\N{ZERO WIDTH SPACE}di testa")
    assert contains_clinical_terms("what\N{ZERO WIDTH SPACE}do i have")


def test_an_invisible_character_between_two_words_does_not_hide_diagnostic_language() -> None:
    assert contains_diagnostic_language("you may\N{ZERO WIDTH SPACE}have strained it")


def test_the_text_length_counts_invisible_characters_read_as_spaces() -> None:
    # A unit of U+33AF and a zero-width space is 6 characters with the invisible one removed and 7
    # with it read as a space, so 333 of them are 666 characters as written, 1,998 folded or
    # normalized with the invisible characters removed, and read as spaces 2,330 normalized and
    # 2,331 folded (the fold keeps the space after the last unit): under the 2,000 of a turn in
    # every form but the last two.
    message = ("\N{SQUARE RAD OVER S SQUARED}" + "\N{ZERO WIDTH SPACE}") * 333
    assert (len(message), len(fold(message)), len(normalize(message))) == (666, 1_998, 1_998)
    assert len(normalize(space_invisible(message))) == 2_330
    assert len(fold(space_invisible(message))) == 2_331
    assert rule_text_length(message) == 2_331


def test_the_text_length_counts_the_text_as_written() -> None:
    # A letter and its accent are one character folded, and the accent is dropped normalized: of
    # the 3,000 characters as written only 1,500 are left in any form the rules read. (A text of
    # one letter and many invisible characters would not do: read as spaces, they stay as long.)
    message = "e\N{COMBINING ACUTE ACCENT}" * 1_500
    assert (len(fold(message)), len(normalize(message))) == (1_500, 1_500)
    assert rule_text_length(message) == len(message) == 3_000


def test_the_text_length_counts_the_fold_with_invisible_characters_read_as_spaces() -> None:
    # The phone pattern reads that fold too. U+3316 is six letters folded, none of them one for
    # `normalize`, so 333 units of it and a zero-width space are 666 characters as written, 1,998
    # folded with the invisible characters removed, nothing normalized, and 2,331 folded with them
    # read as spaces (7 characters a unit): the longest form is one that `normalize` never makes.
    message = ("\N{SQUARE KIROMEETORU}" + "\N{ZERO WIDTH SPACE}") * 333
    assert (len(message), len(fold(message)), len(normalize(message))) == (666, 1_998, 0)
    assert len(normalize(space_invisible(message))) == 0
    assert len(fold(space_invisible(message))) == 2_331
    assert rule_text_length(message) == 2_331
