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
    promise_phrase,
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


# --- Promises of an action no one will take (final review P85) ----------------------------------


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("Ti contatteremo noi al più presto.", "ti contatteremo"),
        ("Passo la conversazione alla reception.", "passo la conversazione"),
        ("Riceverai una conferma via SMS.", "riceverai una conferma"),
        ("We will email you the instructions.", "we will email"),
        ("Someone will call you back shortly.", "someone will call"),
        ("I'm passing this conversation to a colleague.", "passing this conversation"),
        ("Do t'ju telefonojmë nesër.", "do t ju telefonojme"),
        ("Po ia kaloj recepsionit.", "po ia kaloj"),
        ("Do të merrni një konfirmim.", "do te merrni nje konfirmim"),
    ],
)
def test_a_promise_of_an_action_is_found_in_each_language(text: str, phrase: str) -> None:
    assert promise_phrase(text) == phrase


@pytest.mark.parametrize(
    "text",
    [
        "Per parlare con una persona, chiama la reception al +355 00 000 0000.",
        "To talk to a person, write to ciao@bottega-demo.example.",
        "Would you like prices or opening hours, or would you rather talk to reception?",
        "We will be closed on Sunday.",
        "La clinica invia un promemoria il giorno prima.",
        "Mund t'ju ndihmoj me diçka tjetër?",
    ],
)
def test_ordinary_replies_promise_nothing(text: str) -> None:
    assert promise_phrase(text) is None


# A wording for each pattern of the rule that the cases above leave out, and for each form of
# "will" in the first pattern, so that none of them can be dropped or broken unseen.
@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("I'll call you tomorrow.", "i ll call"),
        ("We are going to call you tomorrow.", "we are going to call"),
        ("We're going to send you a link.", "we re going to send"),
        ("A colleague will get back to you.", "a colleague will get back"),
        ("You will receive a confirmation.", "you will receive a confirmation"),
        ("The doctor will see your message.", "will see your message"),
        ("Ti richiamiamo domani.", "ti richiamiamo"),
        ("Qualcuno ti richiamerà domani.", "qualcuno ti richiamera"),
        ("La reception vedrà il tuo messaggio.", "vedra il tuo messaggio"),
        ("Ti faremo sapere.", "ti faremo sapere"),
        ("Dikush do t'ju telefonojë.", "dikush do t ju telefonoje"),
        ("Mjeku do ta shohë mesazhin.", "do ta shohe"),
    ],
)
def test_a_promise_is_found_in_every_pattern_of_the_rule(text: str, phrase: str) -> None:
    assert promise_phrase(text) == phrase


# The promise rule reads the text in the forms the other word patterns read (`_normal_forms`): with
# the invisible characters removed, which keeps a word whole, and read as spaces, which keeps two
# words apart.


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("We will\N{ZERO WIDTH SPACE}email you the instructions.", "we will email"),
        ("Ti\N{ZERO WIDTH SPACE}contatteremo noi al più presto.", "ti contatteremo"),
        ("Do t'ju\N{ZERO WIDTH SPACE}telefonojmë nesër.", "do t ju telefonojme"),
    ],
)
def test_an_invisible_character_between_two_words_does_not_hide_a_promise(
    text: str, phrase: str
) -> None:
    assert promise_phrase(text) == phrase


def test_an_invisible_character_inside_a_word_does_not_hide_a_promise() -> None:
    assert promise_phrase("We will em\N{SOFT HYPHEN}ail you the instructions.") == "we will email"


def test_the_promise_of_the_form_without_the_invisible_characters_is_reported_first() -> None:
    # Removed, only the second promise is whole; read as spaces, only the first is. Each form finds
    # one, and the one found in the form with the characters removed is the one reported.
    text = "We will\N{ZERO WIDTH SPACE}email you. Ti contat\N{SOFT HYPHEN}teremo."
    assert promise_phrase(text) == "ti contatteremo"


# --- More wordings of the same promises (review of task 4) ----------------------------------------


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        # English
        ("The clinic will call you tomorrow.", "the clinic will call you"),
        ("Someone will get in touch shortly.", "will get in touch"),
        ("We have received your request and will reply shortly.", "will reply shortly"),
        ("I'll check with the doctor and come back to you.", "come back to you"),
        ("I will speak to the doctor and let you know.", "and let you know"),
        ("Expect a call from us tomorrow.", "expect a call from"),
        ("I am going to pass your request to the reception.", "i am going to pass"),
        ("You will hear from us soon.", "you will hear from"),
        ("You will receive a reminder by SMS.", "you will receive a reminder by sms"),
        ("We will keep you updated.", "we will keep you updated"),
        ("I'll notify the reception right away.", "i ll notify the reception"),
        ("I will have someone call you.", "i will have someone call"),
        ("I'll make sure the doctor gets your message.", "i ll make sure the doctor"),
        ("I've passed your message to the team.", "i ve passed your message"),
        ("Your request has been forwarded to the reception.", "your request has been forwarded"),
        ("They will call you back.", "they will call you"),
        ("She will contact you tomorrow.", "she will contact you"),
        ("Dr. Rossi will call you.", "dr rossi will call you"),
        ("Our receptionist will call you.", "our receptionist will call you"),
        ("The shop will email you a tracking link.", "the shop will email you"),
        ("You'll hear from the clinic tomorrow.", "you ll hear from"),
        ("You will be notified by SMS.", "you will be notified"),
        # Italian
        ("Ti chiamo domani.", "ti chiamo"),
        ("Ti faccio sapere.", "ti faccio sapere"),
        ("Ti scrivo appena possibile.", "ti scrivo appena"),
        ("Ti mando una mail con i dettagli domani.", "ti mando una mail"),
        ("Ti daremo conferma al più presto.", "ti daremo conferma"),
        ("Ti passo la reception.", "ti passo la reception"),
        ("Ti avvertiremo in caso di variazioni.", "ti avvertiremo"),
        ("Ti confermeremo l'orario.", "ti confermeremo"),
        ("Ti terremo aggiornato.", "ti terremo aggiornat"),
        ("Contatterò la reception per te.", "contattero"),
        ("Chiederò alla reception di richiamarti.", "chiedero alla reception di"),
        ("Provvederemo a contattarti.", "provvederemo a contattarti"),
        ("Risponderemo entro 24 ore.", "risponderemo entro"),
        ("Ci metteremo in contatto con te.", "metteremo in contatto"),
        ("Sarai ricontattato dalla segreteria.", "sarai ricontattato"),
        ("Ti arriverà una conferma via SMS.", "ti arrivera una conferma"),
        ("Riceverai un promemoria via SMS.", "riceverai un promemoria"),
        ("Ho inoltrato la tua richiesta alla reception.", "ho inoltrato la tua richiesta"),
        ("La tua richiesta è stata inoltrata.", "la tua richiesta e stata inoltrata"),
        ("Ti richiamo io.", "ti richiamo"),
        ("Ti avviso io appena ho notizie.", "ti avviso io appena"),
        ("Ti farò sapere domani.", "ti faro sapere"),
        ("Verrai richiamato entro oggi.", "verrai richiamato"),
        ("Le faremo sapere.", "le faremo sapere"),
        # Albanian
        ("Do t'ju telefonojë recepsioni.", "do t ju telefonoje"),
        ("Do t'ju informojmë për orarin.", "do t ju informojme"),
        ("Do t'ju kthejmë përgjigje brenda ditës.", "do t ju kthejme pergjigje"),
        ("Ekipi ynë do të flasë me ju.", "do te flase me ju"),
        ("Do të kontaktoheni nga stafi ynë.", "do te kontaktoheni"),
        ("Do të merrni një SMS konfirmimi.", "do te merrni nje sms"),
        ("Do t'ia kaloj kërkesën tuaj recepsionit.", "do t ia kaloj"),
        ("Do ta informoj mjekun.", "do ta informoj"),
        ("Po ia përcjell kërkesën tuaj mjekut.", "po ia percjell"),
        ("Ju telefonoj nesër.", "ju telefonoj"),
        ("Ia kalova kërkesën tuaj recepsionit.", "ia kalova kerkesen"),
        ("E kam njoftuar recepsionin.", "e kam njoftuar recepsionin"),
        ("Mesazhi juaj u përcoll te recepsioni.", "mesazhi juaj u percoll"),
        ("Do t'ju telefonoj nesër.", "do t ju telefonoj"),
        ("Do t'ju kontaktoj sa më shpejt.", "do t ju kontaktoj"),
        ("Do t'ju telefonojë mjeku nesër.", "do t ju telefonoje"),
        ("Klinika do t'ju kontaktojë për konfirmim.", "do t ju kontaktoje"),
        ("Do të njoftoheni me SMS.", "do te njoftoheni"),
        ("Do t'ju dërgoj një email.", "do t ju dergoj"),
    ],
)
def test_a_promise_in_other_words_is_found_in_each_language(text: str, phrase: str) -> None:
    assert promise_phrase(text) == phrase


@pytest.mark.parametrize(
    "text",
    [
        # English
        "To talk to a person, call reception on +355 00 000 0000.",
        "You can call us at +355 00 000 0000.",
        "Call 127 now.",
        "We are open until 18:00.",
        "We will be closed on Sunday.",
        "Please call reception to confirm your appointment.",
        "Let us know if you need anything else.",
        "Would you like me to pass your message to reception?",
        "I will answer questions about services and opening hours.",
        "Please notify us if you are late.",
        "You will need an ID card.",
        "You will receive a bill at the end of the visit.",
        "The doctor will inform you about the treatment during the visit.",
        "The nurse will call you in when the doctor is ready.",
        "You will be informed of the price before treatment.",
        "You will hear a beep when the door opens.",
        "We have updated the opening hours for October.",
        "The doctor has informed the patients about the new policy.",
        "We will inform you that the clinic is closed on Sunday.",
        "I have checked the passages.",
        "The clinic is on the ground floor and has a ramp.",
        # Italian
        "Per parlare con una persona, chiama la reception al +355 00 000 0000.",
        "La clinica invia un promemoria il giorno prima.",
        "Quando l'ordine parte ricevi un'email con il link per seguire la spedizione.",
        "Ti scrivo qui sotto l'indirizzo: Via Demo 1.",
        "Ti mando qui i dettagli.",
        "Ti scrivo per informarti che l'orario è cambiato.",
        "Ti rispondo subito: siamo aperti fino alle 18:00.",
        "Ti informo che la clinica è chiusa domenica.",
        "Ti confermo che apriamo alle 9:00.",
        "Ti avviso che i prezzi possono cambiare.",
        "Come ti chiami?",
        "Il medico ti informerà sul trattamento durante la visita.",
        "Il medico ti confermerà il trattamento durante la visita.",
        "Verrai informato dei costi prima del trattamento.",
        "Sarai chiamato dal medico quando è il tuo turno.",
        "Riceverai la fattura al termine della visita.",
        "Non posso inoltrare messaggi.",
        "Non posso chiamarti: chiama tu la reception.",
        "Contattaci per qualsiasi domanda.",
        "Vuoi che ti passi la reception?",
        # Albanian
        "Për të folur me një person, telefononi recepsionin në +355 00 000 0000.",
        "Mund t'ju ndihmoj me diçka tjetër?",
        "Ju shkruan asistenti automatik i Klinikës Demo.",
        "Do t'ju presim të hënën në orën 10:00.",
        "Do të merrni një faturë në fund të vizitës.",
        "Do t'ju shërbejmë me kënaqësi.",
        "Do t'ju duhet një dokument identiteti.",
        "Mjeku do t'ju informojë për trajtimin gjatë vizitës.",
        "Mjeku do t'ju konfirmojë trajtimin gjatë vizitës.",
        "Do të informoheni për çmimin para trajtimit.",
        "Ju njoftojmë se orari ndryshon të dielën.",
        "Ju njoftoj se klinika është e mbyllur të dielën.",
        "Nuk mund t'ju telefonoj: telefononi ju recepsionin.",
        "Nuk mund të përcjell mesazhe: telefononi recepsionin në +355 00 000 0000.",
        "Mjeku do t'ju vizitojë në orën 10:00.",
        "Do t'ju pyesim për historikun mjekësor.",
    ],
)
def test_a_reply_about_contact_that_promises_nothing_stays_clear(text: str) -> None:
    assert promise_phrase(text) is None
