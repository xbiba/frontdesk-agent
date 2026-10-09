"""Number words in English, Italian and Albanian, read as digits (ADR-0009)."""

from collections.abc import Sequence

import pytest

from frontdesk.agent.answer_check import AnswerCheck, check_answer, numbers_in
from frontdesk.agent.number_words import spoken_numbers
from frontdesk.agent.responder import ResponderOutput
from frontdesk.knowledge.evidence import Evidence, EvidenceKind

APOSTROPHE = chr(0x2019)
SOFT_HYPHEN = chr(0x00AD)
ZERO_WIDTH_SPACE = chr(0x200B)

# Spelled out by ICU (CLDR) and cross-checked with num2words for English and Italian, one entry per
# number: the words under test do not come from the module's own lists.
SPELLED_0_TO_20 = {
    "en": (
        "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
        "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
    ),
    "it": (
        "zero|uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|tredici|"
        "quattordici|quindici|sedici|diciassette|diciotto|diciannove|venti"
    ),
    "sq": (
        "zero|një|dy|tre|katër|pesë|gjashtë|shtatë|tetë|nëntë|dhjetë|njëmbëdhjetë|dymbëdhjetë|"
        "trembëdhjetë|katërmbëdhjetë|pesëmbëdhjetë|gjashtëmbëdhjetë|shtatëmbëdhjetë|"
        "tetëmbëdhjetë|nëntëmbëdhjetë|njëzet"
    ),
}
SPELLED_21_TO_99 = {
    "en": (
        "twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six|twenty-seven|"
        "twenty-eight|twenty-nine|thirty|thirty-one|thirty-two|thirty-three|thirty-four|"
        "thirty-five|thirty-six|thirty-seven|thirty-eight|thirty-nine|forty|forty-one|"
        "forty-two|forty-three|forty-four|forty-five|forty-six|forty-seven|forty-eight|"
        "forty-nine|fifty|fifty-one|fifty-two|fifty-three|fifty-four|fifty-five|fifty-six|"
        "fifty-seven|fifty-eight|fifty-nine|sixty|sixty-one|sixty-two|sixty-three|sixty-four|"
        "sixty-five|sixty-six|sixty-seven|sixty-eight|sixty-nine|seventy|seventy-one|"
        "seventy-two|seventy-three|seventy-four|seventy-five|seventy-six|seventy-seven|"
        "seventy-eight|seventy-nine|eighty|eighty-one|eighty-two|eighty-three|eighty-four|"
        "eighty-five|eighty-six|eighty-seven|eighty-eight|eighty-nine|ninety|ninety-one|"
        "ninety-two|ninety-three|ninety-four|ninety-five|ninety-six|ninety-seven|ninety-eight|"
        "ninety-nine"
    ),
    "it": (
        "ventuno|ventidue|ventitré|ventiquattro|venticinque|ventisei|ventisette|ventotto|"
        "ventinove|trenta|trentuno|trentadue|trentatré|trentaquattro|trentacinque|trentasei|"
        "trentasette|trentotto|trentanove|quaranta|quarantuno|quarantadue|quarantatré|"
        "quarantaquattro|quarantacinque|quarantasei|quarantasette|quarantotto|quarantanove|"
        "cinquanta|cinquantuno|cinquantadue|cinquantatré|cinquantaquattro|cinquantacinque|"
        "cinquantasei|cinquantasette|cinquantotto|cinquantanove|sessanta|sessantuno|"
        "sessantadue|sessantatré|sessantaquattro|sessantacinque|sessantasei|sessantasette|"
        "sessantotto|sessantanove|settanta|settantuno|settantadue|settantatré|settantaquattro|"
        "settantacinque|settantasei|settantasette|settantotto|settantanove|ottanta|ottantuno|"
        "ottantadue|ottantatré|ottantaquattro|ottantacinque|ottantasei|ottantasette|"
        "ottantotto|ottantanove|novanta|novantuno|novantadue|novantatré|novantaquattro|"
        "novantacinque|novantasei|novantasette|novantotto|novantanove"
    ),
    "sq": (
        "njëzet e një|njëzet e dy|njëzet e tre|njëzet e katër|njëzet e pesë|njëzet e gjashtë|"
        "njëzet e shtatë|njëzet e tetë|njëzet e nëntë|tridhjetë|tridhjetë e një|"
        "tridhjetë e dy|tridhjetë e tre|tridhjetë e katër|tridhjetë e pesë|"
        "tridhjetë e gjashtë|tridhjetë e shtatë|tridhjetë e tetë|tridhjetë e nëntë|dyzet|"
        "dyzet e një|dyzet e dy|dyzet e tre|dyzet e katër|dyzet e pesë|dyzet e gjashtë|"
        "dyzet e shtatë|dyzet e tetë|dyzet e nëntë|pesëdhjetë|pesëdhjetë e një|"
        "pesëdhjetë e dy|pesëdhjetë e tre|pesëdhjetë e katër|pesëdhjetë e pesë|"
        "pesëdhjetë e gjashtë|pesëdhjetë e shtatë|pesëdhjetë e tetë|pesëdhjetë e nëntë|"
        "gjashtëdhjetë|gjashtëdhjetë e një|gjashtëdhjetë e dy|gjashtëdhjetë e tre|"
        "gjashtëdhjetë e katër|gjashtëdhjetë e pesë|gjashtëdhjetë e gjashtë|"
        "gjashtëdhjetë e shtatë|gjashtëdhjetë e tetë|gjashtëdhjetë e nëntë|shtatëdhjetë|"
        "shtatëdhjetë e një|shtatëdhjetë e dy|shtatëdhjetë e tre|shtatëdhjetë e katër|"
        "shtatëdhjetë e pesë|shtatëdhjetë e gjashtë|shtatëdhjetë e shtatë|shtatëdhjetë e tetë|"
        "shtatëdhjetë e nëntë|tetëdhjetë|tetëdhjetë e një|tetëdhjetë e dy|tetëdhjetë e tre|"
        "tetëdhjetë e katër|tetëdhjetë e pesë|tetëdhjetë e gjashtë|tetëdhjetë e shtatë|"
        "tetëdhjetë e tetë|tetëdhjetë e nëntë|nëntëdhjetë|nëntëdhjetë e një|nëntëdhjetë e dy|"
        "nëntëdhjetë e tre|nëntëdhjetë e katër|nëntëdhjetë e pesë|nëntëdhjetë e gjashtë|"
        "nëntëdhjetë e shtatë|nëntëdhjetë e tetë|nëntëdhjetë e nëntë"
    ),
}
# Alone, these are an article or a verb as much as a number, so they are not read.
NOT_READ_ALONE = {"en": {1}, "it": {1, 6}, "sq": {1}}

RETURNS_EN = (
    "You can return an item within fourteen days of delivery, unused and with its label. "
    "Send us the order number and we will email the instructions."
)
ARRIVAL_IT = "Arriva dieci minuti prima per la registrazione alla reception."
VISIT_IT = "Porta un documento d'identità e, se li hai, i referti degli ultimi due anni."
PARKING_EN = "Klinika Demo has six free parking spaces behind the building. Street parking is paid on weekdays."
SESSION_SQ = (
    "Sillni rroba të rehatshme dhe raportet e mëparshme, nëse keni. "
    "Seanca e parë zgjat rreth dyzet e pesë minuta."
)
SESSION_IT = (
    "Porta abiti comodi e gli eventuali referti precedenti. "
    "La prima seduta dura circa quarantacinque minuti."
)
REFUND_IT = "Il rimborso arriva entro 14 giorni dalla consegna del pacco."


class _Similar:
    """Finds every text identical to every other, so that only the number check decides."""

    dimension = 2
    identity = "similar"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _answer(content: str, text: str, quote: str) -> AnswerCheck:
    passage = Evidence.create(
        kind=EvidenceKind.DOCUMENT, source_ref="doc#1", version="v1", title="Doc", content=content
    )
    output = ResponderOutput.model_validate(
        {
            "action": "answer",
            "claims": [{"text": text, "evidence_id": passage.evidence_id, "quote": quote}],
        }
    )
    return check_answer(output, [passage], embedder=_Similar(), min_similarity=0.5)


@pytest.mark.parametrize("language", ["en", "it", "sq"])
def test_numbers_up_to_twenty_are_read_as_icu_spells_them(language: str) -> None:
    words = SPELLED_0_TO_20[language].split("|")
    assert len(words) == 21
    read = {
        value: spoken_numbers(spelled)
        for value, spelled in enumerate(words)
        if value not in NOT_READ_ALONE[language]
    }
    assert {value: found for value, found in read.items() if found != {str(value)}} == {}
    for value in NOT_READ_ALONE[language]:
        assert spoken_numbers(words[value]) == set()


@pytest.mark.parametrize("language", ["en", "it", "sq"])
def test_numbers_from_21_to_99_are_read_as_icu_spells_them(language: str) -> None:
    words = SPELLED_21_TO_99[language].split("|")
    assert len(words) == 79
    wrong = {
        spelled: spoken_numbers(spelled)
        for value, spelled in zip(range(21, 100), words, strict=True)
        if spoken_numbers(spelled) != {str(value)}
    }
    assert wrong == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("hundred", "100", id="en-hundred"),
        pytest.param("thousand", "1000", id="en-thousand"),
        pytest.param("million", "1000000", id="en-million"),
        pytest.param("cento", "100", id="it-cento"),
        pytest.param("duecento", "200", id="it-two-hundred"),
        pytest.param("novecento", "900", id="it-nine-hundred"),
        pytest.param("mille", "1000", id="it-mille"),
        pytest.param("duemila", "2000", id="it-two-thousand"),
        pytest.param("novemila", "9000", id="it-nine-thousand"),
        pytest.param("milione", "1000000", id="it-million"),
        pytest.param("njëqind", "100", id="sq-hundred"),
        pytest.param("dyqind", "200", id="sq-two-hundred"),
        pytest.param("nëntëqind", "900", id="sq-nine-hundred"),
        pytest.param("mijë", "1000", id="sq-thousand"),
        pytest.param("milion", "1000000", id="sq-million"),
    ],
)
def test_round_numbers_are_read(text: str, expected: str) -> None:
    assert spoken_numbers(text) == {expected}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("GJASHTË", id="upper-case-with-diacritics"),
        pytest.param("Gjashte", id="capital-without-diacritics"),
        pytest.param("gjashtë", id="lower-case"),
    ],
)
def test_case_and_diacritics_do_not_matter(text: str) -> None:
    assert spoken_numbers(text) == {"6"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("forty-five", {"45"}, id="en-hyphen"),
        pytest.param("forty five", {"45"}, id="en-space"),
        pytest.param("twenty and five", {"25"}, id="en-and"),
        pytest.param("twenty-one", {"21"}, id="en-one-after-a-tens-word"),
        pytest.param("dyzet e pesë", {"45"}, id="sq-e"),
        pytest.param("njëzet e një", {"21"}, id="sq-nje-after-a-tens-word"),
        pytest.param("quarantacinque", {"45"}, id="it-one-word"),
        pytest.param("ventuno", {"21"}, id="it-elided-uno"),
        pytest.param("ventotto", {"28"}, id="it-elided-otto"),
        pytest.param("trentatré", {"33"}, id="it-accented-tre"),
        pytest.param("forty-five minutes and ten seconds", {"45", "10"}, id="two-numbers"),
    ],
)
def test_a_compound_is_one_number(text: str, expected: set[str]) -> None:
    assert spoken_numbers(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("twenty. One of us", {"20"}, id="a-full-stop-ends-the-number"),
        pytest.param("forty, five", {"40", "5"}, id="a-comma-ends-the-number"),
        pytest.param("forty (five)", {"40", "5"}, id="brackets-end-the-number"),
        pytest.param("ten twenty", {"10", "20"}, id="two-tens-are-two-numbers"),
        pytest.param("due-quattro giorni lavorativi", {"2", "4"}, id="a-range-gives-both-ends"),
        pytest.param("dalle otto alle dieci", {"8", "10"}, id="two-times-of-day"),
    ],
)
def test_numbers_that_are_not_one_number_stay_apart(text: str, expected: set[str]) -> None:
    assert spoken_numbers(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("one of us", id="en-one"),
        pytest.param("no one came", id="en-no-one"),
        pytest.param("Porta un documento", id="it-un"),
        pytest.param("uno studio medico", id="it-uno"),
        pytest.param("Ho una domanda", id="it-una"),
        pytest.param("se sei in ritardo", id="it-sei-is-you-are"),
        pytest.param("sei un paziente", id="it-sei-before-an-article"),
        pytest.param("një person", id="sq-nje-is-a"),
        pytest.param("një vizitë", id="sq-nje-before-a-visit"),
    ],
)
def test_articles_and_verbs_are_not_numbers(text: str) -> None:
    assert spoken_numbers(text) == set()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("one week", "1", id="en-week"),
        pytest.param("one day", "1", id="en-day"),
        pytest.param("one-hour sessions", "1", id="en-hyphen"),
        pytest.param("una settimana", "1", id="it-settimana"),
        pytest.param("un anno", "1", id="it-anno"),
        pytest.param("un'ora prima", "1", id="it-elided-ora"),
        pytest.param(f"un{APOSTROPHE}ora prima", "1", id="it-typographic-apostrophe"),
        pytest.param("sei giorni", "6", id="it-sei-six-days"),
        pytest.param("një javë", "1", id="sq-week"),
        pytest.param("një orë", "1", id="sq-hour"),
        pytest.param("një muaj", "1", id="sq-month"),
    ],
)
def test_such_words_count_before_a_unit_of_time(text: str, expected: str) -> None:
    assert spoken_numbers(text) == {expected}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("settembre", id="it-september-is-not-sette"),
        pytest.param("ottobre", id="it-october-is-not-otto"),
        pytest.param("novembre", id="it-november-is-not-nove"),
        pytest.param("treno", id="it-train-is-not-tre"),
        pytest.param("often", id="en-often-is-not-ten"),
        pytest.param("someone", id="en-someone-is-not-one"),
        pytest.param("sixth floor", id="en-sixth-is-not-six"),
        pytest.param("dyshek", id="sq-mattress-is-not-dy"),
    ],
)
def test_the_words_inside_longer_words_are_not_read(text: str) -> None:
    assert spoken_numbers(text) == set()


def test_per_cento_is_not_a_hundred_but_cento_alone_is() -> None:
    assert spoken_numbers("il 50 per cento") == set()
    assert spoken_numbers("il 50 percento") == set()
    assert spoken_numbers("cento persone") == {"100"}


def test_invisible_characters_cannot_hide_a_number() -> None:
    assert spoken_numbers(f"venti{SOFT_HYPHEN}due") == {"22"}
    assert spoken_numbers(f"qua{ZERO_WIDTH_SPACE}ttro") == {"4"}


def test_what_is_not_read_is_listed() -> None:
    # The limits named in the module docstring: they fail open, so a change here is a decision.
    assert spoken_numbers("two hundred") == {"2", "100"}
    assert spoken_numbers("centocinquanta") == set()
    assert spoken_numbers("second floor") == set()
    assert spoken_numbers("due to roadworks") == {"2"}  # English "due" reads as Italian "due"


def test_numbers_in_gives_digits_and_words_the_same_form() -> None:
    assert numbers_in("14 giorni e quattordici") == {"14"}
    assert numbers_in("alle 8.00 o alle otto") == {"08:00", "8"}
    assert numbers_in("forty-five minutes") == {"45"}


@pytest.mark.parametrize(
    ("content", "text", "quote"),
    [
        pytest.param(
            RETURNS_EN,
            "You can return an item within thirty days of delivery.",
            "return an item within fourteen days of delivery",
            id="en-thirty-for-fourteen",
        ),
        pytest.param(
            ARRIVAL_IT,
            "Arriva venti minuti prima.",
            "Arriva dieci minuti prima per la registrazione",
            id="it-venti-for-dieci",
        ),
        pytest.param(
            VISIT_IT,
            "Porta i referti degli ultimi tre anni.",
            "i referti degli ultimi due anni",
            id="it-tre-for-due",
        ),
        pytest.param(
            PARKING_EN,
            "Klinika Demo has sixty free parking spaces behind the building.",
            "Klinika Demo has six free parking spaces",
            id="en-sixty-for-six",
        ),
        pytest.param(
            SESSION_SQ,
            "Seanca e parë zgjat rreth dyzet minuta.",
            "Seanca e parë zgjat rreth dyzet e pesë minuta",
            id="sq-forty-rounds-down-forty-five",
        ),
        pytest.param(
            SESSION_IT,
            "La prima seduta dura circa quaranta minuti.",
            "La prima seduta dura circa quarantacinque minuti",
            id="it-quaranta-rounds-down-quarantacinque",
        ),
        pytest.param(
            RETURNS_EN,
            "You can return an item within one week of delivery.",
            "return an item within fourteen days of delivery",
            id="en-one-week-for-fourteen-days",
        ),
        pytest.param(
            RETURNS_EN,
            "Mund ta ktheni artikullin brenda tridhjetë ditëve.",
            "return an item within fourteen days of delivery",
            id="sq-wrong-number-for-an-english-quote",
        ),
    ],
)
def test_a_number_word_the_quote_does_not_have_is_refused(
    content: str, text: str, quote: str
) -> None:
    assert _answer(content, text, quote).reasons == ("number_not_in_quote",)


@pytest.mark.parametrize(
    ("content", "text", "quote"),
    [
        pytest.param(
            RETURNS_EN,
            "You can return an item within fourteen days of delivery.",
            "return an item within fourteen days of delivery",
            id="the-same-words",
        ),
        pytest.param(
            RETURNS_EN,
            "You can return an item within 14 days of delivery.",
            "return an item within fourteen days of delivery",
            id="digits-for-words",
        ),
        pytest.param(
            REFUND_IT,
            "Il rimborso arriva entro quattordici giorni.",
            "Il rimborso arriva entro 14 giorni dalla consegna",
            id="words-for-digits",
        ),
        pytest.param(
            RETURNS_EN,
            "Puoi restituire l'articolo entro quattordici giorni dalla consegna.",
            "return an item within fourteen days of delivery",
            id="translated-it-from-en",
        ),
        pytest.param(
            PARKING_EN,
            "Klinika Demo ka gjashtë vende parkimi falas prapa ndërtesës.",
            "Klinika Demo has six free parking spaces",
            id="translated-sq-from-en",
        ),
        pytest.param(
            SESSION_SQ,
            "La prima seduta dura circa quarantacinque minuti.",
            "Seanca e parë zgjat rreth dyzet e pesë minuta",
            id="a-compound-translated-it-from-sq",
        ),
    ],
)
def test_the_same_number_in_any_form_or_language_is_accepted(
    content: str, text: str, quote: str
) -> None:
    result = _answer(content, text, quote)
    assert result.ok, result.reasons
