"""A quote is whole words and was not cut out of a negation (ADR-0009)."""

import re
import unicodedata
from collections.abc import Sequence

import pytest

from frontdesk.agent.answer_check import NEGATION_REACH, AnswerCheck, check_answer, quote_in
from frontdesk.agent.responder import ResponderOutput
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.tenancy.pack import BusinessPack

APOSTROPHE = chr(0x2019)
SOFT_HYPHEN = chr(0x00AD)
ZERO_WIDTH_SPACE = chr(0x200B)


class _Similar:
    """Finds every text identical to every other, so that only the quote check decides."""

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


# --- whole words and numbers


@pytest.mark.parametrize(
    ("quote", "source"),
    [
        pytest.param(
            "available on weekdays",
            "The parking lot is unavailable on weekdays.",
            id="starts-inside-a-word",
        ),
        pytest.param("we can", "we cannot accept cards today", id="ends-inside-a-word"),
        pytest.param(
            "4 giorni dalla consegna",
            "Il rimborso arriva entro 14 giorni dalla consegna",
            id="starts-inside-a-number",
        ),
        pytest.param("entro 1", "arriva entro 14 giorni", id="ends-inside-a-number"),
        pytest.param(
            "30 to 18.00 on weekdays",
            "open from 10.30 to 18.00 on weekdays",
            id="starts-inside-a-time",
        ),
        pytest.param("Open until 10", "Open until 10:30 today", id="ends-inside-a-time"),
        pytest.param(
            "05 per la visita", "Scrivici entro il 12/05 per la visita", id="starts-inside-a-date"
        ),
        pytest.param("5 ore di attesa", "dura 1.5 ore di attesa", id="starts-inside-a-decimal"),
        pytest.param(
            "000 posti dietro",
            "ha 1.000 posti dietro l'edificio",
            id="starts-inside-a-thousands-group",
        ),
        pytest.param(
            "18.00 on weekdays", "open 08.00-18.00 on weekdays", id="starts-inside-a-range"
        ),
        pytest.param(
            "refundable within 14 days",
            "Items are non-refundable within 14 days",
            id="starts-inside-a-hyphenated-word",
        ),
        pytest.param(
            "mail di conferma", "ti mandiamo una e-mail di conferma", id="starts-inside-e-mail"
        ),
    ],
)
def test_a_quote_may_not_start_or_end_inside_a_word_or_a_number(quote: str, source: str) -> None:
    assert not quote_in(quote, source)


@pytest.mark.parametrize(
    ("quote", "source"),
    [
        pytest.param("accept cards today", "We accept cards today", id="to-the-end"),
        pytest.param("We accept cards", "We accept cards today", id="from-the-start"),
        pytest.param(
            "14 giorni dalla consegna", "entro 14 giorni dalla consegna", id="a-whole-number"
        ),
        pytest.param("10.30 to 18.00", "open from 10.30 to 18.00 on weekdays", id="whole-times"),
        pytest.param(
            "1.000 posti", "ha 1.000 posti dietro l'edificio", id="a-whole-thousands-group"
        ),
        pytest.param(
            "identità e i referti",
            "Porta un documento d'identità e i referti.",
            id="after-an-elision",
        ),
        pytest.param(
            "non-refundable within 14 days",
            "Items are non-refundable within 14 days",
            id="a-whole-hyphenated-word",
        ),
    ],
)
def test_whole_words_and_numbers_can_be_quoted(quote: str, source: str) -> None:
    assert quote_in(quote, source)


# --- negations just before the quote


@pytest.mark.parametrize(
    ("quote", "source"),
    [
        pytest.param("accept cards today", "We do not accept cards today", id="en-not"),
        pytest.param("accept cards today", "We cannot accept cards today", id="en-cannot"),
        pytest.param("accept cards today", "We don't accept cards today", id="en-dont"),
        pytest.param(
            "accept cards today",
            f"Sorry, we can{APOSTROPHE}t accept cards today",
            id="en-cant-with-a-typographic-apostrophe",
        ),
        pytest.param(
            "refunds after the first month", "No refunds after the first month", id="en-no"
        ),
        pytest.param(
            "a prior appointment", "Visits without a prior appointment are welcome", id="en-without"
        ),
        pytest.param("open on sundays", "We are never open on sundays", id="en-never"),
        pytest.param("free for patients", "Parking is not free for patients", id="en-after-a-verb"),
        pytest.param(
            "accettiamo carte di credito", "Non accettiamo carte di credito.", id="it-non"
        ),
        pytest.param(
            "pagare con carta", "Non è possibile pagare con carta.", id="it-non-and-two-words"
        ),
        pytest.param(
            "disponibile il sabato",
            "Il parcheggio non è disponibile il sabato.",
            id="it-in-the-middle-of-a-sentence",
        ),
        pytest.param(
            "parcheggio vicino alla clinica",
            "Non c'è parcheggio vicino alla clinica.",
            id="it-non-c-e",
        ),
        pytest.param("accettiamo carte", "NON accettiamo carte", id="upper-case"),
        pytest.param(
            "pranojmë kartën bankare", "Nuk e pranojmë kartën bankare.", id="sq-nuk-and-a-clitic"
        ),
        pytest.param("ka parkim falas", "S'ka parkim falas për pacientët.", id="sq-s-apostrophe"),
        pytest.param(
            "ka parkim falas",
            f"S{APOSTROPHE}ka parkim falas për pacientët.",
            id="sq-s-typographic-apostrophe",
        ),
        pytest.param("hyni me këpucë", "Mos hyni me këpucë në sallë.", id="sq-mos"),
        pytest.param(
            "accettiamo carte",
            "Non accettiamo carte. Poi, accettiamo carte solo in sede.",
            id="one-of-two-places-is-negated",
        ),
        pytest.param(
            "accettiamo carte",
            "Qui accettiamo carte. Altrove non accettiamo carte.",
            id="the-second-of-two-places-is-negated",
        ),
    ],
)
def test_a_quote_cut_out_of_a_negation_is_not_in_the_source(quote: str, source: str) -> None:
    assert not quote_in(quote, source)


@pytest.mark.parametrize(
    ("quote", "source"),
    [
        pytest.param(
            "accettiamo contanti e bancomat",
            "Non accettiamo carte, ma accettiamo contanti e bancomat.",
            id="a-comma-ends-the-reach",
        ),
        pytest.param(
            "Porta un documento valido", "Non fumare. Porta un documento valido.", id="a-full-stop"
        ),
        pytest.param(
            "Porta un documento valido",
            "- Non fumare\n- Porta un documento valido",
            id="a-list-item",
        ),
        pytest.param(
            "porta un documento valido", "Non fumare: porta un documento valido", id="a-colon"
        ),
        pytest.param(
            "non accettiamo carte di credito",
            "Qui non accettiamo carte di credito.",
            id="the-negation-is-in-the-quote",
        ),
        pytest.param(
            "accettiamo cinque per volta",
            "Abbiamo posto: ne accettiamo cinque per volta.",
            id="it-ne-is-not-a-negation",
        ),
        pytest.param(
            "parking is free", "The clinic's parking is free", id="a-possessive-is-not-s-apostrophe"
        ),
        pytest.param("parking is free", "The dentists' parking is free", id="a-plural-possessive"),
        pytest.param("che accettiamo carte", "Nota che accettiamo carte", id="nota-is-not-no"),
    ],
)
def test_a_quote_the_negation_does_not_reach_can_be_quoted(quote: str, source: str) -> None:
    assert quote_in(quote, source)


def test_a_negation_reaches_four_words_and_no_further() -> None:
    assert NEGATION_REACH == 4
    quote = "free parking for patients"
    assert not quote_in(quote, "not a b c free parking for patients")  # the fourth word before
    assert quote_in(quote, "not a b c d free parking for patients")  # the fifth
    assert not quote_in("pagare con carta", "Non è sempre possibile pagare con carta")


def _words(text: str) -> list[str]:
    return text.split()


NEGATION_WORDS = _words(
    "not no never without none nothing nobody neither nor cannot "
    "non né nè mai senza nessun nessuno nessuna niente nulla neanche nemmeno neppure "
    "nuk mos jo pa asnjë asnje askush asgjë asgje kurrë kurre"
)
NOT_NEGATIONS = _words("ne nota note now nuke pasta jolly moscow nonna nobel nothings")


@pytest.mark.parametrize("word", NEGATION_WORDS)
def test_every_negation_word_keeps_a_quote_out_of_its_sentence(word: str) -> None:
    assert not quote_in("is open on weekdays", f"The clinic {word} is open on weekdays")


@pytest.mark.parametrize("word", NOT_NEGATIONS)
def test_words_that_only_look_like_a_negation_do_not(word: str) -> None:
    assert quote_in("is open on weekdays", f"The clinic {word} is open on weekdays")


# --- how text is compared


def test_composed_and_decomposed_letters_are_the_same() -> None:
    composed = "Klinikës Demo ka parking falas"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert quote_in(decomposed, composed)
    assert quote_in(composed, decomposed)


def test_soft_hyphens_and_zero_width_characters_are_not_part_of_the_text() -> None:
    assert quote_in("parcheggio è gratuito", f"Il parcheggio è gratu{SOFT_HYPHEN}ito")
    assert quote_in(f"parcheggio è gratu{ZERO_WIDTH_SPACE}ito", "Il parcheggio è gratuito")
    # ... and cannot hide that a quote starts inside a word, or make a quote of nothing.
    assert not quote_in("able on weekdays", f"unavail{SOFT_HYPHEN}able on weekdays")
    assert not quote_in(ZERO_WIDTH_SPACE * 12, f"x {ZERO_WIDTH_SPACE * 12} y")


# --- through check_answer


def test_a_dropped_negation_or_a_sliced_number_is_a_quote_not_in_the_source() -> None:
    dropped = _answer(
        "We do not accept cards today.", "We accept cards today.", "accept cards today"
    )
    assert dropped.reasons == ("quote_not_in_source",)
    sliced = _answer(
        "Il rimborso arriva entro 14 giorni dalla consegna del pacco.",
        "Il rimborso arriva in 4 giorni dalla consegna del pacco.",
        "4 giorni dalla consegna del pacco",
    )
    # The 4 is in the quote, so only the verbatim check can see that it is not in the passage.
    assert sliced.reasons == ("quote_not_in_source",)


def test_a_quote_from_the_middle_of_a_sentence_still_passes() -> None:
    content = "Non accettiamo carte, ma accettiamo contanti e bancomat in tutta la clinica."
    result = _answer(
        content, "Accettiamo contanti e bancomat.", "accettiamo contanti e bancomat in tutta"
    )
    assert result.ok, result.reasons


# --- the quotes the demo models will make


def _passages(pack: BusinessPack) -> list[tuple[str, str]]:
    return [
        (f"{document.key}#{chunk.ordinal}", chunk.content)
        for document in pack.documents
        for chunk in chunk_markdown(document.title, document.body)
    ]


def _first_sentence_claims(content: str) -> list[tuple[str, str]]:
    """(text, quote) as the stand-in responders write them: the first sentence of the passage."""
    by_punctuation = re.split(r"(?<=[.!?])\s+", content.strip())[0]
    by_full_stop = content.split(". ")[0].rstrip(".")
    return [(by_punctuation, by_punctuation), (f"{by_full_stop}.", by_full_stop)]


@pytest.mark.parametrize("pack_fixture", ["clinic_pack", "shop_pack"])
def test_the_first_sentence_of_every_demo_passage_passes_the_claim_checks(
    pack_fixture: str, request: pytest.FixtureRequest
) -> None:
    pack: BusinessPack = request.getfixturevalue(pack_fixture)
    passages = _passages(pack)
    assert passages
    refused: dict[str, tuple[str, ...]] = {}
    for source_ref, content in passages:
        passage = Evidence.create(
            kind=EvidenceKind.DOCUMENT,
            source_ref=source_ref,
            version="v1",
            title=source_ref,
            content=content,
        )
        for text, quote in _first_sentence_claims(content):
            output = ResponderOutput.model_validate(
                {
                    "action": "answer",
                    "claims": [{"text": text, "evidence_id": passage.evidence_id, "quote": quote}],
                }
            )
            result = check_answer(output, [passage], embedder=HashEmbedder(), min_similarity=0.5)
            if not result.ok:
                refused[f"{source_ref}: {quote}"] = result.reasons
    assert refused == {}
