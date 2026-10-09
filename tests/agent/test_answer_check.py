import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import pytest

from frontdesk.agent.answer_check import (
    MAX_ANSWER_CHARS,
    AnswerCheck,
    check_answer,
    mentions_price,
    numbers_in,
    quote_in,
    to_plain_text,
)
from frontdesk.agent.responder import AnswerAction, Claim, ResponderOutput
from frontdesk.knowledge.embedders import Embedder, HashEmbedder, cosine
from frontdesk.knowledge.evidence import Evidence, EvidenceKind

PARKING = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="parking-en#1",
    version="v1",
    title="Parking",
    content="Klinika Demo has six free parking spaces behind the building. Street parking is paid on weekdays.",
)
VISIT = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="first-visit-it#1",
    version="v1",
    title="Cosa portare",
    content="Porta un documento d'identità e, se li hai, i referti degli ultimi 2 anni.",
)
PRICE = Evidence.create(
    kind=EvidenceKind.RECORD,
    source_ref="price:physiotherapy",
    version="v1",
    title="Prezzo",
    content="Fisioterapia: 30 EUR (a seduta)",
    values={"price": "30 EUR"},
)
ALL = [PARKING, VISIT, PRICE]


def _check(
    *claims: tuple[str, Evidence, str],
    action: str = "answer",
    clarify: str | None = None,
    evidence: Sequence[Evidence] = ALL,
) -> AnswerCheck:
    output = ResponderOutput.model_validate(
        {
            "action": action,
            "clarify": clarify,
            "claims": [{"text": t, "evidence_id": e.evidence_id, "quote": q} for t, e, q in claims],
        }
    )
    return check_answer(output, evidence, embedder=HashEmbedder(), min_similarity=0.5)


def test_a_quoted_sentence_passes() -> None:
    sentence = "Klinika Demo has six free parking spaces behind the building."
    result = _check((sentence, PARKING, sentence))
    assert result.ok
    assert result.text == sentence
    assert [item.evidence_id for item in result.cited] == [PARKING.evidence_id]


def test_numbers_must_come_from_the_quote() -> None:
    quote = "i referti degli ultimi 2 anni"
    assert _check(("Porta i referti degli ultimi 2 anni.", VISIT, quote)).ok
    bad = _check(("Porta i referti degli ultimi 3 anni.", VISIT, quote))
    assert "number_not_in_quote" in bad.reasons


def test_quotes_must_be_in_the_cited_passage() -> None:
    result = _check(
        ("Parking is free every day.", PARKING, "Parking is free every day of the week.")
    )
    assert "quote_not_in_source" in result.reasons


def test_passages_that_were_not_provided_are_refused_and_counted() -> None:
    sentence = "Physiotherapy costs thirty euros."
    other = Evidence.create(
        kind=EvidenceKind.DOCUMENT,
        source_ref="elsewhere#1",
        version="v",
        title="x",
        content=sentence,
    )
    result = _check((sentence, other, sentence))
    assert result.reasons == ("unknown_evidence", "no_claims")
    assert result.invalid_ids == 1
    assert result.cited == ()


def test_prices_never_come_from_documents() -> None:
    shipping = Evidence.create(
        kind=EvidenceKind.DOCUMENT,
        source_ref="shipping-it#1",
        version="v1",
        title="Spedizioni",
        content="La spedizione costa 5 euro in tutta Italia.",
    )
    quote = "La spedizione costa 5 euro"
    result = _check(("La spedizione costa 5 euro.", shipping, quote), evidence=[shipping])
    assert "price_in_answer" in result.reasons


def test_a_sentence_far_from_its_quote_is_unsupported() -> None:
    result = _check(
        (
            "Il parcheggio in strada è sempre gratuito.",
            PARKING,
            "Street parking is paid on weekdays.",
        )
    )
    assert result.reasons == ("unsupported_claim",)


def test_diagnostic_language_fails() -> None:
    result = _check(
        (
            "Potrebbe trattarsi di una tendinite: porta un documento d'identità.",
            VISIT,
            "Porta un documento d'identità",
        )
    )
    assert "diagnostic_language" in result.reasons


def test_clarify_and_handoff_carry_no_text() -> None:
    assert _check(action="clarify", clarify="service").ok
    assert _check(action="clarify").reasons == ("missing_clarify_field",)
    handoff = _check(action="handoff")
    assert handoff.ok
    assert handoff.text == ""


def test_helpers() -> None:
    assert numbers_in("dalle 8.00 alle 19:00, 30,00 € e 14 giorni") == {
        "08:00",
        "19:00",
        "30",
        "14",
    }
    assert mentions_price("costa 30 €")
    assert mentions_price("price: 2000 ALL")
    assert mentions_price("venti euro")
    assert not mentions_price("ALL our services are on the ground floor")
    assert quote_in(
        "documento d\u2019identità", "Porta un documento d'identità e i referti."
    )  # typographic apostrophe
    assert not quote_in("documento di identità", "Porta un documento d'identità e i referti.")
    assert to_plain_text("**Nota**: x\n\n\n\n# Titolo") == "Nota: x\n\nTitolo"


# Beyond the cases above: a claim must never pass because a check had nothing to compare.


# Words, because a quote may not start or end inside a word: 40 of them, the first 28 are quoted.
LONG = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="long#1",
    version="v1",
    title="Long",
    content=" ".join(["abcdefghi"] * 40),
)
LONG_QUOTE = " ".join(["abcdefghi"] * 28)
STRAY = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="elsewhere#1",
    version="v1",
    title="Elsewhere",
    content="A passage that was never provided.",
)
REPORTS = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="reports-it#1",
    version="v1",
    title="Referti",
    content="Porta i referti degli ultimi 1 e 2 anni.",
)
SHIPPING = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="shipping-it#1",
    version="v1",
    title="Spedizioni",
    content="La spedizione costa 5 euro in tutta Italia.",
)
STRAY_CLAIM = ("A passage that was never provided.", STRAY, "A passage that was never provided.")
FIRST = "Klinika Demo has six free parking spaces behind the building."
SECOND = "Street parking is paid on weekdays."
PARKING_CLAIM = (FIRST, PARKING, FIRST)
VISIT_CLAIM = ("Porta un documento.", VISIT, "Porta un documento d'identità")


class _Constant:
    """An embedder that gives every text one vector, so each similarity is that vector's own."""

    dimension = 2
    identity = "constant"

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def _run(
    claim: tuple[str, Evidence, str], *, embedder: Embedder, min_similarity: float
) -> AnswerCheck:
    text, passage, quote = claim
    output = ResponderOutput.model_validate(
        {
            "action": "answer",
            "claims": [{"text": text, "evidence_id": passage.evidence_id, "quote": quote}],
        }
    )
    return check_answer(output, [passage], embedder=embedder, min_similarity=min_similarity)


@pytest.mark.parametrize(
    "quote",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param(" " * 12, id="spaces"),
        pytest.param("\n\t \n", id="newlines-and-tabs"),
        pytest.param("\u00a0\u2003", id="no-break-space-and-em-space"),
    ],
)
def test_a_quote_that_is_empty_once_whitespace_is_collapsed_is_in_no_source(quote: str) -> None:
    # An empty string is "in" every string: without the guard any passage would "contain" it.
    assert not quote_in(quote, PARKING.content)
    assert not quote_in(quote, "")


def test_a_blank_quote_that_got_past_the_schema_still_fails_the_check() -> None:
    claim = Claim.model_construct(
        text="Parking is free.", evidence_id=PARKING.evidence_id, quote=" " * 12
    )
    output = ResponderOutput.model_construct(
        action=AnswerAction.ANSWER, claims=[claim], clarify=None
    )
    # An embedder that finds everything similar leaves the verbatim check alone to refuse it.
    result = check_answer(output, ALL, embedder=_Constant([1.0, 0.0]), min_similarity=0.5)
    assert result.reasons == ("quote_not_in_source",)


@pytest.mark.parametrize(
    "min_similarity", [0.0, 0.5], ids=["lowest-threshold", "default-threshold"]
)
@pytest.mark.parametrize(
    "embedder",
    [
        pytest.param(_Constant([0.0, 0.0]), id="zero-vector"),
        pytest.param(_Constant([math.nan, math.nan]), id="nan"),
        pytest.param(_Constant([math.inf, 1.0]), id="infinity"),
    ],
)
def test_a_similarity_that_is_zero_or_not_a_number_never_supports_a_claim(
    embedder: Embedder, min_similarity: float
) -> None:
    result = _run(PARKING_CLAIM, embedder=embedder, min_similarity=min_similarity)
    assert result.reasons == ("unsupported_claim",)


@pytest.mark.parametrize(
    "min_similarity", [0.0, 0.5], ids=["lowest-threshold", "default-threshold"]
)
def test_a_sentence_without_words_is_unsupported_even_at_the_lowest_threshold(
    min_similarity: float,
) -> None:
    # HashEmbedder gives text without tokens the zero vector, whose cosine is 0.0, not below 0.0.
    quote = "six free parking spaces behind the building"
    result = _run(("?!", PARKING, quote), embedder=HashEmbedder(), min_similarity=min_similarity)
    assert result.reasons == ("unsupported_claim",)


def test_a_similarity_equal_to_the_threshold_supports_a_claim() -> None:
    result = _run(PARKING_CLAIM, embedder=_Constant([1.0, 0.0]), min_similarity=1.0)
    assert result.ok


# What each check refuses, one case at a time.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", set(), id="empty"),
        pytest.param("nessun numero qui", set(), id="no-digits"),
        pytest.param("alle 8.00", {"08:00"}, id="time-with-a-dot"),
        pytest.param("alle 08:00", {"08:00"}, id="time-with-a-colon"),
        pytest.param("dalle 9.30 alle 17.45", {"09:30", "17:45"}, id="two-times"),
        pytest.param("fino alle 23:59", {"23:59"}, id="last-minute-of-the-day"),
        pytest.param("24.00", {"24"}, id="there-is-no-hour-24"),
        pytest.param("10.59", {"10:59"}, id="last-minute-of-the-hour"),
        pytest.param("10.60", {"10.6"}, id="sixty-minutes-is-a-decimal"),
        pytest.param("1.75 m", {"1.75"}, id="minutes-do-not-start-with-seven"),
        pytest.param("1.234 visite", {"1.234"}, id="three-digits-after-the-dot-are-no-time"),
        pytest.param(
            "1.000 pazienti", {"1.000"}, id="a-thousands-group-is-not-the-number-before-it"
        ),
        pytest.param("1,000 patients", {"1.000"}, id="the-comma-group-reads-as-the-dot-group"),
        pytest.param("2.500", {"2.500"}, id="2500-is-not-2.5"),
        pytest.param("10.000", {"10.000"}, id="10000-is-not-10"),
        pytest.param("1.000.000", {"1.000.000"}, id="several-groups-are-one-number"),
        pytest.param("12.345", {"12.345"}, id="a-group-after-two-digits"),
        pytest.param("123.456", {"123.456"}, id="a-group-after-three-digits"),
        pytest.param("0.500 kg", {"0.5"}, id="below-one-keeps-its-zero"),
        pytest.param("1.2345", {"1.2345"}, id="four-decimals-are-not-a-group"),
        pytest.param("1234.567", {"1234.567"}, id="a-group-needs-at-most-three-digits-before-it"),
        pytest.param("1234.500", {"1234.5"}, id="after-four-digits-it-is-a-decimal"),
        pytest.param("1.5.2024", {"1.5", "2024"}, id="a-date-is-still-split"),
        pytest.param("30,00 e 30.00 e 30", {"30"}, id="decimal-zeros-are-dropped"),
        pytest.param("1,5 e 1.5", {"1.5"}, id="comma-and-dot-decimals-agree"),
        pytest.param("007 e 0", {"7", "0"}, id="leading-zeros-are-dropped"),
        pytest.param(
            "10 e 100 e 1000", {"10", "100", "1000"}, id="trailing-zeros-of-integers-stay"
        ),
        pytest.param("2 anni e 3 mesi", {"2", "3"}, id="several-numbers"),
    ],
)
def test_numbers_are_read_in_one_canonical_form(text: str, expected: set[str]) -> None:
    assert numbers_in(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("costa 30 €", id="euro-sign-after"),
        pytest.param("€30", id="euro-sign-before"),
        pytest.param("$30", id="dollar-sign"),
        pytest.param("£30", id="pound-sign"),
        pytest.param("30 EUR", id="eur-code"),
        pytest.param("30 eur", id="eur-lower-case"),
        pytest.param("30 euro", id="euro"),
        pytest.param("30 euros", id="euros"),
        pytest.param("venti euro", id="euro-in-words"),
        pytest.param("2000 lek", id="lek"),
        pytest.param("2000 leke", id="leke"),
        pytest.param("2000 lekë", id="leke-with-diaeresis"),
        pytest.param("2000 ALL", id="all-code"),
        pytest.param("2000ALL", id="all-code-without-space"),
        pytest.param("30 USD", id="usd-code"),
        pytest.param("30 dollars", id="dollars"),
        pytest.param("1 dollar", id="dollar"),
    ],
)
def test_a_price_is_recognised_by_its_currency(text: str) -> None:
    assert mentions_price(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("ALL our services are on the ground floor", id="all-is-not-a-price"),
        pytest.param("all 3 floors", id="all-before-a-number"),
        pytest.param("open 7 all day", id="lower-case-all-after-a-number"),
        pytest.param("3 ALLERGIES", id="all-inside-a-word"),
        pytest.param("We are in Europe", id="euro-inside-a-word"),
        pytest.param("reparto neuro", id="euro-at-the-end-of-a-word"),
    ],
)
def test_text_without_a_currency_names_no_price(text: str) -> None:
    assert not mentions_price(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("plain", "plain", id="plain"),
        pytest.param("**bold**", "bold", id="bold"),
        pytest.param("__bold__", "bold", id="underscore-bold"),
        pytest.param("`code`", "code", id="code"),
        pytest.param("# Title", "Title", id="heading"),
        pytest.param("### Title", "Title", id="deep-heading"),
        pytest.param("  ## Title", "Title", id="indented-heading"),
        pytest.param("Stanza #5", "Stanza #5", id="hash-inside-a-line-stays"),
        pytest.param("a\n\nb", "a\n\nb", id="one-blank-line-stays"),
        pytest.param("a\n\n\nb", "a\n\nb", id="two-blank-lines-become-one"),
        pytest.param("  padded \n", "padded", id="edges-are-trimmed"),
    ],
)
def test_markdown_is_removed_from_what_is_sent(text: str, expected: str) -> None:
    assert to_plain_text(text) == expected


@pytest.mark.parametrize(
    ("quote", "source", "expected"),
    [
        pytest.param("six free parking", "six free parking spaces", True, id="prefix"),
        pytest.param("free parking spaces", "six free parking spaces", True, id="suffix"),
        pytest.param("six free parking spaces", "six free parking spaces", True, id="whole-source"),
        pytest.param("six   free\nparking\t", "six free parking spaces", True, id="quote-spacing"),
        pytest.param("six free parking", "six  free\n\nparking spaces", True, id="source-spacing"),
        pytest.param("Six free parking", "six free parking spaces", False, id="case-differs"),
        pytest.param("six free parking spaces!", "six free parking spaces", False, id="longer"),
        pytest.param("six parking spaces", "six free parking spaces", False, id="word-missing"),
        pytest.param("free six parking", "six free parking spaces", False, id="words-reordered"),
        # The model may copy a typographic mark for the plain one, or the reverse.
        pytest.param("d\u2019identità", "d'identità", True, id="right-single-quote"),
        pytest.param("d\u2018identità", "d'identità", True, id="left-single-quote"),
        pytest.param("d'identità", "d\u2019identità", True, id="source-has-the-typographic-one"),
        pytest.param("\u201cciao\u201d e basta", '"ciao" e basta', True, id="double-quotes"),
        pytest.param('"ciao" e basta', "\u201cciao\u201d e basta", True, id="source-double-quotes"),
        # render_evidence shows < and > as angle quotes, so a quote copied from the prompt has them.
        pytest.param("\u2039b\u203a in grassetto", "usa <b> in grassetto", True, id="angle-quotes"),
        pytest.param(
            "usa <b> in grassetto", "usa \u2039b\u203a in grassetto", True, id="source-angle-quotes"
        ),
    ],
)
def test_a_quote_is_found_word_for_word_apart_from_spacing_and_quote_marks(
    quote: str, source: str, expected: bool
) -> None:
    assert quote_in(quote, source) is expected


# What check_answer reports for answers, clarifications and handoffs.


def test_an_answer_without_claims_is_refused() -> None:
    result = _check()
    assert not result.ok
    assert result.reasons == ("no_claims",)
    assert (result.text, result.cited, result.invalid_ids) == ("", (), 0)


def test_unknown_passages_are_dropped_and_counted_while_known_ones_are_still_checked() -> None:
    result = _check((FIRST, PARKING, FIRST), STRAY_CLAIM)
    assert not result.ok
    assert result.reasons == ("unknown_evidence",)
    assert result.invalid_ids == 1
    assert result.text == FIRST
    assert result.cited == (PARKING,)


def test_a_record_cannot_be_cited_even_under_an_id_that_looks_like_a_document() -> None:
    record = PRICE.model_copy(update={"evidence_id": "D-00000001"})
    result = _check(("Fisioterapia costa 30.", record, "Fisioterapia: 30 EUR"), evidence=[record])
    assert result.reasons == ("unknown_evidence", "no_claims")
    assert result.invalid_ids == 1
    assert result.cited == ()


@pytest.mark.parametrize(
    ("action", "clarify", "reasons"),
    [
        pytest.param("clarify", "service", ("unknown_evidence",), id="clarify"),
        pytest.param("clarify", None, ("unknown_evidence", "missing_clarify_field"), id="no-field"),
        pytest.param("handoff", None, ("unknown_evidence",), id="handoff"),
    ],
)
def test_unknown_passages_are_counted_when_the_answer_is_not_an_answer(
    action: str, clarify: str | None, reasons: tuple[str, ...]
) -> None:
    result = _check(STRAY_CLAIM, action=action, clarify=clarify)
    assert not result.ok
    assert result.reasons == reasons
    assert result.invalid_ids == 1
    assert (result.text, result.cited) == ("", ())


@pytest.mark.parametrize(
    ("action", "clarify"),
    [pytest.param("clarify", "service", id="clarify"), pytest.param("handoff", None, id="handoff")],
)
def test_claims_that_come_with_a_clarification_or_handoff_are_not_sent(
    action: str, clarify: str | None
) -> None:
    result = _check((FIRST, PARKING, FIRST), action=action, clarify=clarify)
    assert result.ok
    assert (result.text, result.cited, result.invalid_ids) == ("", (), 0)


def test_claims_are_joined_with_a_space() -> None:
    result = _check((FIRST, PARKING, FIRST), VISIT_CLAIM, (SECOND, PARKING, SECOND))
    assert result.ok
    assert result.text == f"{FIRST} Porta un documento. {SECOND}"


@pytest.mark.parametrize(
    ("claims", "cited"),
    [
        pytest.param([PARKING_CLAIM, VISIT_CLAIM, PARKING_CLAIM], (PARKING, VISIT), id="parking"),
        pytest.param([VISIT_CLAIM, PARKING_CLAIM, VISIT_CLAIM], (VISIT, PARKING), id="visit"),
        pytest.param([PARKING_CLAIM, PARKING_CLAIM], (PARKING,), id="the-same-passage-twice"),
    ],
)
def test_each_cited_passage_is_listed_once_in_the_order_it_is_first_cited(
    claims: list[tuple[str, Evidence, str]], cited: tuple[Evidence, ...]
) -> None:
    result = _check(*claims)
    assert result.ok
    assert result.cited == cited


def test_every_claim_must_be_close_to_its_own_quote() -> None:
    far = ("Il parcheggio in strada è sempre gratuito.", PARKING, SECOND)
    assert _check(PARKING_CLAIM, far).reasons == ("unsupported_claim",)
    assert _check(far, PARKING_CLAIM).reasons == ("unsupported_claim",)


def test_some_similarity_is_not_enough_it_has_to_reach_the_threshold() -> None:
    sentence = "Street cleaning happens monthly."  # one word in common with the quote
    vectors = HashEmbedder().embed([sentence, SECOND])
    assert 0 < cosine(vectors[0], vectors[1]) < 0.5
    assert _check((sentence, PARKING, SECOND)).reasons == ("unsupported_claim",)


def test_a_shorter_sentence_made_of_words_from_its_quote_is_supported() -> None:
    assert _check(("Street parking is paid.", PARKING, SECOND)).ok


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        pytest.param(
            (
                "Porta i referti degli ultimi 1**2 anni.",
                REPORTS,
                "i referti degli ultimi 1 e 2 anni",
            ),
            "number_not_in_quote",
            id="two-numbers-become-one",
        ),
        pytest.param(
            ("La spedizione costa 5 eu**ro**.", SHIPPING, "La spedizione costa 5 euro"),
            "price_in_answer",
            id="a-currency-is-completed",
        ),
        pytest.param(
            ("Potrebbe tra**ttarsi** di una tendinite.", VISIT, "Porta un documento d'identità"),
            "diagnostic_language",
            id="a-phrase-is-completed",
        ),
    ],
)
def test_checks_read_the_text_that_is_sent_so_markdown_cannot_hide_a_problem(
    claim: tuple[str, Evidence, str], reason: str
) -> None:
    assert reason in _check(claim, evidence=[REPORTS, SHIPPING, VISIT]).reasons


def test_markdown_is_removed_from_the_text_that_is_returned() -> None:
    result = _check(
        ("**Klinika Demo** has six free parking spaces behind the building.", PARKING, FIRST)
    )
    assert result.ok
    assert result.text == FIRST


def _long(length: int) -> tuple[str, Evidence, str]:
    """A claim of `length` characters quoting LONG_QUOTE; full stops make up the length."""
    return (LONG_QUOTE + "." * (length - len(LONG_QUOTE)), LONG, LONG_QUOTE)


def test_the_answer_may_be_as_long_as_the_limit_and_no_longer() -> None:
    assert MAX_ANSWER_CHARS == 600
    at_limit = _check(_long(299), _long(300), evidence=[LONG])
    assert at_limit.ok
    assert len(at_limit.text) == 600
    over = _check(_long(300), _long(300), evidence=[LONG])
    assert over.reasons == ("too_long",)
    assert len(over.text) == 601


def test_the_length_limit_counts_what_is_sent_not_the_markdown() -> None:
    _, _, quote = _long(296)
    marked_up = "**" + quote + "." * 17 + "**"  # 300 characters written, 296 sent
    result = _check(_long(300), (marked_up, LONG, quote), evidence=[LONG])
    assert result.ok
    assert len(result.text) == 597


def test_reasons_come_in_the_order_the_checks_run_and_each_comes_once() -> None:
    failing = (
        "Potrebbe trattarsi di una tendinite, costa 99 euro.",
        PARKING,
        "Parking is free every day of the week.",
    )
    result = _check(
        STRAY_CLAIM,
        failing,
        _long(280),
        _long(280),
        evidence=[PARKING, LONG],
    )
    assert result.reasons == (
        "unknown_evidence",
        "quote_not_in_source",
        "number_not_in_quote",
        "price_in_answer",
        "diagnostic_language",
        "unsupported_claim",
        "too_long",
    )
    twice = _check(
        ("Parking is free.", PARKING, "Parking is free."),
        ("Parking is free.", PARKING, "Parking is free."),
    )
    assert twice.reasons == ("quote_not_in_source",)


def test_a_check_result_cannot_be_changed() -> None:
    result = _check()
    with pytest.raises(FrozenInstanceError):
        result.ok = True  # type: ignore[misc]


THOUSAND = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="car-park-it#1",
    version="v1",
    title="Parcheggio",
    content="Il parcheggio ha 1.000 posti dietro l'edificio.",
)


@pytest.mark.parametrize(
    ("text", "reasons"),
    [
        pytest.param("Il parcheggio ha 1.000 posti.", (), id="as-written"),
        pytest.param("The car park has 1,000 spaces.", (), id="with-the-english-comma"),
        pytest.param(
            "Il parcheggio ha 1 posti.", ("number_not_in_quote",), id="one-for-a-thousand"
        ),
        pytest.param(
            "Il parcheggio ha 1000 posti.", ("number_not_in_quote",), id="without-the-dot"
        ),
        pytest.param("Il parcheggio ha 2.500 posti.", ("number_not_in_quote",), id="another-group"),
    ],
)
def test_a_thousands_group_supports_only_itself(text: str, reasons: tuple[str, ...]) -> None:
    # In Italian and Albanian "1.000" is a thousand, in English it would be one: the check does not
    # guess, so a claim must write the group as its quote does (a comma for a dot is the same group).
    claim = (text, THOUSAND, "ha 1.000 posti dietro l'edificio")
    result = _run(claim, embedder=_Constant([1.0, 0.0]), min_similarity=0.5)
    assert result.reasons == reasons
