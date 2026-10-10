"""What makes an answer wrong, and what makes it dubious (M3 design §1, §4.4).

A wrong answer that passed reached the person and contradicts the expected answer: a text it lacks,
a text it must not hold, another behaviour. The other checks (a source, the outcome, the intent) can
fail without the answer being wrong. A dubious answer adds a number, a link or a negation that the
expected answer does not hold, in any of the three languages.
"""

from __future__ import annotations

import pytest

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, judge
from frontdesk.tenancy.pack import FixedTextKey
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, record

PARKING = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])


def verdict_of(wanted: TurnCase, reply: str, **fields: object) -> CaseVerdict:
    [verdict] = judge([wanted], [record(wanted.id, reply, **fields)], FIXED_TEXTS, {})
    return verdict


# --- wrong: what the answer contradicts ----------------------------------------------------------


def test_a_text_the_answer_must_not_hold_makes_it_wrong() -> None:
    wanted = case("careful", behaviour="answer", not_contains=["24 hours"])
    verdict = verdict_of(wanted, "Open 24 hours.")
    assert verdict.failures == ("unexpected text: 24 hours",)
    assert verdict.wrong_answer


def test_another_behaviour_than_the_expected_one_makes_an_answer_wrong() -> None:
    # The case expects the handoff; the bot answered instead.
    wanted = case("handoff", behaviour="handoff")
    verdict = verdict_of(wanted, "Open 24 hours.")
    assert verdict.failures == ("behaviour: answer",)
    assert verdict.wrong_answer


def test_a_missing_text_makes_an_answer_wrong() -> None:
    verdict = verdict_of(PARKING, "There is a car park.", sources=("parking-en",))
    assert verdict.failures == ("missing text: six",)
    assert verdict.wrong_answer


@pytest.mark.parametrize(
    ("wanted", "fields", "failure"),
    [
        pytest.param(PARKING, {}, "missing source: parking-en", id="source"),
        pytest.param(
            case("fixed", behaviour="answer", outcome="fixed"),
            {},
            "outcome: replied",
            id="outcome",
        ),
        pytest.param(
            case("hours", behaviour="answer", intent="hours"),
            {"intent": "price"},
            "intent: price",
            id="intent",
        ),
        pytest.param(
            case("italian", behaviour="answer", language="it"),
            {"language": "en"},
            "language: en",
            id="language",
        ),
        pytest.param(
            case("rule", behaviour="answer", rule="emergency"), {}, "rule: pass", id="rule"
        ),
    ],
)
def test_a_failed_check_that_does_not_contradict_the_answer_leaves_it_right(
    wanted: TurnCase, fields: dict[str, object], failure: str
) -> None:
    verdict = verdict_of(wanted, "There are six free spaces.", **fields)
    assert failure in verdict.failures
    assert verdict.answered
    assert not verdict.wrong_answer


def test_a_reply_with_diagnostic_language_or_a_promise_is_not_a_contradiction() -> None:
    wanted = case("promise", behaviour="answer")
    verdict = verdict_of(wanted, "We will call you back tomorrow.")
    assert any(failure.startswith("promise:") for failure in verdict.failures)
    assert not verdict.wrong_answer


# --- dubious: what the answer adds ----------------------------------------------------------------


def test_only_a_question_the_documents_answer_has_answers_to_review() -> None:
    # A price from the records is composed by code: nothing in it can be invented.
    prices = case("prices", behaviour="answer", contains=["30 EUR"], sources=["price:physio"])
    verdict = verdict_of(prices, "Physiotherapy: 30 EUR, 5 sessions.", sources=("price:physio",))
    assert verdict.answered
    assert not verdict.dubious


def test_a_link_both_the_answer_and_the_expectation_hold_is_not_new() -> None:
    wanted = case(
        "link", behaviour="answer", contains=["www.parking.example"], sources=["parking-en"]
    )
    same = verdict_of(wanted, "Book at www.parking.example.", sources=("parking-en",))
    other = verdict_of(
        wanted, "Book at www.parking.example or www.other.example", sources=("parking-en",)
    )
    assert not same.dubious
    assert other.dubious


def test_a_negation_both_the_answer_and_the_expectation_hold_is_not_new() -> None:
    wanted = case(
        "card", behaviour="answer", contains=["non si paga con carta"], sources=["payments-it"]
    )
    same = verdict_of(wanted, "Non si paga con carta.", sources=("payments-it",))
    added = verdict_of(wanted, "Non si paga con carta, mai.", sources=("payments-it",))
    assert not same.dubious
    assert added.dubious


@pytest.mark.parametrize(
    ("language", "reply", "expected"),
    [
        pytest.param("en", "There are six spaces, but you can't book them.", "six", id="en-cant"),
        pytest.param("en", "There are six spaces and no roof.", "six", id="en-no"),
        pytest.param("sq", "Ka gjashtë vende, por s'ka çati.", "gjashtë", id="sq-s-ka"),
        pytest.param("sq", "Ka gjashtë vende, por nuk ka çati.", "gjashtë", id="sq-nuk"),
        pytest.param("it", "Ci sono sei posti, ma non il tetto.", "sei", id="it-non"),
    ],
)
def test_a_negation_the_expectation_lacks_is_dubious_in_every_language(
    language: str, reply: str, expected: str
) -> None:
    wanted = case("parking", behaviour="answer", contains=[expected], sources=["parking-xx"])
    verdict = verdict_of(wanted, reply, language=language, sources=("parking-xx",))
    assert verdict.dubious


def test_an_answer_that_holds_only_what_is_expected_is_not_dubious_in_any_language() -> None:
    for language, reply, expected in [
        ("en", "There are six spaces.", "six"),
        ("sq", "Ka gjashtë vende.", "gjashtë"),
        ("it", "Ci sono sei posti.", "sei"),
    ]:
        wanted = case("parking", behaviour="answer", contains=[expected], sources=["parking-xx"])
        verdict = verdict_of(wanted, reply, language=language, sources=("parking-xx",))
        assert not verdict.dubious, language


def test_a_fixed_text_taken_out_of_the_reply_does_not_join_the_words_around_it() -> None:
    # "1", then the handoff text, then "2": taken out without a space they would read as "12".
    wanted = case("twelve", behaviour="answer", contains=["12"], sources=["parking-it"])
    reply = f"Aperto dalle 1{CONTACT}2 ore"
    verdict = verdict_of(wanted, reply, sources=("parking-it",))
    assert verdict.dubious


# --- an answer ahead of a notice, from documents and records together -----------------------------


def test_text_that_cites_a_document_and_a_record_may_be_the_model_s() -> None:
    mixed = case("mixed", behaviour="clinical", contains=["Non posso dare consigli medici"])
    reply = f"Ci sono 6 posti, 30 EUR.\n\n{TEXTS[FixedTextKey.CLINICAL]}\n\n{CONTACT}"
    verdict = verdict_of(
        mixed,
        reply,
        outcome="handoff",
        category="clinical",
        sources=("parking-it", "price:physio"),
    )
    assert verdict.answered_prefix
    assert verdict.dubious
