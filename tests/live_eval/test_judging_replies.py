"""The review gate reads the reply, not the case's declaration (Task 7 review, I2, M6, M11).

Design §1 counts a wrong answer that passed as one that contradicts the expected answer or holds a
fact the approved information does not say, and asks for no invented fact. The brief's code reviewed
an answer only when its case declared a document source (a proxy for "written from the documents")
and only in the last turn. What reaches the person is the reply: any text the model wrote, in any
turn, that is not a record answer or a text of the pack goes through the number, link and negation
check, and the invented-fact criterion is measured only when the model wrote some.
"""

from __future__ import annotations

import pytest

from frontdesk.conversation.behaviour import Behaviour
from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.records import TurnRecord
from frontdesk.live_eval.report import criteria, summarize
from frontdesk.live_eval.review import review_cases
from frontdesk.tenancy.pack import FixedTextKey
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, record
from tests.live_eval.outage_scenario import outage

INVENTED = "Ci sono sei posti, aperti 24 ore su 24."  # the 24 is nobody's
CLEAN = "Ci sono sei posti."
INVENTED_CRITERION = "Invented facts that passed the checks"


def invented(verdicts: tuple[CaseVerdict, ...]) -> tuple[str, str]:
    found = criteria(summarize(verdicts), spent_eur=None, cap_eur=None)
    return next((c.value, c.status) for c in found if c.name == INVENTED_CRITERION)


# --- the case's declaration does not decide what is reviewed -----------------------------------

UNDECLARED = [
    pytest.param(
        case("price", behaviour="answer", contains=["30 EUR"], sources=["price:physio"]),
        "Fisioterapia: 30 EUR, 25 per studenti. Aperti 24 ore.",
        ("parking-it",),
        id="a-case-that-expects-a-record-answer",
    ),
    pytest.param(
        case("nosrc", behaviour="answer", contains=["sei"]),
        INVENTED,
        ("parking-it",),
        id="a-case-that-names-no-source",
    ),
    pytest.param(
        case("phone", behaviour="handoff"),
        "Chiami il +39 055 123456.",
        ("contacts-it",),
        id="a-case-that-expects-a-handoff",
    ),
]


@pytest.mark.parametrize(("wanted", "reply", "sources"), UNDECLARED)
def test_a_model_answer_with_a_new_fact_is_reviewed_whatever_the_case_declares(
    wanted: TurnCase, reply: str, sources: tuple[str, ...]
) -> None:
    [verdict] = judge([wanted], [record(wanted.id, reply, sources=sources)], FIXED_TEXTS, {})
    assert verdict.dubious
    assert verdict.answer_pending
    assert review_cases([verdict]) == (verdict,)
    assert invented((verdict,)) == ("0", "pending")


@pytest.mark.parametrize(("wanted", "reply", "sources"), UNDECLARED)
def test_an_invented_fact_in_such_an_answer_fails_the_criterion_once_the_reviewer_says_so(
    wanted: TurnCase, reply: str, sources: tuple[str, ...]
) -> None:
    labels = {wanted.id: Label(answer="wrong", invented_fact=True)}
    [verdict] = judge([wanted], [record(wanted.id, reply, sources=sources)], FIXED_TEXTS, labels)
    assert verdict.invented_fact
    assert verdict.wrong_answer
    assert invented((verdict,)) == ("1", "fail")


def test_an_answer_to_a_case_that_expects_a_handoff_is_a_wrong_answer_and_is_reviewed() -> None:
    [verdict] = judge(
        [case("phone", behaviour="handoff")],
        [record("phone", "Chiami il +39 055 123456.", sources=("contacts-it",))],
        FIXED_TEXTS,
        {},
    )
    assert verdict.failures[0] == "behaviour: answer"
    assert verdict.wrong_answer
    assert verdict.dubious


# --- what is not the model's is not reviewed ---------------------------------------------------


def test_a_record_answer_is_never_reviewed_whatever_the_case_expects() -> None:
    # Prices and hours are composed by code from the approved data: nothing in them can be invented.
    wanted = case("handoff", behaviour="handoff")  # a case that expects a handoff, got a price
    sent = record("handoff", "Fisioterapia: 30 EUR, 5 sedute.", sources=("price:physio",))
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    assert verdict.answered
    assert not verdict.model_text
    assert not verdict.dubious


@pytest.mark.parametrize("key", [FixedTextKey.GREETING, FixedTextKey.PERSONAL_DATA])
def test_a_fixed_text_of_the_pack_is_not_the_model_s(key: FixedTextKey) -> None:
    text = "Ciao, non scrivere il 118: rispondo 24 ore su 24."
    texts = {language: {**TEXTS, key: text} for language in ("it", "en", "sq")}
    [verdict] = judge(
        [case("fixed", behaviour="answer")], [record("fixed", text, outcome="fixed")], texts, {}
    )
    assert verdict.observed is Behaviour.ANSWER  # the brief's convention: a notice is an answer
    assert verdict.answered
    assert not verdict.model_text
    assert not verdict.dubious


def test_a_reply_the_pipeline_marks_fixed_is_the_pack_s_text_by_construction() -> None:
    # The texts given to the judging may lack a key (a pack edited since, a test's partial texts):
    # a fixed outcome is still a text of the pack, never the model's.
    [verdict] = judge(
        [case("hello", behaviour="answer", intent="greeting")],
        [record("hello", "Ciao, sono aperto 24 ore su 24!", outcome="fixed", intent="greeting")],
        FIXED_TEXTS,
        {},
    )
    assert FixedTextKey.GREETING not in FIXED_TEXTS["it"]
    assert verdict.answered
    assert not verdict.model_text
    assert not verdict.dubious


def test_text_no_source_explains_is_the_model_s_so_a_lost_source_hides_nothing() -> None:
    wanted = case("nosrc", behaviour="answer", contains=["sei"])
    [new] = judge([wanted], [record("nosrc", INVENTED)], FIXED_TEXTS, {})
    [same] = judge([wanted], [record("nosrc", CLEAN)], FIXED_TEXTS, {})
    assert (new.model_text, new.dubious) == (True, True)
    assert (same.model_text, same.dubious) == (True, False)


def test_text_that_cites_a_document_and_a_record_is_the_model_s() -> None:
    wanted = case("mixed-sources", behaviour="answer", contains=["sei"])
    sent = record("mixed-sources", INVENTED, sources=("parking-it", "price:physio"))
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    assert verdict.model_text
    assert verdict.dubious


# --- every turn --------------------------------------------------------------------------------

CONVERSATION = case(
    "conv", behaviour="answer", contains=["sei"], sources=["parking-it"]
).model_copy(update={"message": None, "messages": ("posti?", "e di notte?")})
ENDING_IN_A_HANDOFF = case("conv", behaviour="handoff").model_copy(
    update={"message": None, "messages": ("posti?", "vorrei parlare con qualcuno")}
)


def conversation(
    first: str, last: str, *, first_sources: tuple[str, ...] = ("parking-it",), **fields: object
) -> list[TurnRecord]:
    return [
        record("conv", first, turn=1, turns=2, sources=first_sources),
        record("conv", last, turn=2, turns=2, **{"sources": ("parking-it",), **fields}),
    ]


def test_an_invented_fact_in_an_earlier_turn_is_reviewed() -> None:
    # The review's A2: the 24 reached the person in turn 1; turn 2 is clean and the case passes.
    [verdict] = judge([CONVERSATION], conversation(INVENTED, CLEAN), FIXED_TEXTS, {})
    assert verdict.passed
    assert verdict.dubious
    assert verdict.answer_pending
    assert review_cases([verdict]) == (verdict,)
    assert invented((verdict,)) == ("0", "pending")


def test_a_clean_earlier_turn_is_not_reviewed() -> None:
    [verdict] = judge([CONVERSATION], conversation(CLEAN, CLEAN), FIXED_TEXTS, {})
    assert not verdict.dubious


def test_an_earlier_record_answer_is_not_reviewed() -> None:
    first = conversation("Fisioterapia: 30 EUR, 5 sedute.", CLEAN, first_sources=("price:physio",))
    [verdict] = judge([CONVERSATION], first, FIXED_TEXTS, {})
    assert not verdict.dubious


def test_a_link_in_an_earlier_turn_is_reviewed_too() -> None:
    first = conversation("Ci sono sei posti: www.parcheggio.example", CLEAN)
    [verdict] = judge([CONVERSATION], first, FIXED_TEXTS, {})
    assert verdict.dubious


def test_an_invented_fact_the_reviewer_finds_in_an_earlier_turn_is_counted() -> None:
    labels = {"conv": Label(answer="wrong", invented_fact=True)}
    [verdict] = judge([CONVERSATION], conversation(INVENTED, CLEAN), FIXED_TEXTS, labels)
    assert verdict.invented_fact
    assert verdict.wrong_answer
    assert invented((verdict,)) == ("1", "fail")


def test_an_invented_fact_before_a_handoff_is_counted_though_the_last_reply_is_no_answer() -> None:
    # The conversation ends with the person sent to reception, and that is what the case expects.
    records = conversation(INVENTED, CONTACT, outcome="handoff")
    labels = {"conv": Label(answer="wrong", invented_fact=True)}
    [verdict] = judge([ENDING_IN_A_HANDOFF], records, FIXED_TEXTS, labels)
    assert verdict.observed is Behaviour.HANDOFF
    assert verdict.passed
    assert not verdict.answered_prefix
    assert verdict.answered  # the model wrote an answer in turn 1
    assert verdict.dubious
    assert verdict.invented_fact
    assert verdict.wrong_answer
    assert invented((verdict,)) == ("1", "fail")


def test_a_last_reply_that_is_no_answer_makes_no_wrong_answer_by_contradiction() -> None:
    # The case expects an answer; the person got one in turn 1 and a handoff in turn 2. The case
    # fails on what the last reply lacks, but no wrong answer reached the person.
    [verdict] = judge(
        [CONVERSATION], conversation(CLEAN, CONTACT, outcome="handoff"), FIXED_TEXTS, {}
    )
    assert "behaviour: handoff" in verdict.failures
    assert verdict.answered
    assert not verdict.wrong_answer


def test_a_document_question_answered_in_an_earlier_turn_is_not_lost() -> None:
    [verdict] = judge(
        [CONVERSATION], conversation(CLEAN, CONTACT, outcome="handoff"), FIXED_TEXTS, {}
    )
    assert verdict.document_question
    assert not verdict.unanswered_document_question


# --- the invented-fact criterion is measured on the model's answers -------------------------------


def test_a_responder_outage_leaves_the_invented_facts_unmeasured() -> None:
    # 0 of 15 document questions answered, a greeting and an emergency per language: no text of the
    # model reached anybody, so there is nothing to measure (it read PASS).
    verdicts = outage()
    assert invented(verdicts) == ("0", "pending")
    assert summarize(verdicts)["all"].model_answers == 0


def test_fixed_texts_and_record_answers_alone_measure_nothing() -> None:
    texts = {
        language: {
            **TEXTS,
            FixedTextKey.GREETING: "Come posso aiutarti?",
            FixedTextKey.PERSONAL_DATA: f"Non scrivere dati personali. {CONTACT}",
        }
        for language in ("it", "en", "sq")
    }
    cases = [
        case("hello", behaviour="answer", intent="greeting"),
        case("personal", behaviour="answer", outcome="fixed"),
        case("price", behaviour="answer", contains=["30 EUR"], sources=["price:physio"]),
    ]
    records = [
        record("hello", "Come posso aiutarti?", outcome="fixed", intent="greeting"),
        record("personal", f"Non scrivere dati personali. {CONTACT}", outcome="fixed"),
        record("price", "Fisioterapia: 30 EUR.", sources=("price:physio",)),
    ]
    verdicts = judge(cases, records, texts, {})
    assert all(v.answered for v in verdicts)
    assert invented(verdicts) == ("0", "pending")


def test_one_answer_the_model_wrote_measures_the_criterion() -> None:
    wanted = case("parking", behaviour="answer", contains=["sei"], sources=["parking-it"])
    [verdict] = judge(
        [wanted], [record("parking", CLEAN, sources=("parking-it",))], FIXED_TEXTS, {}
    )
    assert invented((verdict,)) == ("0", "pass")


def test_an_answer_the_model_wrote_in_an_earlier_turn_measures_it_too() -> None:
    [verdict] = judge(
        [ENDING_IN_A_HANDOFF], conversation(CLEAN, CONTACT, outcome="handoff"), FIXED_TEXTS, {}
    )
    assert invented((verdict,)) == ("0", "pass")


def test_the_summary_counts_the_answers_the_model_wrote() -> None:
    wanted = case("parking", behaviour="answer", contains=["sei"], sources=["parking-it"])
    price = case("price", behaviour="answer", contains=["30 EUR"], sources=["price:physio"])
    verdicts = judge(
        [wanted, price],
        [
            record("parking", CLEAN, sources=("parking-it",)),
            record("price", "Fisioterapia: 30 EUR.", sources=("price:physio",)),
        ],
        FIXED_TEXTS,
        {},
    )
    summary = summarize(verdicts)["it"]
    assert (summary.answered, summary.model_answers) == (2, 1)
