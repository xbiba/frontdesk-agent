"""A question the documents answer that nobody answered counts against the 20% criterion.

The criterion counts the right answers the checks refused, over the questions the documents answer.
A right answer is lost just as surely when no check ever saw it: the classifier slipped, the model
failed or chose to hand off, the message went to the wrong place. Counting only the refused read PASS
on a run that answered nothing (Task 7 review, I1: W1, a handoff the model chose, a refused case that
names no document, a clinical message with a document question).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from frontdesk.conversation.behaviour import Behaviour
from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.records import TurnRecord
from frontdesk.live_eval.report import criteria, render_report, report_json, summarize
from frontdesk.live_eval.review import review_cases
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, draft, manifest, record
from tests.live_eval.outage_scenario import outage

PARKING = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
SIX = "There are six free spaces."
NO_INFORMATION = f"{TEXTS['no_information']}\n\n{CONTACT}"
NOTICE = f"{TEXTS['clinical']}\n\n{CONTACT}"
REFUSED_DRAFT = draft(SIX, "six free parking spaces")
REFUSED = {"check_reasons": ("quote_not_in_source",), "draft": REFUSED_DRAFT}
CRITERION = "Right answers refused (it)"


def lost_run(
    *, answered: int = 0, refused_wrong: int = 0, refused_unlabelled: int = 0, lost: int = 0
) -> tuple[CaseVerdict, ...]:
    """Document questions, one case each: right answers, answers a check refused and the reviewer
    calls wrong, refused and still unlabelled, and questions nobody answered (a responder failure)."""
    kinds = (
        ["answered"] * answered
        + ["wrong"] * refused_wrong
        + ["unlabelled"] * refused_unlabelled
        + ["lost"] * lost
    )
    cases: list[TurnCase] = []
    records: list[TurnRecord] = []
    labels: dict[str, Label] = {}
    for number, kind in enumerate(kinds):
        name = f"d{number}"
        cases.append(case(name, behaviour="answer", contains=["six"], sources=["parking-en"]))
        if kind == "answered":
            records.append(record(name, SIX, sources=("parking-en",)))
        elif kind == "lost":
            records.append(record(name, CONTACT, outcome="handoff", rejection="responder_error"))
        else:
            records.append(record(name, NO_INFORMATION, outcome="handoff", **REFUSED))
        if kind == "wrong":
            labels[name] = Label(refusal="wrong")
    return judge(cases, records, FIXED_TEXTS, labels)


def criterion(verdicts: tuple[CaseVerdict, ...], name: str = CRITERION) -> tuple[str, str]:
    found = criteria(summarize(verdicts), spent_eur=None, cap_eur=None)
    return next((c.value, c.status) for c in found if c.name == name)


# --- every way a right answer is lost without a check refusing it -------------------------------

LOST = [
    pytest.param(
        {"reply": CONTACT, "outcome": "handoff", "rejection": "unknown_service_id"},
        id="classifier-slip",
    ),
    pytest.param(
        {"reply": CONTACT, "outcome": "handoff", "rejection": "responder_error"},
        id="responder-error",
    ),
    pytest.param({"reply": CONTACT, "outcome": "handoff", "rule": "timeout"}, id="timeout"),
    pytest.param({"reply": CONTACT, "outcome": "handoff", "rule": "error"}, id="crash"),
    pytest.param({"reply": NO_INFORMATION, "outcome": "handoff"}, id="the-model-declined"),
    pytest.param(
        {"reply": CONTACT, "outcome": "handoff", "intent": "human"}, id="sent-to-a-person"
    ),
    pytest.param(
        {"reply": CONTACT, "outcome": "handoff", "rejection": "intent_not_allowed"},
        id="an-intent-the-business-lacks",
    ),
    pytest.param({"reply": "Per quale servizio?", "outcome": "asked"}, id="asked-back"),
    pytest.param(
        {
            "reply": f"{TEXTS['emergency']}\n\n{CONTACT}",
            "outcome": "handoff",
            "rule": "emergency",
            "category": "emergency",
        },
        id="taken-for-an-emergency",
    ),
    pytest.param(
        {"reply": "Come posso aiutarti?", "outcome": "fixed", "intent": "greeting"},
        id="taken-for-a-greeting",
    ),
]


@pytest.mark.parametrize("fields", LOST)
def test_a_question_nobody_answered_counts_against_the_criterion(fields: dict[str, Any]) -> None:
    rest = {name: value for name, value in fields.items() if name != "reply"}
    [verdict] = judge([PARKING], [record("parking", fields["reply"], **rest)], FIXED_TEXTS, {})
    assert criterion((verdict,)) == ("100% (0+1/1)", "fail")
    assert verdict.document_question
    assert verdict.unanswered_document_question
    assert not verdict.refused
    summary = summarize([verdict])["it"]
    assert (summary.document_questions, summary.unanswered_documents) == (1, 1)
    assert (summary.wrong_refusals, summary.refusals_pending) == (0, 0)
    assert summary.lost_answer_rate == 1.0


def test_an_answered_question_and_a_refused_one_are_not_unanswered() -> None:
    [answered] = judge(
        [PARKING], [record("parking", SIX, sources=("parking-en",))], FIXED_TEXTS, {}
    )
    refused = record("parking", NO_INFORMATION, outcome="handoff", **REFUSED)
    [handed_off] = judge([PARKING], [refused], FIXED_TEXTS, {})
    assert not answered.unanswered_document_question
    assert handed_off.refused
    assert not handed_off.unanswered_document_question


def test_a_question_answered_from_the_records_was_answered_though_wrongly() -> None:
    # The classifier sent it to the prices: the person got an answer, the case fails on its text.
    fact = record("parking", "Fisioterapia: 30 EUR.", sources=("price:physiotherapy",))
    [verdict] = judge([PARKING], [fact], FIXED_TEXTS, {})
    assert verdict.document_question
    assert not verdict.unanswered_document_question
    assert not verdict.passed


@pytest.mark.parametrize("behaviour", ["handoff", "clarify", "emergency", None])
def test_a_case_that_expects_no_answer_has_no_document_question_whatever_it_names(
    behaviour: str | None,
) -> None:
    named = case("named", behaviour=behaviour, sources=["parking-en"])
    [verdict] = judge([named], [record("named", CONTACT, outcome="handoff")], FIXED_TEXTS, {})
    assert not verdict.document_question
    assert not verdict.unanswered_document_question


def test_a_case_that_expects_no_document_answer_has_no_question_to_lose() -> None:
    wanted = case("handoff", behaviour="handoff")
    [verdict] = judge([wanted], [record("handoff", CONTACT, outcome="handoff")], FIXED_TEXTS, {})
    assert not verdict.document_question
    assert not verdict.unanswered_document_question


# --- the value of the criterion ------------------------------------------------------------------


def test_the_value_shows_the_wrong_refusals_and_the_lost_questions_apart() -> None:
    assert criterion(lost_run(answered=7, refused_wrong=1, lost=2)) == ("30% (1+2/10)", "fail")


def test_a_run_that_lost_nothing_keeps_the_briefs_value() -> None:
    assert criterion(lost_run(answered=4, refused_wrong=1)) == ("20% (1/5)", "pass")
    assert criterion(lost_run(answered=5)) == ("0% (0/5)", "pass")


def test_lost_questions_alone_show_a_zero_for_the_refusals() -> None:
    assert criterion(lost_run(answered=3, lost=2)) == ("40% (0+2/5)", "fail")


def test_exactly_the_limit_passes_and_one_question_more_fails() -> None:
    assert criterion(lost_run(answered=4, lost=1)) == ("20% (0+1/5)", "pass")
    assert criterion(lost_run(answered=3, lost=2))[1] == "fail"


def test_lost_questions_fail_the_criterion_while_refusals_wait_for_a_label() -> None:
    # 2 of 5 are lost, 40% whatever the reviewer says of the refused one: a failure stays a failure.
    assert criterion(lost_run(answered=2, refused_unlabelled=1, lost=2)) == (
        "40% (0+2/5)",
        "fail",
    )
    # 1 of 5 is lost, 20%: the refused one can still tip it, so it waits.
    assert criterion(lost_run(answered=3, refused_unlabelled=1, lost=1)) == (
        "20% (0+1/5)",
        "pending",
    )


def test_one_wrong_refusal_and_two_lost_questions_are_not_a_pass() -> None:
    # The review's A4a: shown as 20% (1/5) PASS, when the bot failed 3 of 5 (60%).
    assert criterion(lost_run(answered=2, refused_wrong=1, lost=2)) == ("60% (1+2/5)", "fail")


def test_questions_the_model_declined_to_answer_are_lost_too() -> None:
    # The review's A4b: 3 of 5 handed off by the model itself, no draft and no failure.
    cases = [
        case(f"d{n}", behaviour="answer", contains=["six"], sources=["parking-en"])
        for n in range(5)
    ]
    records = [
        *[record(f"d{n}", SIX, sources=("parking-en",)) for n in range(2)],
        *[record(f"d{n}", NO_INFORMATION, outcome="handoff") for n in range(2, 5)],
    ]
    assert criterion(judge(cases, records, FIXED_TEXTS, {})) == ("60% (0+3/5)", "fail")


def test_the_language_table_and_the_json_count_the_lost_questions() -> None:
    verdicts = lost_run(answered=2, refused_wrong=1, lost=2)
    text = render_report(manifest(), verdicts, total_cases=5, budget=None)
    assert "| Right answers refused (it) | at most 20% | 60% (1+2/5) | FAIL |" in text
    [row] = [line for line in text.splitlines() if line.startswith("| it ")]
    assert row.split(" | ")[5] == "1+2/5"
    document = json.loads(report_json(manifest(), verdicts, budget=None))
    italian = document["languages"]["it"]
    assert (italian["document_questions"], italian["wrong_refusals"]) == (5, 1)
    assert italian["unanswered_documents"] == 2
    assert italian["lost_answer_rate"] == pytest.approx(0.6)
    assert italian["wrong_refusal_rate"] == pytest.approx(0.2)
    flags = {
        c["case_id"]: (c["document_question"], c["unanswered_document_question"])
        for c in document["cases"]
    }
    assert flags["d0"] == (True, False)
    assert flags["d3"] == (True, True)


# --- a responder outage -------------------------------------------------------------------------


def test_a_responder_outage_with_nothing_answered_fails_in_every_language() -> None:
    verdicts = outage()
    found = {
        c.name: (c.value, c.status)
        for c in criteria(summarize(verdicts), spent_eur=None, cap_eur=None)
    }
    for language in ("it", "en", "sq"):
        assert found[f"Right answers refused ({language})"] == ("100% (0+5/5)", "fail")
    questions = [v for v in verdicts if v.document_question]
    assert len(questions) == 15
    assert not any(v.answered for v in questions)  # none of the 15 was answered


def test_the_outage_report_says_so_in_its_criteria_and_in_every_language_row() -> None:
    text = render_report(manifest(), outage(), total_cases=len(outage()), budget=None)
    for language in ("it", "en", "sq"):
        assert f"| Right answers refused ({language}) | at most 20% | 100% (0+5/5) | FAIL |" in text
        [row] = [line for line in text.splitlines() if line.startswith(f"| {language} ")]
        assert row.split(" | ")[5] == "0+5/5"
    [every] = [line for line in text.splitlines() if line.startswith("| all ")]
    assert every.split(" | ")[5] == "0+15/15"


# --- a refused case that names no document -------------------------------------------------------


def test_a_refused_answer_counts_whether_or_not_the_case_names_its_document() -> None:
    # The review's guard probe: one of two right answers refused, the reviewer says so, and the case
    # names no document in `sources`: the report said 0% (0/1) PASS.
    declared = case("declared", behaviour="answer", contains=["six"], sources=["parking-en"])
    undeclared = case("undeclared", behaviour="answer", contains=["six"])
    records = [
        record("declared", SIX, sources=("parking-en",)),
        record("undeclared", NO_INFORMATION, outcome="handoff", **REFUSED),
    ]
    verdicts = judge(
        [declared, undeclared], records, FIXED_TEXTS, {"undeclared": Label(refusal="wrong")}
    )
    assert criterion(verdicts) == ("50% (1/2)", "fail")
    refused = verdicts[1]
    assert refused.refused
    assert refused.document_question
    assert refused.wrong_refusal
    assert refused in review_cases(verdicts)


def test_a_refused_answer_the_reviewer_calls_right_is_in_the_denominator_only() -> None:
    undeclared = case("undeclared", behaviour="answer", contains=["six"])
    sent = record("undeclared", NO_INFORMATION, outcome="handoff", **REFUSED)
    verdicts = judge([undeclared], [sent], FIXED_TEXTS, {"undeclared": Label(refusal="right")})
    assert criterion(verdicts) == ("0% (0/1)", "pass")
    assert not verdicts[0].unanswered_document_question


# --- a clinical message with a document question --------------------------------------------------

MIXED = case(
    "mixed",
    behaviour="clinical",
    contains=["Non posso dare consigli medici"],
    sources=["parking-it"],
)


def mixed_run(reply: str, **fields: Any) -> CaseVerdict:
    labels = fields.pop("labels", {})
    [verdict] = judge(
        [MIXED],
        [record("mixed", reply, outcome="handoff", category="clinical", **fields)],
        FIXED_TEXTS,
        labels,
    )
    return verdict


def test_a_mixed_message_whose_answer_was_refused_is_a_refused_case() -> None:
    verdict = mixed_run(NOTICE, labels={"mixed": Label(refusal="wrong")}, **REFUSED)
    assert criterion((verdict,)) == ("100% (1/1)", "fail")
    assert verdict.observed is Behaviour.CLINICAL
    assert verdict.refused
    assert verdict.document_question
    assert verdict.wrong_refusal
    assert not verdict.unanswered_document_question
    assert review_cases([verdict]) == (verdict,)


def test_a_refused_mixed_case_waits_for_the_reviewer_like_any_refusal() -> None:
    verdict = mixed_run(NOTICE, **REFUSED)
    assert verdict.refusal_pending
    assert criterion((verdict,)) == ("0% (0/1)", "pending")


def test_a_mixed_message_answered_ahead_of_the_notice_is_an_answered_question() -> None:
    reply = f"Ci sono sei posti.\n\n{NOTICE}"
    verdict = mixed_run(reply, sources=("parking-it",))
    assert verdict.document_question
    assert verdict.answered
    assert not verdict.unanswered_document_question
    assert criterion((verdict,)) == ("0% (0/1)", "pass")


def test_a_mixed_message_whose_answer_never_came_is_lost() -> None:
    declined = mixed_run(NOTICE)
    assert criterion((declined,)) == ("100% (0+1/1)", "fail")
    assert declined.document_question
    assert declined.unanswered_document_question
    assert not declined.refused
    failed = mixed_run(NOTICE, rejection="responder_error")
    assert failed.unanswered_document_question
    assert failed.failed_by_pipeline


@pytest.mark.parametrize(
    "named",
    [
        pytest.param([], id="nothing"),
        pytest.param(["price:physiotherapy"], id="a-record"),
    ],
)
def test_a_clinical_case_that_names_no_document_has_no_document_question(named: list[str]) -> None:
    plain = case(
        "plain", behaviour="clinical", contains=["Non posso dare consigli medici"], sources=named
    )
    kept = record("plain", NOTICE, outcome="handoff", category="clinical", **REFUSED)
    [verdict] = judge([plain], [kept], FIXED_TEXTS, {})
    assert not verdict.document_question
    assert not verdict.refused
    assert not verdict.unanswered_document_question
