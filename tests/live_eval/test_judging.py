"""Judging recorded turns against the expected answers, and the reviewer's labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontdesk.conversation.behaviour import Behaviour
from frontdesk.live_eval.judging import Label, LabelsError, judge, load_labels
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, draft, record

PARKING = case(
    "parking", "Is there parking?", behaviour="answer", contains=["six"], sources=["parking-en"]
)
REFUSED = record(
    "parking",
    f"{TEXTS['no_information']}\n\n{CONTACT}",
    outcome="handoff",
    check_reasons=("number_not_in_quote",),
    draft=draft("There are six free spaces.", "six free parking spaces"),
)


def test_a_check_that_stopped_the_model_s_answer_makes_a_refused_case() -> None:
    [verdict] = judge([PARKING], [REFUSED], FIXED_TEXTS, {})
    assert verdict.refused
    assert verdict.observed is Behaviour.HANDOFF
    assert verdict.refusal_pending
    assert not verdict.wrong_refusal  # not until a reviewer says so


def test_the_reviewer_decides_whether_the_refusal_was_wrong() -> None:
    labels = {"parking": Label(refusal="wrong", note="the quote was there")}
    [verdict] = judge([PARKING], [REFUSED], FIXED_TEXTS, labels)
    assert verdict.wrong_refusal
    assert not verdict.refusal_pending


def test_a_handoff_without_a_draft_is_no_refusal() -> None:
    plain = REFUSED.model_copy(update={"draft": None, "check_reasons": ()})
    [verdict] = judge([PARKING], [plain], FIXED_TEXTS, {})
    assert not verdict.refused


def test_an_answer_with_a_number_the_expected_answer_lacks_is_dubious() -> None:
    reply = "There are six free spaces, open 24 hours."
    answered = record("parking", reply, language="en", sources=("parking-en",))
    [verdict] = judge([PARKING], [answered], FIXED_TEXTS, {})
    assert verdict.passed
    assert verdict.dubious
    assert verdict.answer_pending


def test_an_answer_that_says_only_what_is_expected_is_not_dubious() -> None:
    answered = record("parking", "There are six free spaces.", sources=("parking-en",))
    [verdict] = judge([PARKING], [answered], FIXED_TEXTS, {})
    assert verdict.passed
    assert not verdict.dubious


def test_a_reply_that_contradicts_the_expected_answer_is_a_wrong_answer() -> None:
    [verdict] = judge([PARKING], [record("parking", "There is no parking.")], FIXED_TEXTS, {})
    assert verdict.failures[0] == "missing text: six"
    assert verdict.wrong_answer


def test_a_wrong_answer_the_reviewer_finds_counts_and_may_be_an_invented_fact() -> None:
    labels = {"parking": Label(answer="wrong", invented_fact=True)}
    reply = "There are six free spaces, open 24 hours."
    [verdict] = judge([PARKING], [record("parking", reply)], FIXED_TEXTS, labels)
    assert verdict.wrong_answer
    assert verdict.invented_fact


def test_an_emergency_answered_with_the_clinical_notice_is_missed() -> None:
    emergency = case("chest", "Mi fa male il petto", behaviour="emergency")
    notice = record("chest", f"{TEXTS['clinical']}\n\n{CONTACT}", outcome="handoff", category=None)
    [verdict] = judge([emergency], [notice], FIXED_TEXTS, {})
    assert verdict.missed_emergency
    assert verdict.critical


def test_only_cases_with_every_turn_recorded_are_judged() -> None:
    conversation = case("two-turns", behaviour="answer").model_copy(
        update={"message": None, "messages": ("uno", "due")}
    )
    first = record("two-turns", "ok", turn=1, turns=2)
    assert judge([conversation], [first], FIXED_TEXTS, {}) == ()


def test_labels_are_read_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text('parking:\n  refusal: wrong\n  note: "fine"\n', encoding="utf-8")
    assert load_labels(path, [PARKING]) == {"parking": Label(refusal="wrong", note="fine")}
    assert load_labels(tmp_path / "missing.yaml", [PARKING]) == {}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("nobody:\n  answer: right\n", "'nobody' is not a case of this run"),
        ("parking:\n  answer: maybe\n", "case 'parking'"),
        ("parking:\n  invented_fact: true\n", "invented_fact needs answer: wrong"),
        ("- parking\n", "expected a mapping"),
    ],
)
def test_a_bad_label_is_refused_naming_the_case(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LabelsError, match=message):
        load_labels(path, [PARKING])


def test_a_reviewer_s_note_keeps_its_quotes_colons_and_line_breaks(tmp_path: Path) -> None:
    # The review page writes each note as a JSON string, which YAML reads as a quoted scalar.
    path = tmp_path / "labels.yaml"
    path.write_text(
        'parking:\n  refusal: wrong\n  note: "says: \\"six\\"\\nsecond line"\n', encoding="utf-8"
    )
    assert load_labels(path, [PARKING])["parking"].note == 'says: "six"\nsecond line'
