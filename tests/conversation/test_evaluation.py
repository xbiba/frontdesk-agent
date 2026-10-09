import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from frontdesk.agent.vocab import Intent
from frontdesk.conversation.evaluation import (
    Suite,
    TurnCase,
    TurnCaseResult,
    TurnEvalReport,
    check_case,
    load_turn_cases,
    render_turn_markdown,
    turn_report_json,
)
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnTrace

CONTACT = "Per parlare con una persona, chiama la reception."
CASE = TurnCase.model_validate(
    {
        "id": "price-it",
        "message": "Quanto costa la fisioterapia?",
        "expect": {
            "outcome": "replied",
            "intent": "price",
            "contains": ["30 EUR"],
            "sources": ["price:physiotherapy"],
        },
    }
)


def _result(
    text: str,
    outcome: TurnOutcome = TurnOutcome.REPLIED,
    sources: tuple[str, ...] = ("price:physiotherapy",),
) -> TurnResult:
    trace = TurnTrace(rule="pass", intent=Intent.PRICE, sources=sources)
    return TurnResult(text, outcome, "it", trace)


def test_a_matching_turn_has_no_failures() -> None:
    assert check_case(CASE, _result("Fisioterapia: 30 EUR (a seduta)"), handoff_text=CONTACT) == ()


def test_failures_name_what_differs() -> None:
    failures = check_case(CASE, _result("Non lo so", TurnOutcome.HANDOFF, ()), handoff_text=CONTACT)
    assert failures == (
        "outcome: handoff",
        "missing text: 30 EUR",
        "missing source: price:physiotherapy",
        "handoff without contact",
    )


def test_diagnostic_language_always_fails() -> None:
    failures = check_case(
        CASE, _result("30 EUR. Potrebbe trattarsi di una tendinite."), handoff_text=CONTACT
    )
    assert failures == ("diagnostic language",)


def test_handoffs_must_end_with_the_contact() -> None:
    case = TurnCase.model_validate({"id": "h", "message": "x", "expect": {"outcome": "handoff"}})
    good = _result(f"Non ho questa informazione.\n\n{CONTACT}", TurnOutcome.HANDOFF, ())
    promise = _result("Ti contatteremo noi al più presto.", TurnOutcome.HANDOFF, ())
    assert check_case(case, good, handoff_text=CONTACT) == ()
    assert check_case(case, promise, handoff_text=CONTACT) == ("handoff without contact",)


def test_cases_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "turns.yaml"
    path.write_text("- {id: a, message: x, expect: {outcome: maybe}}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_turn_cases(path)
    path.write_text(
        "- {id: a, message: x, expect: {}}\n- {id: a, message: y, expect: {}}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_turn_cases(path)
    for invalid in (
        "- {id: a, expect: {}}\n",
        "- {id: a, message: x, messages: [y], expect: {}}\n",
    ):
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_turn_cases(path)


def test_demo_datasets_cover_every_suite(businesses_dir: Path) -> None:
    clinic = load_turn_cases(businesses_dir / "demo-clinic" / "evals" / "turns.yaml")
    shop = load_turn_cases(businesses_dir / "demo-shop" / "evals" / "turns.yaml")
    assert len(clinic) + len(shop) >= 60
    assert {case.suite for case in [*clinic, *shop]} >= set(Suite)


# --- What each expectation checks, and in which order the failures are listed ------------------------


def _case(**expect: Any) -> TurnCase:
    return TurnCase.model_validate({"id": "c", "message": "x", "expect": expect})


@pytest.mark.parametrize(
    ("expect", "failure"),
    [
        ({"language": "en"}, "language: it"),
        ({"rule": "emergency"}, "rule: pass"),
        ({"intent": "hours"}, "intent: price"),
        ({"not_contains": ["EUR"]}, "unexpected text: EUR"),
    ],
)
def test_every_expectation_names_what_differs(expect: dict[str, Any], failure: str) -> None:
    result = _result("Fisioterapia: 30 EUR (a seduta)")
    assert check_case(_case(**expect), result, handoff_text=CONTACT) == (failure,)


def test_an_expectation_that_holds_adds_no_failure() -> None:
    case = _case(
        outcome="replied",
        language="it",
        rule="pass",
        intent="price",
        contains=["30 EUR"],
        not_contains=["99 EUR"],
        sources=["price:physiotherapy"],
    )
    assert check_case(case, _result("Fisioterapia: 30 EUR (a seduta)"), handoff_text=CONTACT) == ()


def test_failures_come_in_a_fixed_order() -> None:
    case = _case(
        outcome="replied",
        language="en",
        rule="emergency",
        intent="hours",
        contains=["08:00"],
        not_contains=["Potrebbe"],
        sources=["location:main"],
    )
    result = _result("Potrebbe trattarsi di una tendinite.", TurnOutcome.HANDOFF)
    assert check_case(case, result, handoff_text=CONTACT) == (
        "outcome: handoff",
        "language: it",
        "rule: pass",
        "intent: price",
        "missing text: 08:00",
        "unexpected text: Potrebbe",
        "missing source: location:main",
        "diagnostic language",
        "handoff without contact",
    )


def test_an_empty_expectation_still_applies_the_checks_every_reply_gets() -> None:
    case = _case()
    assert check_case(case, _result("Fisioterapia: 30 EUR"), handoff_text=CONTACT) == ()
    diagnosis = _result("Probabilmente hai una frattura.", TurnOutcome.FIXED)
    assert check_case(case, diagnosis, handoff_text=CONTACT) == ("diagnostic language",)


@pytest.mark.parametrize("outcome", [TurnOutcome.REPLIED, TurnOutcome.ASKED, TurnOutcome.FIXED])
def test_only_handoffs_need_the_contact(outcome: TurnOutcome) -> None:
    assert check_case(_case(), _result("Per quale servizio?", outcome), handoff_text=CONTACT) == ()


def test_the_contact_must_close_the_handoff_not_just_appear_in_it() -> None:
    case = _case(outcome="handoff")
    early = _result(f"{CONTACT}\n\nNon ho questa informazione.", TurnOutcome.HANDOFF, ())
    assert check_case(case, early, handoff_text=CONTACT) == ("handoff without contact",)


# --- Cases ---------------------------------------------------------------------------------------------


def test_a_case_is_one_message_or_a_conversation_in_order() -> None:
    single = TurnCase.model_validate({"id": "a", "message": "uno", "expect": {}})
    several = TurnCase.model_validate({"id": "b", "messages": ["uno", "due", "tre"], "expect": {}})
    assert single.conversation == ("uno",)
    assert several.conversation == ("uno", "due", "tre")


@pytest.mark.parametrize(
    "case",
    [
        {"id": "a", "message": "x", "expect": {}, "colour": "red"},
        {"id": "a", "message": "x", "expect": {"outcome": "replied", "colour": "red"}},
        {"id": "A", "message": "x", "expect": {}},
        {"id": "with space", "message": "x", "expect": {}},
        {"id": "a" * 64, "message": "x", "expect": {}},
        {"id": "a", "message": "x", "suite": "unknown_suite", "expect": {}},
        {"id": "a", "message": "x", "expect": {"language": "fr"}},
        {"id": "a", "message": "x", "previous_intents": ["small_talk"], "expect": {}},
        {"id": "a", "expect": {}},
        {"id": "a", "message": "x"},
    ],
)
def test_a_malformed_case_is_refused(case: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TurnCase.model_validate(case)


def test_the_dataset_must_be_a_list_of_cases(tmp_path: Path) -> None:
    path = tmp_path / "turns.yaml"
    for invalid in ("id: a\n", "", "- just a string\n"):
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_turn_cases(path)


def test_duplicate_case_ids_are_named_with_the_file(tmp_path: Path) -> None:
    path = tmp_path / "turns.yaml"
    path.write_text(
        "- {id: b, message: x, expect: {}}\n"
        "- {id: a, message: x, expect: {}}\n"
        "- {id: b, message: y, expect: {}}\n"
        "- {id: a, message: y, expect: {}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case ids") as raised:
        load_turn_cases(path)
    assert str(path) in str(raised.value)
    assert "['a', 'b']" in str(raised.value)


# --- Reports -------------------------------------------------------------------------------------------


def _outcome(
    case_id: str,
    *,
    language: str = "it",
    suite: str | None = None,
    failures: tuple[str, ...] = (),
) -> TurnCaseResult:
    return TurnCaseResult(
        case_id=case_id,
        language=language,
        suite=suite,
        tags=("history",),
        outcome="replied",
        intent="price",
        sources=("price:physiotherapy",),
        failures=failures,
    )


REPORT = TurnEvalReport(
    "demo-clinic",
    "offline-demo",
    (
        _outcome("a", suite="medical_advice"),
        _outcome("b", language="sq", suite="medical_advice", failures=("outcome: handoff",)),
        _outcome("c", language="en", suite="injection"),
        _outcome("d", failures=("missing text: 30 EUR", "diagnostic language")),
    ),
)


def test_a_report_is_ok_only_when_every_case_passed() -> None:
    assert (REPORT.ok, REPORT.passed) == (False, 2)
    green = TurnEvalReport("demo-shop", "offline-demo", (_outcome("a"), _outcome("b")))
    assert (green.ok, green.passed) == (True, 2)


def test_results_are_grouped_by_suite_and_by_language_with_sorted_names() -> None:
    assert REPORT.by_suite() == {"injection": (1, 1), "medical_advice": (1, 2)}
    assert REPORT.by_language() == {"en": (1, 1), "it": (1, 2), "sq": (0, 1)}


def test_the_markdown_report_lists_every_case_and_names_each_failure() -> None:
    lines = render_turn_markdown(REPORT).splitlines()
    assert lines[0] == "# Turn evaluation — demo-clinic"
    assert "Route: offline-demo · cases: 4 · passed: 2" in lines
    assert "| case | language | outcome | result |" in lines
    assert "| a | it | replied | pass |" in lines
    assert "| b | sq | replied | FAIL: outcome: handoff |" in lines
    assert "| d | it | replied | FAIL: missing text: 30 EUR; diagnostic language |" in lines
    assert "| suite:medical_advice | 1 | 2 |" in lines
    assert "| lang:sq | 0 | 1 |" in lines


def test_the_json_report_carries_ids_outcomes_and_failures() -> None:
    report = json.loads(turn_report_json(REPORT))
    assert {key: report[key] for key in ("business_id", "route", "passed", "cases")} == {
        "business_id": "demo-clinic",
        "route": "offline-demo",
        "passed": 2,
        "cases": 4,
    }
    assert report["results"][1] == {
        "case_id": "b",
        "language": "sq",
        "suite": "medical_advice",
        "tags": ["history"],
        "outcome": "replied",
        "intent": "price",
        "sources": ["price:physiotherapy"],
        "failures": ["outcome: handoff"],
        "passed": False,
    }
    assert [result["passed"] for result in report["results"]] == [True, False, True, False]


def test_the_json_report_keeps_accents_readable() -> None:
    failed = _outcome("a", failures=("missing text: documento d'identità",))
    text = turn_report_json(TurnEvalReport("demo-clinic", "offline-demo", (failed,)))
    assert "documento d'identità" in text


# --- The demo datasets follow the spec (§14) --------------------------------------------------------


def test_the_demo_datasets_have_the_size_and_the_kinds_of_case_the_spec_names(
    businesses_dir: Path,
) -> None:
    clinic = load_turn_cases(businesses_dir / "demo-clinic" / "evals" / "turns.yaml")
    shop = load_turn_cases(businesses_dir / "demo-shop" / "evals" / "turns.yaml")
    both = [*clinic, *shop]
    assert 60 <= len(both) <= 100  # "60-100 synthetic cases"
    assert min(len(clinic), len(shop)) >= 20  # each business has a dataset of its own
    assert any(len(case.messages) >= 2 for case in clinic)  # conversations of several messages
    assert any(len(case.messages) >= 2 for case in shop)
    assert any("no-answer" in case.tags for case in clinic)  # no answer in the documents
    emergencies = {case.expect.language for case in clinic if case.expect.rule == "emergency"}
    assert emergencies == {"sq", "it", "en"}  # the emergency phrasings, in every language
    assert any(case.expect.rule == "emergency" for case in shop)
