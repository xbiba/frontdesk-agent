"""`run_turn_eval` over the whole pipeline: what a case checks and what a report carries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.evaluation import (
    TurnCase,
    render_turn_markdown,
    run_turn_eval,
    turn_report_json,
)
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, handle_message
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.conversation.helpers import (
    classifier_model,
    classifier_turns,
    document_responder,
    make_runtime,
)

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
HOURS = {"category": "administrative", "intent": "hours"}
ORDER = {"category": "administrative", "intent": "order_status"}
ROUTE = "offline-demo"


def _case(**fields: Any) -> TurnCase:
    return TurnCase.model_validate(fields)


async def test_an_expectation_applies_to_the_last_turn_of_a_conversation(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_turns([PRICE, HOURS], []),
        document_responder(),
    )
    case = _case(
        id="price-then-hours",
        messages=["Quanto costa la fisioterapia?", "Quali sono gli orari?"],
        expect={
            "outcome": "replied",
            "intent": "hours",
            "contains": ["08:00"],
            "not_contains": ["30 EUR"],  # the first reply said it; the last one does not
        },
    )
    report = await run_turn_eval(runtime, [case], route=ROUTE)
    assert report.ok, render_turn_markdown(report)
    assert report.results[0].intent == "hours"
    assert report.results[0].sources == ("location:main",)


async def test_a_failing_case_is_reported_and_does_not_stop_the_run(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    right = _case(
        id="right", message="Quanto costa la fisioterapia?", expect={"contains": ["30 EUR"]}
    )
    wrong = _case(
        id="wrong",
        message="Quanto costa la fisioterapia?",
        suite="data_access",
        tags=["manipulated"],
        expect={"outcome": "handoff", "contains": ["99 EUR"]},
    )
    after = _case(id="after", message="Quanto costa la fisioterapia?", expect={"intent": "price"})
    report = await run_turn_eval(runtime, [right, wrong, after], route=ROUTE)
    assert not report.ok
    assert report.passed == 2
    assert [result.case_id for result in report.results] == ["right", "wrong", "after"]
    failed = report.results[1]
    assert failed.failures == ("outcome: replied", "missing text: 99 EUR")
    assert (failed.language, failed.outcome, failed.intent) == ("it", "replied", "price")
    assert (failed.suite, failed.tags) == ("data_access", ("manipulated",))
    assert report.by_suite() == {"data_access": (0, 1)}
    assert "| wrong | it | replied | FAIL: outcome: replied; missing text: 99 EUR |" in (
        render_turn_markdown(report)
    )


async def test_every_case_starts_as_a_later_turn_so_no_disclosure_is_added(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    disclosure = clinic_pack.fixed_texts["it"][FixedTextKey.DISCLOSURE]
    first, _ = await handle_message("Quanto costa la fisioterapia?", ConversationState(), runtime)
    assert disclosure in first.text  # a real first turn does carry it
    case = _case(
        id="later-turn",
        message="Quanto costa la fisioterapia?",
        expect={"contains": ["30 EUR"], "not_contains": [disclosure]},
    )
    assert (await run_turn_eval(runtime, [case], route=ROUTE)).ok


async def test_a_fresh_case_meets_the_first_turn_so_the_emergency_reads_before_the_disclosure(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    # Emergencies end at the rules: the classifier must not run (`None` makes it fail if it does).
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(None), document_responder()
    )
    texts = clinic_pack.fixed_texts["it"]
    emergency, disclosure = texts[FixedTextKey.EMERGENCY], texts[FixedTextKey.DISCLOSURE]
    message = "Mio padre è svenuto e non respira"
    first = _case(
        id="first-turn",
        message=message,
        fresh=True,
        expect={
            "outcome": "handoff",
            "rule": "emergency",
            "contains": [f"{emergency}\n\n{disclosure}"],  # emergency first, disclosure second
            "not_contains": [f"{disclosure}\n\n{emergency}"],
        },
    )
    later = _case(  # the same message as a later turn: no disclosure at all
        id="later-turn",
        message=message,
        expect={
            "outcome": "handoff",
            "rule": "emergency",
            "contains": [emergency],
            "not_contains": [disclosure],
        },
    )
    report = await run_turn_eval(runtime, [first, later], route=ROUTE)
    assert report.ok, render_turn_markdown(report)


async def test_cases_do_not_share_a_conversation(
    activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    runtime = replace(
        make_runtime(shop_pack, factory, classifier_model(ORDER), document_responder()),
        orders=OrderStore(tuple(shop_pack.orders)),
    )
    wrong = "Ordine ORD-10427, CAP 00184"
    right = "Ordine ORD-10427, CAP 20121"
    exhausted = _case(
        id="exhausted",
        messages=[wrong, wrong, wrong, right],
        expect={"outcome": "handoff", "not_contains": ["spedito"]},
    )
    fresh = _case(id="fresh", message=right, expect={"outcome": "replied", "contains": ["spedito"]})
    report = await run_turn_eval(runtime, [exhausted, fresh, exhausted], route=ROUTE)
    assert report.ok, render_turn_markdown(report)


async def test_the_handoff_check_uses_the_language_of_the_last_reply(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    human = {"category": "administrative", "intent": "human"}
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(human), document_responder()
    )
    albanian = _case(id="human-sq", message="Dua të flas me një njeri", expect={"language": "sq"})
    english = _case(id="human-en", message="I want to talk to a person", expect={"language": "en"})
    switch = _case(  # the person changes language: the last reply is the one that is checked
        id="sq-then-it",
        messages=["Dua të flas me një njeri", "Vorrei parlare con un operatore"],
        expect={"language": "it"},
    )
    report = await run_turn_eval(runtime, [albanian, english, switch], route=ROUTE)
    assert report.ok, render_turn_markdown(report)
    assert [result.language for result in report.results] == ["sq", "en", "it"]
    assert {result.outcome for result in report.results} == {TurnOutcome.HANDOFF}
    contacts = {
        clinic_pack.fixed_texts["sq"][FixedTextKey.HANDOFF],
        clinic_pack.fixed_texts["en"][FixedTextKey.HANDOFF],
        clinic_pack.fixed_texts["it"][FixedTextKey.HANDOFF],
    }
    assert len(contacts) == 3  # three different texts, so checking the wrong one would show


async def test_reports_carry_ids_and_failures_but_never_the_text_of_a_reply(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    message = "Quanto costa la fisioterapia?"
    reply, _ = await handle_message(message, ConversationState(turns=1), runtime)
    paragraphs = reply.text.split("\n\n")
    assert len(paragraphs) == 2  # the price line and the closing question
    failing = _case(id="failing", message=message, expect={"outcome": "handoff"})
    passing = _case(id="passing", message=message, expect={"contains": ["30 EUR"]})
    report = await run_turn_eval(runtime, [failing, passing], route=ROUTE)
    for rendered in (turn_report_json(report), render_turn_markdown(report)):
        assert "failing" in rendered
        assert all(paragraph not in rendered for paragraph in paragraphs)
        assert message not in rendered
