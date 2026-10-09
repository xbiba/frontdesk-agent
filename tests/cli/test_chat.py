"""The terminal chat: the trace line it prints and the read-reply loop (no database needed)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from frontdesk.agent.vocab import Category, Intent
from frontdesk.cli.chat import format_trace, run_chat
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, TurnTrace


def _result(
    text: str = "reply", outcome: TurnOutcome = TurnOutcome.REPLIED, **trace: Any
) -> TurnResult:
    return TurnResult(text, outcome, "it", TurnTrace(**{"rule": "pass", **trace}))


def test_the_trace_line_lists_what_the_pipeline_decided() -> None:
    line = format_trace(
        _result(
            category=Category.ADMINISTRATIVE,
            intent=Intent.PRICE,
            sources=("price:physiotherapy",),
        )
    )
    assert line == (
        "[rule=pass category=administrative intent=price "
        "sources=price:physiotherapy outcome=replied]"
    )


def test_a_trace_without_decisions_shows_dashes() -> None:
    line = format_trace(_result(outcome=TurnOutcome.FIXED, rule="empty"))
    assert line == "[rule=empty category=- intent=- sources=- outcome=fixed]"


def test_a_rejection_and_the_failed_checks_are_appended() -> None:
    line = format_trace(
        _result(
            outcome=TurnOutcome.HANDOFF,
            rejection="intent_not_allowed",
            check_reasons=("quote_not_in_source", "number_not_in_quote"),
        )
    )
    assert line.endswith(
        "outcome=handoff rejection=intent_not_allowed "
        "checks=quote_not_in_source,number_not_in_quote]"
    )


def test_a_live_order_source_is_shown_as_order_without_its_reference() -> None:
    line = format_trace(
        _result(intent=Intent.ORDER_STATUS, live_evidence=1, sources=("order:ORD-10427",))
    )
    assert "sources=order " in line
    assert "ORD-10427" not in line
    assert "10427" not in line


def test_only_order_sources_are_masked() -> None:
    line = format_trace(
        _result(sources=("price:physiotherapy", "order:ORD-10458", "parking-en", "location:main"))
    )
    assert "sources=price:physiotherapy,order,parking-en,location:main " in line
    assert "ORD-10458" not in line


@pytest.mark.parametrize("farewell", ["exit", "quit", "EXIT", "  Quit  "])
async def test_the_loop_answers_each_line_until_the_person_leaves(
    monkeypatch: pytest.MonkeyPatch, farewell: str
) -> None:
    seen: list[tuple[str, int]] = []

    async def fake_handle(
        text: str, state: ConversationState, runtime: TurnRuntime
    ) -> tuple[TurnResult, ConversationState]:
        seen.append((text, state.turns))
        return _result(f"echo {text}"), state.model_copy(update={"turns": state.turns + 1})

    monkeypatch.setattr("frontdesk.cli.chat.handle_message", fake_handle)
    lines = iter(["ciao", "", "   ", "  Quanto costa?  ", farewell, "never read"])
    written: list[str] = []
    await run_chat(
        cast(TurnRuntime, None),
        read_line=lambda: next(lines, None),
        write=written.append,
        show_trace=False,
    )
    # Blank lines are skipped, the text is trimmed, the state travels from turn to turn.
    assert seen == [("ciao", 0), ("Quanto costa?", 1)]
    assert written == ["bot> echo ciao", "bot> echo Quanto costa?"]
    assert next(lines) == "never read"


async def test_the_loop_ends_with_the_input_and_shows_the_trace_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result("echo ciao", intent=Intent.GREETING, sources=("order:ORD-10427",))

    async def fake_handle(
        text: str, state: ConversationState, runtime: TurnRuntime
    ) -> tuple[TurnResult, ConversationState]:
        return result, state

    monkeypatch.setattr("frontdesk.cli.chat.handle_message", fake_handle)
    lines = iter(["ciao"])
    written: list[str] = []
    await run_chat(
        cast(TurnRuntime, None),
        read_line=lambda: next(lines, None),
        write=written.append,
        show_trace=True,
    )
    assert written == ["bot> echo ciao", f"     {format_trace(result)}"]
    assert "ORD-10427" not in written[1]
