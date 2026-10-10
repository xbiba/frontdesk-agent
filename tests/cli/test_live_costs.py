"""What the live chat prints and counts: the banner, the cost line, the loop (M3 Task 6b).

No database, no model and no network: the meter is fed by a fake model, and the loop by a stand-in
for the turn.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from frontdesk.cli.chat import run_chat
from frontdesk.cli.live_chat import LiveCosts, chat_run_id, format_cost_line, live_banner
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, TurnTrace
from frontdesk.model_gateway.metering import Meter
from frontdesk.model_gateway.routes import ROUTES
from tests.cli.live_chat_helpers import BANNER
from tests.model_gateway.budget_helpers import ROUTE, write_budget

# --- The lines the chat prints ----------------------------------------------------------------------


def test_the_banner_names_the_model_the_budget_what_is_left_and_the_rule(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)
    banner = live_banner(ROUTES[ROUTE], budget, spent_usd=2.5)  # 2.50 USD are 2.00 EUR
    assert banner == BANNER.format(path=budget.path, left="8.00", cap="10.00")


def test_the_banner_never_shows_a_negative_amount_left(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=1.0, usd_per_eur=1.0)
    banner = live_banner(ROUTES[ROUTE], budget, spent_usd=1.5)
    assert "(0.00 EUR left of 1.00 EUR)" in banner
    assert "-0.00" not in banner


def test_the_cost_line_has_four_decimals_for_spend_and_two_for_what_is_left() -> None:
    line = format_cost_line(turn_eur=0.00126, session_eur=0.5, left_eur=9.499)
    assert line == "cost: 0.0013 EUR \N{MIDDLE DOT} session 0.5000 EUR \N{MIDDLE DOT} left 9.50 EUR"


def test_a_session_is_filed_under_the_utc_time_it_began() -> None:
    assert chat_run_id(datetime(2026, 10, 12, 9, 5, 7, tzinfo=UTC)) == "chat-20261012T090507Z"
    plus_two = timezone(timedelta(hours=2))
    assert chat_run_id(datetime(2026, 10, 12, 11, 5, 7, tzinfo=plus_two)) == "chat-20261012T090507Z"


# --- What a session has spent -----------------------------------------------------------------------


def _answering(input_tokens: int, output_tokens: int) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        return ModelResponse(parts=[TextPart("ok")], usage=usage, finish_reason="stop")

    return FunctionModel(model)


async def _request(meter: Meter, input_tokens: int, output_tokens: int) -> None:
    agent = Agent(meter.wrap(_answering(input_tokens, output_tokens), "classifier"))
    await agent.run("x", model_settings={"max_tokens": 600})


async def test_the_cost_line_counts_the_turn_from_the_meter_and_the_rest_from_the_ledger(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=1.25)  # 1.00 EUR before
    meter = Meter(budget, route_id=ROUTE, run_id="chat-x")
    costs = LiveCosts.begin(meter, budget)
    await _request(meter, 1_000_000, 100_000)  # 1.00 + 1.00 USD, 1.60 EUR
    assert costs.cost_line() == (
        "cost: 1.6000 EUR \N{MIDDLE DOT} session 1.6000 EUR \N{MIDDLE DOT} left 7.40 EUR"
    )
    await _request(meter, 500_000, 0)  # 0.50 USD, 0.40 EUR
    assert costs.cost_line() == (
        "cost: 0.4000 EUR \N{MIDDLE DOT} session 2.0000 EUR \N{MIDDLE DOT} left 7.00 EUR"
    )
    assert costs.stop_reason is None


def test_a_turn_that_cost_nothing_says_so(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    costs = LiveCosts.begin(Meter(budget, route_id=ROUTE, run_id="chat-x"), budget)
    assert costs.cost_line() == (
        "cost: 0.0000 EUR \N{MIDDLE DOT} session 0.0000 EUR \N{MIDDLE DOT} left 10.00 EUR"
    )


def test_past_the_cap_what_is_left_is_zero_not_negative(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=1.0, usd_per_eur=1.0)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=1.5)
    costs = LiveCosts.begin(Meter(budget, route_id=ROUTE, run_id="chat-x"), budget)
    assert costs.cost_line().endswith("left 0.00 EUR")


def test_the_session_stops_when_its_meter_does(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    meter = Meter(budget, route_id=ROUTE, run_id="chat-x")
    costs = LiveCosts.begin(meter, budget)
    assert costs.stop_reason is None
    meter.rate_limited()
    assert costs.stop_reason == "rate_limited"


# --- The loop ---------------------------------------------------------------------------------------


def _result(text: str) -> TurnResult:
    return TurnResult(text, TurnOutcome.REPLIED, "it", TurnTrace(rule="pass"))


class StubCosts:
    """A metered session as the loop sees it; a test sets `stop_reason` while a turn runs."""

    def __init__(self) -> None:
        self.stop_reason: str | None = None
        self.lines: list[str] = []

    def cost_line(self) -> str:
        self.lines.append(f"cost after reply {len(self.lines) + 1}")
        return self.lines[-1]


def _patch_turns(
    monkeypatch: pytest.MonkeyPatch, costs: StubCosts, *, stop_on: str | None, reason: str
) -> None:
    async def fake_handle(
        text: str, state: ConversationState, runtime: TurnRuntime
    ) -> tuple[TurnResult, ConversationState]:
        if text == stop_on:
            costs.stop_reason = reason  # the meter stops while the turn runs
        return _result(f"echo {text}"), state

    monkeypatch.setattr("frontdesk.cli.chat.handle_message", fake_handle)


async def test_a_cost_line_follows_each_reply_and_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    costs = StubCosts()
    _patch_turns(monkeypatch, costs, stop_on=None, reason="")
    written: list[str] = []
    lines = iter(["ciao", "", "Quanto costa?", "exit", "never read"])
    stopped = await run_chat(
        cast(TurnRuntime, None),
        read_line=lambda: next(lines, None),
        write=written.append,
        show_trace=True,
        costs=costs,
    )
    assert stopped is None
    assert written == [
        "bot> echo ciao",
        "     [rule=pass category=- intent=- sources=- outcome=replied]",
        "cost after reply 1",
        "bot> echo Quanto costa?",
        "     [rule=pass category=- intent=- sources=- outcome=replied]",
        "cost after reply 2",
    ]
    assert next(lines) == "never read"


@pytest.mark.parametrize("reason", ["budget", "ledger", "rate_limited"])
async def test_a_stopped_meter_ends_the_loop_without_the_reply_of_that_turn(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    costs = StubCosts()
    _patch_turns(monkeypatch, costs, stop_on="second", reason=reason)
    written: list[str] = []
    lines = iter(["first", "second", "third"])
    stopped = await run_chat(
        cast(TurnRuntime, None),
        read_line=lambda: next(lines, None),
        write=written.append,
        show_trace=False,
        costs=costs,
    )
    assert stopped == reason
    # The turn the stop cut short printed no reply and no cost line, and "third" was never read.
    assert written == ["bot> echo first", "cost after reply 1"]
    assert next(lines) == "third"


async def test_a_loop_without_costs_prints_no_cost_line_and_never_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_turns(monkeypatch, StubCosts(), stop_on=None, reason="")
    written: list[str] = []
    lines = iter(["ciao"])
    stopped = await run_chat(
        cast(TurnRuntime, None),
        read_line=lambda: next(lines, None),
        write=written.append,
        show_trace=False,
    )
    assert stopped is None
    assert written == ["bot> echo ciao"]
