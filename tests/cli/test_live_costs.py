"""What the live chat prints and counts: the banner, the cost line, the loop (M3 Task 6b).

No database, no model and no network: the meter is fed by a fake model, and the loop by a stand-in
for the turn.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from frontdesk.cli.chat import run_chat
from frontdesk.cli.live_chat import (
    LiveCosts,
    chat_run_id,
    format_cost_line,
    live_banner,
    model_errors_on_stderr,
)
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, TurnTrace
from frontdesk.model_gateway.costs import BudgetError
from frontdesk.model_gateway.metering import Meter
from frontdesk.model_gateway.routes import ROUTES
from frontdesk.observability.logs import log_event
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


def test_a_ledger_that_becomes_unreadable_during_the_session_is_a_budget_error(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    costs = LiveCosts.begin(Meter(budget, route_id=ROUTE, run_id="chat-x"), budget)
    budget.ledger_path.mkdir()  # the ledger is now something that cannot be read
    with pytest.raises(BudgetError, match=r"spend\.jsonl: cannot be read \(Is a directory\)"):
        costs.cost_line()


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
async def test_a_stopped_meter_ends_the_loop_with_the_cost_but_not_the_reply_of_that_turn(
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
    # The turn the stop cut short printed no reply, but what it had cost (a request may have been
    # paid before the next one was refused), and "third" was never read.
    assert written == ["bot> echo first", "cost after reply 1", "cost after reply 2"]
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


# --- A failed model request -------------------------------------------------------------------------

TURN_LOGGER = logging.getLogger("frontdesk.conversation.turn")


@pytest.mark.parametrize("event", ["turn.failed", "classifier.failed", "responder.failed"])
def test_a_failed_turn_event_is_one_model_error_line_with_the_exception_type(
    event: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with model_errors_on_stderr():
        log_event(TURN_LOGGER, event, business_id="demo-clinic", reason="ModelHTTPError")
    captured = capsys.readouterr()
    assert captured.err == "model error: ModelHTTPError\n"
    assert captured.out == ""


def test_the_other_turn_events_print_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    with model_errors_on_stderr():
        # Its `reason` is a rejection code, not an exception type, and the turn did not fail.
        log_event(
            TURN_LOGGER, "turn.completed", business_id="demo-clinic", reason="classifier_error"
        )
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({}, id="none"),
        pytest.param({"reason": None}, id="null"),
        pytest.param({"reason": "free text, with spaces"}, id="text"),
        pytest.param({"reason": "two\nlines"}, id="newline"),
        pytest.param({"reason": "A" * 101}, id="too-long"),
    ],
)
def test_a_reason_that_is_not_a_type_name_is_never_shown(
    fields: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    with model_errors_on_stderr():
        log_event(TURN_LOGGER, "turn.failed", business_id="demo-clinic", **fields)
    assert capsys.readouterr().err == "model error: unknown\n"


def test_the_logger_is_left_as_it_was_after_a_normal_exit() -> None:
    TURN_LOGGER.setLevel(logging.WARNING)
    try:
        before = (list(TURN_LOGGER.handlers), TURN_LOGGER.level)
        with model_errors_on_stderr():
            assert TURN_LOGGER.level == logging.INFO  # turn.py logs at INFO
            assert len(TURN_LOGGER.handlers) == len(before[0]) + 1
        assert (TURN_LOGGER.handlers, TURN_LOGGER.level) == before
    finally:
        TURN_LOGGER.setLevel(logging.NOTSET)


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_the_logger_is_left_as_it_was_when_the_session_fails_or_is_interrupted(
    failure: type[BaseException],
) -> None:
    before = (list(TURN_LOGGER.handlers), TURN_LOGGER.level)
    with pytest.raises(failure), model_errors_on_stderr():
        raise failure
    assert (TURN_LOGGER.handlers, TURN_LOGGER.level) == before


def test_a_line_that_cannot_be_written_does_not_break_the_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def closed(*args: object, **kwargs: object) -> None:
        raise OSError("stderr is closed")

    monkeypatch.setattr("frontdesk.cli.live_chat.typer.echo", closed)
    with model_errors_on_stderr():
        log_event(TURN_LOGGER, "turn.failed", business_id="demo-clinic", reason="ConnectError")
    # The logging module reports it and goes on; the exception does not reach the turn.
    assert "Logging error" in capsys.readouterr().err


# --- Ctrl-C at the prompt ---------------------------------------------------------------------------

WAIT_LIMIT = 8.0  # how long a stand-in for readline() waits for a person who never types


class PendingRead:
    """A `read_line` that waits for a person, as `readline()` does, until the test lets it go."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> str | None:
        self.started.set()
        self.release.wait(timeout=WAIT_LIMIT)
        return None


async def _cancelled_while_waiting_for_a_line(read: PendingRead) -> None:
    """The session as `asyncio.run` leaves it at the first Ctrl-C: its task is cancelled."""
    task = asyncio.create_task(
        run_chat(cast(TurnRuntime, None), read_line=read, write=lambda line: None, show_trace=False)
    )
    assert await asyncio.to_thread(read.started.wait, 5)  # the read is under way
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _join_the_input_threads() -> None:
    for thread in threading.enumerate():
        if thread.name == "chat-input":
            thread.join(timeout=5)


def test_a_session_waiting_for_a_line_ends_at_once_when_it_is_cancelled() -> None:
    read = PendingRead()
    started = time.monotonic()
    try:
        # `asyncio.run` also joins the loop's own executor when it ends: a line read there would
        # keep the session (and the budget it holds) alive until the person pressed Enter.
        asyncio.run(_cancelled_while_waiting_for_a_line(read))
        waited = time.monotonic() - started
    finally:
        read.release.set()
    assert waited < WAIT_LIMIT / 2


def test_a_line_typed_after_the_loop_closed_is_dropped_without_a_thread_error() -> None:
    read = PendingRead()
    errors: list[threading.ExceptHookArgs] = []
    previous, threading.excepthook = threading.excepthook, errors.append
    try:
        asyncio.run(_cancelled_while_waiting_for_a_line(read))
        read.release.set()  # the person types at last, in a process whose loop is gone
        _join_the_input_threads()
    finally:
        read.release.set()
        threading.excepthook = previous
    assert errors == []


def test_a_line_typed_after_the_wait_was_cancelled_is_dropped_by_a_loop_that_goes_on() -> None:
    read = PendingRead()
    problems: list[dict[str, object]] = []

    async def session() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: problems.append(context))
        await _cancelled_while_waiting_for_a_line(read)
        read.release.set()  # the person types after the wait was cancelled
        await asyncio.to_thread(_join_the_input_threads)
        await asyncio.sleep(0.05)  # the loop runs what the thread handed over

    asyncio.run(session())
    assert problems == []  # setting a result on the cancelled wait would be one


def test_an_interrupt_raised_while_waiting_for_a_line_reaches_the_session() -> None:
    def interrupted() -> str | None:
        raise KeyboardInterrupt

    session = run_chat(
        cast(TurnRuntime, None), read_line=interrupted, write=lambda line: None, show_trace=False
    )
    with pytest.raises(KeyboardInterrupt):
        # The limit only turns a session that never hears of the interrupt into a failure.
        asyncio.run(asyncio.wait_for(session, timeout=5))
