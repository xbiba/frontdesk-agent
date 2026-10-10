"""Terminal chat: one line in, one reply and its trace out (and, when metered, what it cost)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Final, Protocol

from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnResult, TurnRuntime, handle_message

_ORDER_SOURCE_PREFIX: Final = "order:"


def _shown(source: str) -> str:
    """A source as the trace line shows it: an order's reference is not for display.

    `order:ORD-10427` is a live source, and the reference identifies a customer's order. The other
    sources are record keys and document keys of the business, which are safe to show.
    """
    return "order" if source.startswith(_ORDER_SOURCE_PREFIX) else source


def format_trace(result: TurnResult) -> str:
    trace = result.trace
    sources = dict.fromkeys(_shown(source) for source in trace.sources)
    parts = [
        f"rule={trace.rule}",
        f"category={trace.category or '-'}",
        f"intent={trace.intent or '-'}",
        f"sources={','.join(sources) or '-'}",
        f"outcome={result.outcome}",
    ]
    if trace.rejection:
        parts.append(f"rejection={trace.rejection}")
    if trace.check_reasons:
        parts.append("checks=" + ",".join(trace.check_reasons))
    return "[" + " ".join(parts) + "]"


class SessionCosts(Protocol):
    """What a metered chat asks of its session (`frontdesk.cli.live_chat.LiveCosts`)."""

    @property
    def stop_reason(self) -> str | None:
        """Why the session must end, once it must; None while it may go on."""
        ...

    def cost_line(self) -> str:
        """What the turn that just ended cost, as the line printed after its reply."""
        ...


async def _read_on_a_daemon_thread(read_line: Callable[[], str | None]) -> str | None:
    """`read_line()` on a thread of its own, which the interpreter does not wait for.

    `asyncio.to_thread` would run it on the loop's default executor, and `asyncio.run` joins that
    executor when it ends: after a Ctrl-C at the prompt the session would live on until the person
    pressed Enter, and a session that holds the budget would keep it until then. A daemon thread is
    left behind, so the interrupt ends the session at once.
    """
    loop = asyncio.get_running_loop()
    line: asyncio.Future[str | None] = loop.create_future()

    def settle(result: str | None, error: BaseException | None) -> None:
        if line.done():  # the wait was cancelled
            return
        if error is None:
            line.set_result(result)
        else:
            line.set_exception(error)

    def read() -> None:
        result: str | None = None
        error: BaseException | None = None
        try:
            result = read_line()
        except BaseException as caught:  # handed to the task that waits for the line
            error = caught
        with suppress(RuntimeError):  # the loop has closed: nobody is waiting for the line
            loop.call_soon_threadsafe(settle, result, error)

    threading.Thread(target=read, name="chat-input", daemon=True).start()
    return await line


async def run_chat(
    runtime: TurnRuntime,
    *,
    read_line: Callable[[], str | None],
    write: Callable[[str], None],
    show_trace: bool,
    costs: SessionCosts | None = None,
) -> str | None:
    """Answer each line until the person leaves, and return None then.

    A metered chat (`costs`) ends when its meter stops, and returns the reason. The turn that the
    stop cut short shows no reply, because what it fell back to is no answer to the person's
    message, but it shows what it cost: a request may have been paid before the next was refused.
    """
    state = ConversationState()
    while (line := await _read_on_a_daemon_thread(read_line)) is not None:
        message = line.strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        result, state = await handle_message(message, state, runtime)
        if costs is not None and (reason := costs.stop_reason) is not None:
            write(costs.cost_line())
            return reason
        write(f"bot> {result.text}")
        if show_trace:
            write(f"     {format_trace(result)}")
        if costs is not None:
            write(costs.cost_line())
    return None
