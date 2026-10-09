"""Terminal chat for the keyless demo: one line in, one reply and its trace out."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Final

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


async def run_chat(
    runtime: TurnRuntime,
    *,
    read_line: Callable[[], str | None],
    write: Callable[[str], None],
    show_trace: bool,
) -> None:
    state = ConversationState()
    while (line := await asyncio.to_thread(read_line)) is not None:
        message = line.strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        result, state = await handle_message(message, state, runtime)
        write(f"bot> {result.text}")
        if show_trace:
            write(f"     {format_trace(result)}")
