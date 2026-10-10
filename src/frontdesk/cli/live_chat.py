"""A chat with the real model on a demo business, paid from the M3 budget (M3 Task 6b).

Gemini is approved for evaluation data only, until Vertex AI's data retention is checked (M3 design
§4.2): no conversation may carry a patient's words. The demo businesses hold synthetic data, and a
person typing test questions produces evaluation data, so this chat builds its runtime for
`DataClass.EVALUATION` and says so before the first prompt. Every request is metered, each reply
shows what it cost, and the session holds the budget from its first request to its last line: one
live use runs at a time. Before the banner it makes the checks a live evaluation makes
(`frontdesk.cli.live_common`): no proxy or certificate variable, a free request that proves the
credentials, the API and the model, and a ledger that takes a line. Nothing is written to disk but
the ledger's amounts.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer

from frontdesk.cli.chat import run_chat
from frontdesk.cli.live_common import (
    check_access,
    eur_left,
    refuse_environment_overrides,
    spent_usd,
)
from frontdesk.cli.runtime import build_runtime, fail
from frontdesk.composition import open_deployment
from frontdesk.conversation import turn
from frontdesk.model_gateway.costs import Budget, BudgetError, load_budget
from frontdesk.model_gateway.factory import pinned_location
from frontdesk.model_gateway.metering import Meter
from frontdesk.model_gateway.routes import (
    CHAT_PROVIDERS,
    DataClass,
    ModelRoute,
    UnapprovedRouteError,
    require_route,
)
from frontdesk.settings import Settings, load_settings

NOTICE: Final = "Test messages only: never type a real patient's data."
# turn.py logs these at INFO when a request to the model fails, with the class name of the exception
# in `reason` and never its message. The trace cannot tell them apart: a request that raises
# something other than an AgentRunError ends the turn with rule=error, and one that raises an
# AgentRunError (an HTTP 403 or 404 from Google) leaves rule=pass and a rejection code.
_FAILED_REQUEST_EVENTS: Final = frozenset({"turn.failed", "classifier.failed", "responder.failed"})
_TYPE_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,99}")


class _ModelErrorLine(logging.Handler):
    """Writes `model error: <ExceptionType>` to stderr when a turn logs a failed model request.

    The type is shown only if it looks like a type name: what else a record might hold under
    `reason` is not something to put on a terminal.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg not in _FAILED_REQUEST_EVENTS:
            return
        try:
            reason = (getattr(record, "fields", None) or {}).get("reason")
            shown = (
                reason if isinstance(reason, str) and _TYPE_NAME.fullmatch(reason) else "unknown"
            )
            typer.echo(f"model error: {shown}", err=True)
        except Exception:  # a log handler must not break the turn that is being logged
            self.handleError(record)


@contextmanager
def model_errors_on_stderr() -> Iterator[None]:
    """While it lasts, a turn that logs a failed model request writes `model error: <type>` to
    stderr, before the hand-off it falls back to.

    The CLI sets up no logging, and the failure is logged at INFO, so without this the person sees
    a hand-off reply and nothing says why (no credentials, Vertex AI not enabled, no network). The
    turn logger is given back as it was on every way out.
    """
    log = logging.getLogger(turn.__name__)
    handler, level = _ModelErrorLine(), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        yield
    finally:
        log.removeHandler(handler)
        log.setLevel(level)


def chat_run_id(now: datetime) -> str:
    """The run a session's ledger lines are filed under: `chat-` and the UTC time it began."""
    return f"chat-{now.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def live_banner(route: ModelRoute, budget: Budget, *, spent_usd: float) -> str:
    """The line before the first prompt: where the messages go, who pays, and the rule."""
    return (
        f"Live: {route.model_name} on Vertex AI {pinned_location(route)}, "
        f"metered against {budget.path} ({eur_left(budget, spent_usd):.2f} EUR left of "
        f"{budget.config.cap_eur:.2f} EUR). {NOTICE}"
    )


def format_cost_line(*, turn_eur: float, session_eur: float, left_eur: float) -> str:
    """The line after a reply: this turn, the session so far, and what the budget has left."""
    return f"cost: {turn_eur:.4f} EUR · session {session_eur:.4f} EUR · left {left_eur:.2f} EUR"


@dataclass(frozen=True)
class LiveCosts:
    """What a live session has spent. The turn's requests come from the meter; the rest is read
    from the ledger, which is the state (`Budget`): the session's total is what the ledger holds
    now less what it held when the session began, and what is left is the cap less what it holds."""

    meter: Meter
    budget: Budget
    spent_before_usd: float

    @classmethod
    def begin(cls, meter: Meter, budget: Budget) -> LiveCosts:
        return cls(meter, budget, spent_usd(budget))

    @property
    def stop_reason(self) -> str | None:
        return self.meter.stop_reason

    def cost_line(self) -> str:
        turn_usd = sum(call.usd for call in self.meter.drain())
        spent_now = spent_usd(self.budget)
        return format_cost_line(
            turn_eur=self.budget.eur(turn_usd),
            session_eur=self.budget.eur(spent_now - self.spent_before_usd),
            left_eur=eur_left(self.budget, spent_now),
        )


def run_live_chat(
    env_file: Path,
    budget_path: Path | None,
    *,
    show_trace: bool,
    read_line: Callable[[], str | None],
) -> None:
    """`frontdesk chat --live`. A problem prints `error: ...` and a stop prints `stopped: ...`;
    both end the process with exit code 1, and neither prints a traceback.

    The budget file is checked here by `load_budget`, not by the option, so that a missing or wrong
    file is the same `error:` line and exit code as every other problem with the budget.
    """
    if budget_path is None:
        raise fail("--live needs --budget: every live request is counted against it")
    settings = load_settings(env_file)
    route = _live_route(settings.model_route, env_file)
    refuse_environment_overrides()
    try:
        budget = load_budget(budget_path)
        meter = Meter(budget, route_id=route.route_id, run_id=chat_run_id(datetime.now(UTC)))
        with budget.exclusive():
            stopped = _converse(
                settings, env_file, route, budget, meter, show_trace=show_trace, read_line=read_line
            )
    except BudgetError as error:
        raise fail(str(error)) from error
    if stopped is not None:
        typer.echo(f"stopped: {stopped}", err=True)
        raise typer.Exit(1)


def _live_route(route_id: str, env_file: Path) -> ModelRoute:
    """The route to chat on: one that evaluation data may reach and that calls a live model."""
    try:
        route = require_route(route_id, DataClass.EVALUATION)
    except UnapprovedRouteError as error:
        raise fail(f"{error} (--env-file {env_file})") from error
    if route.provider not in CHAT_PROVIDERS:
        raise fail(f"--live needs a live model route; FRONTDESK_MODEL_ROUTE is {route_id}")
    return route


def _converse(
    settings: Settings,
    env_file: Path,
    route: ModelRoute,
    budget: Budget,
    meter: Meter,
    *,
    show_trace: bool,
    read_line: Callable[[], str | None],
) -> str | None:
    """The session, once it holds the budget. Returns why the meter stopped it, or None."""
    costs = LiveCosts.begin(meter, budget)
    deployment = open_deployment(settings)
    runtime = build_runtime(
        deployment, env_file, data_class=DataClass.EVALUATION, wrap_model=meter.wrap
    )
    check_access(route, settings.gcp_project)
    meter.check_ledger()  # the last gate before the first paid request
    typer.echo(live_banner(route, budget, spent_usd=costs.spent_before_usd))
    typer.echo(f"{deployment.pack.name} ({deployment.pack.profile}) — type 'exit' to quit")
    with model_errors_on_stderr():
        return asyncio.run(
            run_chat(
                runtime, read_line=read_line, write=typer.echo, show_trace=show_trace, costs=costs
            )
        )
