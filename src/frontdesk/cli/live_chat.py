"""A chat with the real model on a demo business, paid from the M3 budget (M3 Task 6b).

Gemini is approved for evaluation data only, until Vertex AI's data retention is checked (M3 design
§4.2): no conversation may carry a patient's words. The demo businesses hold synthetic data, and a
person typing test questions produces evaluation data, so this chat builds its runtime for
`DataClass.EVALUATION` and says so before the first prompt. Every request is metered, each reply
shows what it cost, and the session holds the budget from its first request to its last line: one
live use runs at a time. Nothing is written to disk but the ledger's amounts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import typer

from frontdesk.cli.chat import run_chat
from frontdesk.composition import (
    Deployment,
    NoActiveKnowledgeError,
    build_turn_runtime,
    open_deployment,
)
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.knowledge.versions import EmbedderMismatchError
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


def _fail(message: str) -> typer.Exit:
    """Print `error: <message>` and give back the exit to raise; no traceback follows."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def chat_run_id(now: datetime) -> str:
    """The run a session's ledger lines are filed under: `chat-` and the UTC time it began."""
    return f"chat-{now.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def eur_left(budget: Budget, spent_usd: float) -> float:
    """What the cap leaves once `spent_usd` is spent; never a negative amount."""
    return max(budget.eur(budget.cap_usd - spent_usd), 0.0)


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
        return cls(meter, budget, budget.spent_usd())

    @property
    def stop_reason(self) -> str | None:
        return self.meter.stop_reason

    def cost_line(self) -> str:
        turn_usd = sum(call.usd for call in self.meter.drain())
        spent_usd = self.budget.spent_usd()
        return format_cost_line(
            turn_eur=self.budget.eur(turn_usd),
            session_eur=self.budget.eur(spent_usd - self.spent_before_usd),
            left_eur=eur_left(self.budget, spent_usd),
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
        raise _fail("--live needs --budget: every live request is counted against it")
    settings = load_settings(env_file)
    route = _live_route(settings.model_route, env_file)
    try:
        budget = load_budget(budget_path)
        meter = Meter(budget, route_id=route.route_id, run_id=chat_run_id(datetime.now(UTC)))
        with budget.exclusive():
            stopped = _converse(
                settings, env_file, route, budget, meter, show_trace=show_trace, read_line=read_line
            )
    except BudgetError as error:
        raise _fail(str(error)) from error
    if stopped is not None:
        typer.echo(f"stopped: {stopped}", err=True)
        raise typer.Exit(1)


def _live_route(route_id: str, env_file: Path) -> ModelRoute:
    """The route to chat on: one that evaluation data may reach and that calls a live model."""
    try:
        route = require_route(route_id, DataClass.EVALUATION)
    except UnapprovedRouteError as error:
        raise _fail(f"{error} (--env-file {env_file})") from error
    if route.provider not in CHAT_PROVIDERS:
        raise _fail(f"--live needs a live model route; FRONTDESK_MODEL_ROUTE is {route_id}")
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
    runtime = _build_runtime(deployment, env_file, meter)
    typer.echo(live_banner(route, budget, spent_usd=costs.spent_before_usd))
    typer.echo(f"{deployment.pack.name} ({deployment.pack.profile}) — type 'exit' to quit")
    return asyncio.run(
        run_chat(runtime, read_line=read_line, write=typer.echo, show_trace=show_trace, costs=costs)
    )


def _build_runtime(deployment: Deployment, env_file: Path, meter: Meter) -> TurnRuntime:
    """The pipeline for evaluation data, with every model request going through the meter."""
    try:
        return build_turn_runtime(
            deployment, data_class=DataClass.EVALUATION, wrap_model=meter.wrap
        )
    except NoActiveKnowledgeError as error:
        raise _fail(
            f"{error}. Draft, approve and activate a version first "
            f"(demo businesses: frontdesk knowledge bootstrap-demo --env-file {env_file})."
        ) from error
    except EmbedderMismatchError as error:
        raise _fail(f"{error} --env-file {env_file}") from error
    except ValueError as error:
        # What is left is a setting the live route needs, such as the Google Cloud project, or a
        # route whose endpoint is not pinned. The route itself was approved by `_live_route`.
        raise _fail(str(error)) from error
