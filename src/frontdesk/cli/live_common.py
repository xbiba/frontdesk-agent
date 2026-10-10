"""What the live commands share: the checks before they spend, and how they read the budget.

`chat --live` and `eval turns --live` refuse the same environment, send the same free request to
prove the credentials, the API and the model (M3 design §5, §6), and read and show the budget the
same way. The ledger check is the meter's (`Meter.check_ledger`): both commands have a meter.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from datetime import UTC, datetime

from frontdesk.cli.runtime import fail
from frontdesk.model_gateway.access import (
    RouteAccessError,
    check_route_access,
    environment_overrides,
)
from frontdesk.model_gateway.costs import Budget, BudgetError
from frontdesk.model_gateway.routes import ModelRoute, UnapprovedRouteError


def refuse_environment_overrides(environ: Mapping[str, str] | None = None) -> None:
    """Refuse a live run while a proxy, a certificate bundle or another billing project is set in
    the environment: Google's token refresh would read it, and the models would be reached another
    way than the pinned one. The line names each variable that is set."""
    overrides = environment_overrides(os.environ if environ is None else environ)
    if overrides:
        raise fail(
            "a live run does not go through proxies, other certificates or another billing "
            "project: unset " + ", ".join(overrides)
        )


def refuse_retired(route: ModelRoute) -> None:
    """Refuse a route whose model Google has retired: from the day it retires, not the day after."""
    if route.retires_on is not None and route.retires_on <= datetime.now(UTC).date():
        raise fail(
            f"route '{route.route_id}': Google retired {route.model_name} on {route.retires_on}"
        )


def check_access(route: ModelRoute, gcp_project: str | None) -> None:
    """Send the free request that proves the credentials, the API and the model at the route's
    pinned endpoint; a failure is an `error:` line that says what to do."""
    if gcp_project is None or not gcp_project.strip():
        raise fail(f"FRONTDESK_GCP_PROJECT is required for the {route.route_id} route")
    try:
        asyncio.run(check_route_access(route, gcp_project=gcp_project))
    except (RouteAccessError, UnapprovedRouteError) as error:
        raise fail(str(error)) from error


def spent_usd(budget: Budget) -> float:
    """What the ledger holds. A ledger that cannot be read at all (a directory, no permission) is a
    BudgetError that names it, like every other problem with the budget."""
    try:
        return budget.spent_usd()
    except OSError as error:
        raise BudgetError(
            f"{budget.ledger_path}: cannot be read ({error.strerror or error})"
        ) from error


def eur_left(budget: Budget, spent: float) -> float:
    """What the cap leaves once `spent` US dollars are spent; never a negative amount."""
    return max(budget.eur(budget.cap_usd - spent), 0.0)
