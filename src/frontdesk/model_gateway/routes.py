"""Approved model routes per data class (blueprint §12; ADR-0004).

Text derived from patient messages — original or rewritten — may only reach a
route approved for it. Any other route is refused in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal


class DataClass(StrEnum):
    PATIENT_DERIVED = "patient_derived"


@dataclass(frozen=True)
class ModelRoute:
    route_id: str
    provider: Literal["offline", "anthropic-vertex"]
    model_name: str
    region: str | None
    approved_for: frozenset[DataClass]


ROUTES: Final[dict[str, ModelRoute]] = {
    "offline-demo": ModelRoute(
        route_id="offline-demo",
        provider="offline",
        model_name="deterministic-demo",
        region=None,
        approved_for=frozenset({DataClass.PATIENT_DERIVED}),
    ),
    "vertex-eu-opus": ModelRoute(
        route_id="vertex-eu-opus",
        provider="anthropic-vertex",
        model_name="claude-opus-5-5",
        region="eu",
        approved_for=frozenset({DataClass.PATIENT_DERIVED}),
    ),
}


class UnapprovedRouteError(ValueError):
    """A model route is unknown or not approved for the data it would receive."""


def require_route(route_id: str, data_class: DataClass) -> ModelRoute:
    route = ROUTES.get(route_id)
    if route is None:
        raise UnapprovedRouteError(f"model route '{route_id}' is not registered")
    if data_class not in route.approved_for:
        raise UnapprovedRouteError(f"model route '{route_id}' is not approved for {data_class}")
    return route
