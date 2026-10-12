"""Approved model routes per data class (ADR-0004).

Text derived from patient messages — original or rewritten — may only reach a route approved for
`patient_derived`. Evaluation data (synthetic questions and the business's approved public
information, never a patient's message) may reach a route approved for `evaluation`. Any other
route is refused in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Final, Literal


class DataClass(StrEnum):
    PATIENT_DERIVED = "patient_derived"
    EVALUATION = "evaluation"


Provider = Literal["offline", "anthropic-vertex", "google-vertex", "google-vertex-embedding"]
CHAT_PROVIDERS: Final = frozenset({"anthropic-vertex", "google-vertex"})


@dataclass(frozen=True)
class OutputLimits:
    """The most output tokens each agent may spend in one request, thinking included (M3 §4.3)."""

    classifier: int = 600
    responder: int = 1200
    agent: int = 2000  # the reception agent: thinking, a tool call or the reply


@dataclass(frozen=True)
class ModelRoute:
    route_id: str
    provider: Provider
    model_name: str
    region: str | None
    approved_for: frozenset[DataClass]
    output_limits: OutputLimits = field(default_factory=OutputLimits)
    # The earliest retirement date Google lists for the pinned version; None when none is listed.
    retires_on: date | None = None
    # Vector size of an embedding route; None for chat routes.
    dimension: int | None = None
    # Gemini's thinking level for this model: the lowest it accepts (3.8 Flash refuses "minimal").
    thinking: str = "minimal"


ROUTES: Final[dict[str, ModelRoute]] = {
    "offline-demo": ModelRoute(
        route_id="offline-demo",
        provider="offline",
        model_name="deterministic-demo",
        region=None,
        approved_for=frozenset({DataClass.PATIENT_DERIVED, DataClass.EVALUATION}),
    ),
    "vertex-eu-opus": ModelRoute(
        route_id="vertex-eu-opus",
        provider="anthropic-vertex",
        model_name="claude-opus-5-5",
        region="eu",
        approved_for=frozenset({DataClass.PATIENT_DERIVED, DataClass.EVALUATION}),
    ),
    # Evaluation data only, until Vertex's data retention for Gemini is checked (spec §18,
    # technical question 2; M3 design §4.2). Google lists one version of this model: GA since
    # 2026-05-07, retiring "May 7, 2027 or later"; served on the eu multi-region, not europe-west1.
    "vertex-eu-gemini-flash-lite": ModelRoute(
        route_id="vertex-eu-gemini-flash-lite",
        provider="google-vertex",
        model_name="gemini-3.1-flash-lite",
        region="eu",
        approved_for=frozenset({DataClass.EVALUATION}),
        retires_on=date(2027, 5, 7),
    ),
    # The reception agent (Albi's live chat, 2026-10-12: Flash-Lite read "Friday, what time?" as a
    # pick). GA since 2026-09-02 on the eu multi-region; Google lists no retirement date yet.
    # Evaluation data only, as for Flash-Lite, until Vertex's data retention for Gemini is checked.
    "vertex-eu-gemini-flash": ModelRoute(
        route_id="vertex-eu-gemini-flash",
        provider="google-vertex",
        model_name="gemini-3.8-flash",
        region="eu",
        approved_for=frozenset({DataClass.EVALUATION}),
        thinking="low",
    ),
    # The embedding candidates of the M3 experiment (design §4.5). Neither is served on the eu
    # multi-region; both are on europe-west1 (Belgium).
    "vertex-ew1-text-multilingual-embedding-002": ModelRoute(
        route_id="vertex-ew1-text-multilingual-embedding-002",
        provider="google-vertex-embedding",
        model_name="text-multilingual-embedding-002",
        region="europe-west1",
        approved_for=frozenset({DataClass.EVALUATION}),
        retires_on=date(2027, 4, 1),
        dimension=768,
    ),
    "vertex-ew1-gemini-embedding-001": ModelRoute(
        route_id="vertex-ew1-gemini-embedding-001",
        provider="google-vertex-embedding",
        model_name="gemini-embedding-001",
        region="europe-west1",
        approved_for=frozenset({DataClass.EVALUATION}),
        retires_on=date(2028, 5, 20),
        dimension=3072,
    ),
}


class UnapprovedRouteError(ValueError):
    """A model route is unknown, not approved for the data it would receive, or not one that can
    answer messages."""


def require_route(route_id: str, data_class: DataClass) -> ModelRoute:
    route = ROUTES.get(route_id)
    if route is None:
        raise UnapprovedRouteError(f"model route '{route_id}' is not registered")
    if data_class not in route.approved_for:
        raise UnapprovedRouteError(f"model route '{route_id}' is not approved for {data_class}")
    return route
