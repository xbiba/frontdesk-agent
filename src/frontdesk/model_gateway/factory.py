"""Live model construction. The only module that talks to a model provider SDK."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from frontdesk.model_gateway.routes import ModelRoute, UnapprovedRouteError

# The Vertex AI endpoint pinned for each approved region: for "eu", the multi-region endpoint the
# Anthropic SDK itself computes. Left to choose, the SDK reads ANTHROPIC_VERTEX_BASE_URL from the
# environment, so one stray variable would move patient-derived text to another host. A region
# with no entry here has no live model, whatever its route says.
VERTEX_ENDPOINTS: Final[Mapping[str, str]] = MappingProxyType(
    {"eu": "https://aiplatform.eu.rep.googleapis.com/v1"}
)


def build_live_model(route: ModelRoute, *, gcp_project: str | None) -> Model:
    if route.provider != "anthropic-vertex":
        raise ValueError(f"route '{route.route_id}' has no live model")
    region = route.region
    if region is None or region not in VERTEX_ENDPOINTS:
        raise UnapprovedRouteError(
            f"model route '{route.route_id}' has no pinned endpoint for region {region!r}"
        )
    if not gcp_project:
        raise ValueError("FRONTDESK_GCP_PROJECT is required for the vertex-eu-opus route")
    from anthropic import AsyncAnthropicVertex

    client = AsyncAnthropicVertex(
        region=region, project_id=gcp_project, base_url=VERTEX_ENDPOINTS[region]
    )
    return AnthropicModel(route.model_name, provider=AnthropicProvider(anthropic_client=client))


def live_model_settings(*, max_tokens: int) -> AnthropicModelSettings:
    # Opus 5.5 always thinks adaptively and rejects sampling parameters; effort is the lever.
    return AnthropicModelSettings(max_tokens=max_tokens, anthropic_effort="medium")
