"""Live model construction. The only module that talks to a model provider SDK."""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from frontdesk.model_gateway.routes import ModelRoute


def build_live_model(route: ModelRoute, *, gcp_project: str | None) -> Model:
    if route.provider != "anthropic-vertex":
        raise ValueError(f"route '{route.route_id}' has no live model")
    if not gcp_project:
        raise ValueError("FRONTDESK_GCP_PROJECT is required for the vertex-eu-opus route")
    from anthropic import AsyncAnthropicVertex

    client = AsyncAnthropicVertex(region=route.region or "eu", project_id=gcp_project)
    return AnthropicModel(route.model_name, provider=AnthropicProvider(anthropic_client=client))


def live_model_settings(*, max_tokens: int) -> AnthropicModelSettings:
    # Opus 5.5 always thinks adaptively and rejects sampling parameters; effort is the lever.
    return AnthropicModelSettings(max_tokens=max_tokens, anthropic_effort="medium")
