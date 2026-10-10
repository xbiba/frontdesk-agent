"""Live model construction. The only module that talks to a model provider SDK."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx2
from google.genai import Client
from google.genai.client import DebugConfig
from google.genai.types import HttpOptions, HttpRetryOptions, ThinkingLevel
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.settings import ModelSettings

from frontdesk.model_gateway.routes import CHAT_PROVIDERS, ModelRoute, UnapprovedRouteError

# The Vertex AI hosts pinned for each approved location. Left to choose, the SDKs read
# ANTHROPIC_VERTEX_BASE_URL, GOOGLE_VERTEX_BASE_URL and GOOGLE_CLOUD_LOCATION from the
# environment, so one stray variable would move the text to another host. A location with no
# entry here has no live model, whatever its route says.
VERTEX_HOSTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "eu": "https://aiplatform.eu.rep.googleapis.com",
        "europe-west1": "https://europe-west1-aiplatform.googleapis.com",
    }
)
VERTEX_API_VERSION: Final = "v1"
# One turn may take 45 seconds in all (conversation.turn.TURN_TIMEOUT_SECONDS), which bounds the
# requests of a turn and their retries. For Gemini, google-genai sends this value as the timeout of
# each request, and that replaces the 5 second connect timeout of model_http_client (Claude's
# client keeps it): a Gemini connection may take up to 30 seconds to open.
MODEL_TIMEOUT_SECONDS: Final = 30.0
# 429 and 5xx answers, timeouts and refused connections are retried with growing waits of 2, 4 and
# 8 seconds (M3 design §6). The engine retries, not the SDK, so that each attempt is metered
# (frontdesk.model_gateway.metering): the SDK makes one attempt. The delays and statuses below are
# the policy the engine follows, and a test holds the two together.
GOOGLE_RETRIES: Final = HttpRetryOptions(
    attempts=1,
    initial_delay=2.0,
    max_delay=8.0,
    exp_base=2.0,
    http_status_codes=[429, 500, 502, 503, 504],
)


def pinned_location(route: ModelRoute) -> str:
    """The route's location, if Vertex AI has a pinned host for it; any other is refused."""
    if route.region is None or route.region not in VERTEX_HOSTS:
        raise UnapprovedRouteError(
            f"model route '{route.route_id}' has no pinned endpoint for region {route.region!r}"
        )
    return route.region


def _require_project(route: ModelRoute, gcp_project: str | None) -> str:
    """The Google Cloud project to call. An empty one would make the SDK leave the project and the
    location out of the request path."""
    if gcp_project is None or not gcp_project.strip():
        raise ValueError(f"FRONTDESK_GCP_PROJECT is required for the {route.route_id} route")
    return gcp_project


def model_http_client() -> httpx2.AsyncClient:
    """The HTTP client of every live model: nothing is taken from the environment.

    A client that trusts the environment sends its requests through HTTP(S)_PROXY or ALL_PROXY and
    trusts the certificates of SSL_CERT_FILE or SSL_CERT_DIR (final review P63). This one connects
    directly and uses the system trust store.
    """
    return httpx2.AsyncClient(
        trust_env=False, timeout=httpx2.Timeout(MODEL_TIMEOUT_SECONDS, connect=5.0)
    )


def google_http_options(location: str) -> HttpOptions:
    return HttpOptions(
        base_url=f"{VERTEX_HOSTS[location]}/",
        api_version=VERTEX_API_VERSION,
        httpx_async_client=model_http_client(),
        timeout=int(MODEL_TIMEOUT_SECONDS * 1000),  # milliseconds
        retry_options=GOOGLE_RETRIES,
    )


def google_client(route: ModelRoute, *, gcp_project: str) -> Client:
    """A google-genai client for Vertex AI, pinned to the route's host, location and project.

    Explicit values win over GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_VERTEX_BASE_URL
    and an API key in the environment. The debug configuration is given in full, so that
    GOOGLE_GENAI_CLIENT_MODE cannot make the client record requests on disk.
    """
    location = pinned_location(route)
    project = _require_project(route, gcp_project)
    return Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=google_http_options(location),
        debug_config=DebugConfig(client_mode=None, replays_directory=None, replay_id=None),
    )


def build_live_model(route: ModelRoute, *, gcp_project: str | None) -> Model:
    if route.provider not in CHAT_PROVIDERS:
        raise ValueError(f"route '{route.route_id}' has no live model")
    location = pinned_location(route)
    project = _require_project(route, gcp_project)
    if route.provider == "google-vertex":
        client = google_client(route, gcp_project=project)
        return GoogleModel(route.model_name, provider=GoogleCloudProvider(client=client))
    if route.provider == "anthropic-vertex":
        from anthropic import AsyncAnthropicVertex

        vertex = AsyncAnthropicVertex(
            region=location,
            project_id=project,
            base_url=f"{VERTEX_HOSTS[location]}/{VERTEX_API_VERSION}",
            http_client=model_http_client(),
        )
        return AnthropicModel(route.model_name, provider=AnthropicProvider(anthropic_client=vertex))
    # A provider added to CHAT_PROVIDERS needs its builder above; none falls through to Claude's.
    raise ValueError(f"route '{route.route_id}' has no live model: no builder for {route.provider}")


def live_model_settings(route: ModelRoute, *, max_tokens: int) -> ModelSettings:
    """What a live model is asked with: a cap on output tokens and its provider's thinking lever.

    Opus 5.5 always thinks adaptively and rejects sampling parameters; effort is its lever. Gemini
    3.1 Flash-Lite counts its thinking in the output tokens and takes a thinking level: the
    classifier and the responder fill closed forms, so it thinks as little as it can.
    """
    if route.provider == "google-vertex":
        return GoogleModelSettings(
            max_tokens=max_tokens,
            google_thinking_config={"thinking_level": ThinkingLevel.MINIMAL},
        )
    if route.provider == "anthropic-vertex":
        return AnthropicModelSettings(max_tokens=max_tokens, anthropic_effort="medium")
    raise ValueError(f"route '{route.route_id}' has no live model")
