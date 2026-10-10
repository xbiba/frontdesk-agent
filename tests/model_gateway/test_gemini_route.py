"""The Gemini route: approved for evaluation data only, pinned to Vertex AI `eu`, deaf to the
environment. Nothing here sends a request: tests forbid model requests (conftest)."""

from __future__ import annotations

from datetime import date

import httpx2
import pytest
from google.genai.types import ThinkingLevel
from pydantic_ai.models.google import GoogleModel

from frontdesk.model_gateway.factory import (
    VERTEX_HOSTS,
    build_live_model,
    google_client,
    live_model_settings,
    model_http_client,
)
from frontdesk.model_gateway.routes import ROUTES, DataClass, UnapprovedRouteError, require_route

GEMINI = "vertex-eu-gemini-flash-lite"
EU_HOST = "https://aiplatform.eu.rep.googleapis.com/"
STRAY_ENVIRONMENT = {
    "GOOGLE_VERTEX_BASE_URL": "https://models.stray.example/",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "GOOGLE_CLOUD_PROJECT": "someone-elses-project",
    "GOOGLE_API_KEY": "not-a-real-key",
    "GOOGLE_GENAI_USE_VERTEXAI": "false",
    "GOOGLE_GENAI_CLIENT_MODE": "record",
    "GOOGLE_GENAI_REPLAYS_DIRECTORY": "replays",
    "HTTPS_PROXY": "http://proxy.stray.example:3128",
    "ALL_PROXY": "http://proxy.stray.example:3128",
}


def test_the_gemini_route_is_pinned_to_one_model_version_on_the_eu_multi_region() -> None:
    route = require_route(GEMINI, DataClass.EVALUATION)
    assert (route.provider, route.model_name, route.region) == (
        "google-vertex",
        "gemini-3.1-flash-lite",
        "eu",
    )
    assert route.retires_on == date(2027, 5, 7)


def test_the_gemini_route_is_refused_for_patient_derived_text() -> None:
    with pytest.raises(UnapprovedRouteError, match="not approved for patient_derived"):
        require_route(GEMINI, DataClass.PATIENT_DERIVED)


def test_only_evaluation_data_may_reach_a_google_route() -> None:
    google = [route for route in ROUTES.values() if route.provider.startswith("google")]
    assert google  # the Gemini route and the two embedding candidates
    assert all(route.approved_for == frozenset({DataClass.EVALUATION}) for route in google)


def test_the_gemini_model_talks_to_the_eu_host_with_api_v1() -> None:
    model = build_live_model(require_route(GEMINI, DataClass.EVALUATION), gcp_project="demo")
    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-3.1-flash-lite"
    assert model.base_url == EU_HOST
    api = model.client._api_client
    assert (api.project, api.location, api.vertexai) == ("demo", "eu", True)
    assert api._http_options.api_version == "v1"
    # One attempt: the engine retries and meters each one (frontdesk.model_gateway.metering).
    assert api._http_options.retry_options.attempts == 1
    assert api._http_options.timeout == 30000  # milliseconds


def test_the_environment_moves_neither_host_nor_project_nor_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in STRAY_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    route = require_route(GEMINI, DataClass.EVALUATION)
    client = google_client(route, gcp_project="clinica-mcu")
    api = client._api_client
    assert api._http_options.base_url == EU_HOST
    assert (api.project, api.location, api.vertexai, api.api_key) == (
        "clinica-mcu",
        "eu",
        True,
        None,
    )
    assert type(api).__name__ == "BaseApiClient"  # not the recording replay client


def test_model_requests_ignore_proxies_and_certificates_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.stray.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/stray-ca.pem")
    client = model_http_client()  # a client that trusted the environment would fail to build
    url = httpx2.URL(f"{VERTEX_HOSTS['eu']}/v1/projects/p/locations/eu")
    assert client.trust_env is False
    assert client._transport_for_url(url) is client._transport  # no proxy transport


def test_the_gemini_client_uses_the_client_that_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.stray.example:3128")
    client = google_client(require_route(GEMINI, DataClass.EVALUATION), gcp_project="p")
    http = client._api_client._async_httpx_client
    assert isinstance(http, httpx2.AsyncClient)
    assert http.trust_env is False


def test_gemini_thinks_as_little_as_it_can_within_the_route_limits() -> None:
    route = require_route(GEMINI, DataClass.EVALUATION)
    settings = live_model_settings(route, max_tokens=route.output_limits.classifier)
    assert settings == {
        "max_tokens": 600,
        "google_thinking_config": {"thinking_level": ThinkingLevel.MINIMAL},
    }
    assert settings.keys().isdisjoint({"temperature", "top_p", "top_k", "anthropic_effort"})


def test_an_embedding_route_has_no_chat_model() -> None:
    route = require_route("vertex-ew1-gemini-embedding-001", DataClass.EVALUATION)
    with pytest.raises(ValueError, match="no live model"):
        build_live_model(route, gcp_project="p")


@pytest.mark.parametrize("project", ["", " ", "\t\n"])
def test_the_gemini_client_needs_a_project(project: str) -> None:
    # Given an empty project, the SDK drops the project and the location from the request path.
    route = require_route(GEMINI, DataClass.EVALUATION)
    with pytest.raises(ValueError, match="FRONTDESK_GCP_PROJECT"):
        google_client(route, gcp_project=project)
