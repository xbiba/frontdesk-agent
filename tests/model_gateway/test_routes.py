import httpx2
import pytest
from anthropic import AsyncAnthropicVertex
from pydantic_ai.models.anthropic import AnthropicModel

from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.routes import (
    CHAT_PROVIDERS,
    ROUTES,
    DataClass,
    ModelRoute,
    UnapprovedRouteError,
    require_route,
)

# What the Anthropic SDK computes for Vertex region "eu"; its client ends a base URL with a slash.
EU_ENDPOINT = "https://aiplatform.eu.rep.googleapis.com/v1/"


def test_vertex_route_is_the_approved_eu_opus_route() -> None:
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    assert route.model_name == "claude-opus-5-5"
    assert route.region == "eu"
    assert route.provider == "anthropic-vertex"


def test_unknown_routes_are_refused() -> None:
    with pytest.raises(UnapprovedRouteError, match="not registered"):
        require_route("openai-us", DataClass.PATIENT_DERIVED)


def test_a_registered_route_not_approved_for_the_data_class_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = ModelRoute(
        route_id="vertex-unapproved",
        provider="anthropic-vertex",
        model_name="claude-opus-5-5",
        region="eu",
        approved_for=frozenset(),
    )
    monkeypatch.setitem(ROUTES, route.route_id, route)
    with pytest.raises(UnapprovedRouteError, match="not approved"):
        require_route(route.route_id, DataClass.PATIENT_DERIVED)


def test_live_model_needs_a_project() -> None:
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    with pytest.raises(ValueError, match="FRONTDESK_GCP_PROJECT"):
        build_live_model(route, gcp_project=None)


def test_the_offline_route_has_no_live_model() -> None:
    route = require_route("offline-demo", DataClass.PATIENT_DERIVED)
    with pytest.raises(ValueError, match="no live model"):
        build_live_model(route, gcp_project="p")


@pytest.mark.parametrize("route_id", ["vertex-eu-opus", "vertex-eu-gemini-flash-lite"])
@pytest.mark.parametrize("project", [None, "", "  "])
def test_a_blank_project_is_refused_for_every_chat_provider(
    route_id: str, project: str | None
) -> None:
    with pytest.raises(ValueError, match="FRONTDESK_GCP_PROJECT"):
        build_live_model(ROUTES[route_id], gcp_project=project)


def test_a_chat_provider_without_a_builder_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # A provider added to CHAT_PROVIDERS must not become the Claude client by falling through.
    monkeypatch.setattr(
        "frontdesk.model_gateway.factory.CHAT_PROVIDERS", CHAT_PROVIDERS | {"mistral-vertex"}
    )
    route = ModelRoute(
        route_id="vertex-eu-mistral",
        provider="mistral-vertex",
        model_name="mistral-large",
        region="eu",
        approved_for=frozenset({DataClass.EVALUATION}),
    )
    with pytest.raises(ValueError, match="mistral-vertex"):
        build_live_model(route, gcp_project="demo-project")


def test_a_route_without_a_live_model_has_no_live_settings() -> None:
    route = require_route("offline-demo", DataClass.PATIENT_DERIVED)
    with pytest.raises(ValueError, match="no live model"):
        live_model_settings(route, max_tokens=800)


def test_live_settings_use_medium_effort_without_sampling() -> None:
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    settings = live_model_settings(route, max_tokens=800)
    assert settings["anthropic_effort"] == "medium"
    assert settings["max_tokens"] == 800
    assert "temperature" not in settings


def test_the_vertex_route_is_pinned_to_the_eu_endpoint() -> None:
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    model = build_live_model(route, gcp_project="demo-project")
    assert model.model_name == "claude-opus-5-5"
    assert model.base_url == EU_ENDPOINT
    assert isinstance(model, AnthropicModel)
    client = model.client
    assert isinstance(client, AsyncAnthropicVertex)
    assert client.project_id == "demo-project"
    assert client.region == "eu"
    settings = live_model_settings(route, max_tokens=800)
    assert settings["anthropic_effort"] == "medium"
    assert settings.keys().isdisjoint({"temperature", "top_p", "top_k"})


def test_a_stray_vertex_base_url_in_the_environment_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "https://models.stray.example/v1")
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    model = build_live_model(route, gcp_project="demo-project")
    assert model.base_url == EU_ENDPOINT


def test_the_claude_client_ignores_proxies_and_certificates_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.stray.example:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.stray.example:3128")
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    http = build_live_model(route, gcp_project="p").client._client
    assert isinstance(http, httpx2.AsyncClient)
    assert http.trust_env is False
    # no proxy transport is mounted
    assert http._transport_for_url(httpx2.URL(EU_ENDPOINT)) is http._transport
    assert http.timeout == httpx2.Timeout(30.0, connect=5.0)


@pytest.mark.parametrize("region", ["us", None])
def test_a_region_without_a_pinned_endpoint_is_refused(region: str | None) -> None:
    route = ModelRoute(
        route_id="vertex-other-opus",
        provider="anthropic-vertex",
        model_name="claude-opus-5-5",
        region=region,
        approved_for=frozenset({DataClass.PATIENT_DERIVED}),
    )
    with pytest.raises(UnapprovedRouteError, match="no pinned endpoint"):
        build_live_model(route, gcp_project="demo-project")


@pytest.mark.parametrize(
    "route", [route for route in ROUTES.values() if route.fallback or route.second_try]
)
def test_a_fallback_and_a_second_try_take_the_data_the_route_takes(route: ModelRoute) -> None:
    # The model a turn moves to may receive what the route receives.
    for other_id in filter(None, (route.fallback, route.second_try)):
        other = ROUTES[other_id]
        assert other.provider in CHAT_PROVIDERS
        assert route.approved_for <= other.approved_for


@pytest.mark.parametrize("route", [route for route in ROUTES.values() if route.second_try])
def test_a_second_try_thinks_as_its_route(route: ModelRoute) -> None:
    # It writes the reply with the route's own settings (frontdesk.model_gateway.second_try).
    assert route.second_try is not None
    assert ROUTES[route.second_try].thinking == route.thinking
