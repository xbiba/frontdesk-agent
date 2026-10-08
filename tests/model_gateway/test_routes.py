import pytest

from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.routes import DataClass, UnapprovedRouteError, require_route


def test_vertex_route_is_the_approved_eu_opus_route() -> None:
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    assert route.model_name == "claude-opus-5-5"
    assert route.region == "eu"
    assert route.provider == "anthropic-vertex"


def test_unknown_routes_are_refused() -> None:
    with pytest.raises(UnapprovedRouteError, match="not registered"):
        require_route("openai-us", DataClass.PATIENT_DERIVED)


def test_live_model_needs_a_project() -> None:
    route = require_route("vertex-eu-opus", DataClass.PATIENT_DERIVED)
    with pytest.raises(ValueError, match="FRONTDESK_GCP_PROJECT"):
        build_live_model(route, gcp_project=None)


def test_the_offline_route_has_no_live_model() -> None:
    route = require_route("offline-demo", DataClass.PATIENT_DERIVED)
    with pytest.raises(ValueError, match="no live model"):
        build_live_model(route, gcp_project="p")


def test_live_settings_use_medium_effort_without_sampling() -> None:
    settings = live_model_settings(max_tokens=800)
    assert settings["anthropic_effort"] == "medium"
    assert settings["max_tokens"] == 800
    assert "temperature" not in settings
