"""The Haiku route on Anthropic's own API: approved for both data classes (Albi, 2026-11-07), pinned
to api.anthropic.com, deaf to the environment, no SDK retries. Nothing here sends a request."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic_ai.models.anthropic import AnthropicModel

from frontdesk.model_gateway.access import RouteAccessError, check_anthropic_access
from frontdesk.model_gateway.factory import (
    ANTHROPIC_API_HOST,
    build_live_model,
    endpoint_name,
    live_model_settings,
    require_pinned,
)
from frontdesk.model_gateway.routes import CHAT_PROVIDERS, DataClass, require_route

HAIKU = "anthropic-haiku-5-5"
KEY = "sk-ant-test-not-a-key"


def test_the_haiku_route_is_approved_for_both_data_classes() -> None:
    for data_class in DataClass:
        route = require_route(HAIKU, data_class)
        assert (route.provider, route.model_name) == ("anthropic", "claude-haiku-5-5")
    assert route.fallback is None
    assert route.second_try is None
    assert "anthropic" in CHAT_PROVIDERS


def test_the_haiku_model_talks_to_the_anthropic_api_with_the_key_it_is_given() -> None:
    model = build_live_model(
        require_route(HAIKU, DataClass.EVALUATION), gcp_project=None, api_key=KEY
    )
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-haiku-5-5"
    client = model.client
    assert isinstance(client, AsyncAnthropic)
    assert str(client.base_url).rstrip("/") == ANTHROPIC_API_HOST
    assert client.api_key == KEY
    # One attempt: the engine retries and meters each one (frontdesk.model_gateway.metering).
    assert client.max_retries == 0


def test_the_haiku_model_needs_a_key() -> None:
    route = require_route(HAIKU, DataClass.EVALUATION)
    for missing in (None, "", "  "):
        with pytest.raises(ValueError, match="frontdesk-anthropic-api-key"):
            build_live_model(route, gcp_project=None, api_key=missing)


def test_the_environment_moves_neither_host_nor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://models.stray.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "someone-elses-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.stray.example:3128")
    route = require_route(HAIKU, DataClass.EVALUATION)
    client = build_live_model(route, gcp_project=None, api_key=KEY).client
    assert str(client.base_url).rstrip("/") == ANTHROPIC_API_HOST
    assert client.api_key == KEY
    http = client._client
    assert isinstance(http, httpx2.AsyncClient)
    assert http.trust_env is False


def test_haiku_thinks_adaptively_at_medium_effort_without_sampling_settings() -> None:
    settings = live_model_settings(require_route(HAIKU, DataClass.EVALUATION), max_tokens=900)
    assert settings["max_tokens"] == 900
    assert settings["anthropic_effort"] == "medium"
    for refused in ("temperature", "top_p", "anthropic_thinking"):
        assert refused not in settings


def test_the_haiku_route_has_a_pinned_endpoint_and_names_it() -> None:
    route = require_route(HAIKU, DataClass.EVALUATION)
    require_pinned(route)  # no Vertex location is needed
    assert endpoint_name(route) == "the Anthropic API"
    assert endpoint_name(require_route("vertex-eu-gemini-flash-lite", DataClass.EVALUATION)) == (
        "Vertex AI eu"
    )


def _client(error: Exception | None, seen: list[dict[str, Any]]) -> AsyncAnthropic:
    async def count_tokens(**request: Any) -> object:
        seen.append(request)
        if error is not None:
            raise error
        return SimpleNamespace(input_tokens=1)

    return cast(
        AsyncAnthropic, SimpleNamespace(messages=SimpleNamespace(count_tokens=count_tokens))
    )


async def test_a_free_count_tokens_request_proves_the_haiku_route() -> None:
    seen: list[dict[str, Any]] = []
    route = require_route(HAIKU, DataClass.EVALUATION)
    await check_anthropic_access(route, api_key=KEY, client=_client(None, seen))
    assert seen == [
        {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "ping"}]}
    ]


def _status_error(kind: type[anthropic.APIStatusError], status: int) -> Exception:
    request = httpx2.Request("POST", f"{ANTHROPIC_API_HOST}/v1/messages/count_tokens")
    response = httpx2.Response(status, request=request, json={"error": {"message": "no"}})
    return kind("no", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "says"),
    [
        (lambda: _status_error(anthropic.AuthenticationError, 401), "API key"),
        (lambda: _status_error(anthropic.PermissionDeniedError, 403), "API key"),
        (lambda: _status_error(anthropic.NotFoundError, 404), "claude-haiku-5-5"),
        (lambda: _status_error(anthropic.InternalServerError, 529), "not answering"),
        (
            lambda: anthropic.APIConnectionError(
                request=httpx2.Request("POST", ANTHROPIC_API_HOST)
            ),
            "cannot be reached",
        ),
    ],
)
async def test_a_failed_proof_says_what_to_do_and_never_the_key(error: Any, says: str) -> None:
    route = require_route(HAIKU, DataClass.EVALUATION)
    with pytest.raises(RouteAccessError, match=says) as raised:
        await check_anthropic_access(route, api_key=KEY, client=_client(error(), []))
    assert KEY not in str(raised.value)
