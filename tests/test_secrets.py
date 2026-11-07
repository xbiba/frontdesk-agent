"""Secrets (Albi, 2026-10-13, before Instagram): in production cal.com's key and Instagram's come
from Google Secret Manager, never from a file; in development, from the environment."""

import base64
from pathlib import Path

import httpx2
import pytest
from pydantic import SecretStr

from frontdesk.secrets import SecretUnavailableError, read_secret
from frontdesk.settings import SecretsBackend, Settings


def _settings(**fields: object) -> Settings:
    return Settings(business_pack=Path("businesses/demo-clinic"), **fields)  # type: ignore[arg-type]


def _client(answer: httpx2.Response, seen: list[httpx2.Request]) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return answer

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_in_development_a_secret_comes_from_the_environment() -> None:
    settings = _settings(calcom_api_key=SecretStr("cal_dev"))
    assert read_secret(settings, "calcom_api_key") == "cal_dev"


def test_in_production_a_secret_comes_from_secret_manager() -> None:
    seen: list[httpx2.Request] = []
    data = base64.b64encode(b"cal_prod\n").decode()
    client = _client(httpx2.Response(200, json={"payload": {"data": data}}), seen)
    settings = _settings(secrets_backend=SecretsBackend.GCP, gcp_project="demo-clinic")
    value = read_secret(settings, "calcom_api_key", token=lambda: "t0ken", client=client)
    assert value == "cal_prod"
    (request,) = seen
    assert request.url.path == (
        "/v1/projects/demo-clinic/secrets/frontdesk-calcom-api-key/versions/latest:access"
    )
    assert request.headers["authorization"] == "Bearer t0ken"


def test_with_secret_manager_a_key_in_the_environment_is_refused() -> None:
    # Never in a file: an env file holding the key is a mistake, not a fallback.
    settings = _settings(
        secrets_backend=SecretsBackend.GCP,
        gcp_project="demo-clinic",
        calcom_api_key=SecretStr("cal_in_a_file"),
    )
    with pytest.raises(ValueError, match="never in the environment"):
        read_secret(
            settings, "calcom_api_key", token=lambda: "t", client=_client(httpx2.Response(200), [])
        )


def test_a_secret_that_cannot_be_read_names_itself_and_never_a_value() -> None:
    settings = _settings(secrets_backend=SecretsBackend.GCP, gcp_project="demo-clinic")
    client = _client(httpx2.Response(403, json={"error": "denied"}), [])
    with pytest.raises(SecretUnavailableError, match="frontdesk-calcom-api-key: status 403"):
        read_secret(settings, "calcom_api_key", token=lambda: "t0ken", client=client)


def test_in_development_the_anthropic_key_comes_from_anthropic_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dev")
    assert read_secret(_settings(), "anthropic_api_key") == "sk-ant-dev"


def test_in_production_the_anthropic_key_is_frontdesk_anthropic_api_key() -> None:
    seen: list[httpx2.Request] = []
    data = base64.b64encode(b"sk-ant-prod").decode()
    client = _client(httpx2.Response(200, json={"payload": {"data": data}}), seen)
    settings = _settings(secrets_backend=SecretsBackend.GCP, gcp_project="demo-clinic")
    value = read_secret(settings, "anthropic_api_key", token=lambda: "t0ken", client=client)
    assert value == "sk-ant-prod"
    assert seen[0].url.path.endswith("/secrets/frontdesk-anthropic-api-key/versions/latest:access")


def test_the_anthropic_key_is_never_shown_with_the_settings() -> None:
    settings = _settings(anthropic_api_key=SecretStr("sk-ant-hidden"))
    assert "sk-ant-hidden" not in repr(settings)
