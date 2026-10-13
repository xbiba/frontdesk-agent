"""Certificates: every live HTTP client verifies with certifi's bundle through one explicit context,
whatever SSL_CERT_FILE and SSL_CERT_DIR say, on every operating system (review of Task 1, I2).

httpx2's default context asks the operating system on macOS and reads those two variables
elsewhere, and google-genai reads them itself when it builds a client: a missing file made
`google_client` raise FileNotFoundError, an empty one SSLError. Nothing here sends a request.
"""

from __future__ import annotations

import ssl
from pathlib import Path

import certifi
import httpx2
import pytest

from frontdesk.model_gateway import factory
from frontdesk.model_gateway.factory import google_client, google_http_options, model_http_client
from frontdesk.model_gateway.routes import DataClass, require_route

GEMINI = require_route("vertex-eu-gemini-flash-lite", DataClass.EVALUATION)
EMBEDDING = require_route("vertex-ew1-gemini-embedding-001", DataClass.EVALUATION)
ROUTES = [
    pytest.param(GEMINI, id="the Gemini route on the eu multi-region"),
    pytest.param(EMBEDDING, id="an embedding route on europe-west1"),
]


def _one_stray_certificate(directory: Path) -> Path:
    """A bundle that holds one certificate, as an operator's own CA file would."""
    pem = certifi.contents().split("-----END CERTIFICATE-----")[0] + "-----END CERTIFICATE-----\n"
    path = directory / "one-ca.pem"
    path.write_text(pem, encoding="ascii")
    return path


def _bad_bundles(directory: Path) -> dict[str, dict[str, str]]:
    """Ways the environment can name certificates that are not certifi's."""
    empty = directory / "empty.pem"
    empty.write_text("", encoding="ascii")
    return {
        "a missing file": {"SSL_CERT_FILE": str(directory / "missing.pem")},
        "an empty file": {"SSL_CERT_FILE": str(empty)},
        "one certificate": {"SSL_CERT_FILE": str(_one_stray_certificate(directory))},
        "a missing directory": {"SSL_CERT_DIR": str(directory / "missing-directory")},
        "both": {
            "SSL_CERT_FILE": str(directory / "missing.pem"),
            "SSL_CERT_DIR": str(directory / "missing-directory"),
        },
    }


BUNDLES = ["a missing file", "an empty file", "one certificate", "a missing directory", "both"]


def _authorities(context: ssl.SSLContext) -> int:
    return context.cert_store_stats()["x509_ca"]


def _certifi_authorities() -> int:
    return _authorities(ssl.create_default_context(cafile=certifi.where()))


def _pool_contexts(client: httpx2.Client | httpx2.AsyncClient) -> list[ssl.SSLContext]:
    """The context a client's connection pool will open its connections with."""
    transport = client._transport
    assert isinstance(transport, httpx2.HTTPTransport | httpx2.AsyncHTTPTransport)
    return [transport._pool._ssl_context]


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("route", ROUTES)
def test_the_google_client_builds_whatever_the_certificate_variables_say(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: object, bundle: str
) -> None:
    for name, value in _bad_bundles(tmp_path)[bundle].items():
        monkeypatch.setenv(name, value)
    client = google_client(route, gcp_project="clinica-mcu")  # type: ignore[arg-type]
    assert client._api_client.vertexai is True


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("route", ROUTES)
def test_every_google_client_verifies_with_the_one_context_of_certifi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: object, bundle: str
) -> None:
    for name, value in _bad_bundles(tmp_path)[bundle].items():
        monkeypatch.setenv(name, value)
    api = google_client(route, gcp_project="clinica-mcu")._api_client  # type: ignore[arg-type]
    context = factory.tls_context()
    options = api._http_options
    assert (options.client_args or {}).get("verify") is context
    assert (options.async_client_args or {}).get("verify") is context
    assert (options.async_client_args or {}).get("ssl") is context  # the websocket path reads it
    assert api._websocket_ssl_ctx["ssl"] is context
    assert [*_pool_contexts(api._httpx_client), *_pool_contexts(api._async_httpx_client)] == [
        context,
        context,
    ]
    assert _authorities(context) == _certifi_authorities() > 100
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_the_two_model_clients_use_the_same_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    context = factory.tls_context()
    assert _pool_contexts(model_http_client()) == [context]
    assert _pool_contexts(factory.model_http_client_sync()) == [context]
    assert model_http_client().trust_env is False
    assert factory.model_http_client_sync().trust_env is False


def test_the_context_is_built_once_and_from_the_bundle_of_certifi() -> None:
    assert factory.tls_context() is factory.tls_context()
    assert _authorities(factory.tls_context()) == _certifi_authorities()


def test_the_sdk_options_carry_the_context_for_the_sync_and_the_async_client() -> None:
    options = google_http_options("europe-west1")
    context = factory.tls_context()
    assert options.client_args == {"verify": context}
    assert options.async_client_args == {"verify": context, "ssl": context}
    # The clients themselves are the ones that ignore the environment.
    assert isinstance(options.httpx_client, httpx2.Client)
    assert isinstance(options.httpx_async_client, httpx2.AsyncClient)


def test_each_route_has_its_own_request_timeout() -> None:
    timeout = {
        route: require_route(route, DataClass.EVALUATION).request_timeout_seconds
        for route in ("vertex-eu-gemini-flash", "vertex-eu-gemini-flash-lite")
    }
    flash = google_http_options("eu", timeout["vertex-eu-gemini-flash"])
    lite = google_http_options("eu", timeout["vertex-eu-gemini-flash-lite"])
    assert (flash.timeout, lite.timeout) == (25_000, 30_000)  # milliseconds
