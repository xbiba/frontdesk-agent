"""What a live run checks before it spends anything (M3 design §6)."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from google.auth import exceptions as auth_errors
from google.genai import Client
from google.genai import errors as genai_errors

from frontdesk.model_gateway import access
from frontdesk.model_gateway.access import (
    ENVIRONMENT_OVERRIDES,
    RouteAccessError,
    check_route_access,
    environment_overrides,
    explain_failure,
)
from frontdesk.model_gateway.costs import BudgetError, BudgetExhaustedError
from frontdesk.model_gateway.routes import ROUTES, DataClass, require_route

ROUTE = require_route("vertex-eu-gemini-flash-lite", DataClass.EVALUATION)


def _client(error: Exception | None) -> Client:
    async def count_tokens(**request: Any) -> object:
        assert request == {"model": "gemini-3.1-flash-lite", "contents": "ping"}
        if error is not None:
            raise error
        return SimpleNamespace(total_tokens=1)

    return cast(
        Client,
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens))),
    )


def _api_error(kind: type[genai_errors.APIError], code: int, status: str) -> Exception:
    return kind(code, {"error": {"code": code, "message": "Vertex AI says no", "status": status}})


def test_proxy_and_certificate_variables_are_named_in_either_case() -> None:
    environ = {
        "https_proxy": "http://p:3128",
        "SSL_CERT_FILE": "/ca.pem",
        "PATH": "/bin",
        "ALL_PROXY": " ",
    }
    assert environment_overrides(environ) == ("https_proxy", "SSL_CERT_FILE")
    assert environment_overrides({"NO_PROXY": "localhost"}) == ()


async def test_a_free_count_tokens_request_proves_the_route() -> None:
    await check_route_access(ROUTE, gcp_project="demo-clinic", client=_client(None))


@pytest.mark.parametrize(
    ("error", "advice"),
    [
        (auth_errors.DefaultCredentialsError("no ADC"), "gcloud auth application-default login"),
        (auth_errors.RefreshError("expired"), "gcloud auth application-default login"),
        (
            _api_error(genai_errors.ClientError, 403, "PERMISSION_DENIED"),
            "gcloud services enable aiplatform.googleapis.com --project demo-clinic",
        ),
        (
            _api_error(genai_errors.ClientError, 404, "NOT_FOUND"),
            "'gemini-3.1-flash-lite' is not served at https://aiplatform.eu.rep.googleapis.com",
        ),
        (_api_error(genai_errors.ServerError, 503, "UNAVAILABLE"), "try again later"),
    ],
)
async def test_each_failure_says_what_to_do(error: Exception, advice: str) -> None:
    with pytest.raises(RouteAccessError, match=re.escape(advice)):
        await check_route_access(ROUTE, gcp_project="demo-clinic", client=_client(error))


# --- Beyond the brief (Task 8 rulings) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "REQUESTS_CA_BUNDLE",
        "GOOGLE_CLOUD_QUOTA_PROJECT",
        "HTTP_PROXY",
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
    ],
)
def test_every_variable_that_moves_a_request_or_its_billing_is_an_override(name: str) -> None:
    assert name in ENVIRONMENT_OVERRIDES
    assert environment_overrides({name: "x"}) == (name,)
    assert environment_overrides({name.lower(): "x"}) == (name.lower(),)  # named as it is set
    assert environment_overrides({name: " \t"}) == ()  # set to nothing: nothing is overridden


def test_a_variable_is_named_as_the_shell_sets_it_so_that_unsetting_the_name_works() -> None:
    environ = {
        "https_proxy": "http://p:3128",
        "HTTPS_PROXY": "http://q:3128",
        "Http_Proxy": "http://r:3128",
        "SSL_CERT_FILE": "/ca.pem",
        "PATH": "/bin",
    }
    named = environment_overrides(environ)
    # Each once, sorted by name without regard to case (then as spelled).
    assert named == ("HTTPS_PROXY", "https_proxy", "Http_Proxy", "SSL_CERT_FILE")
    for name in named:
        del environ[name]  # `unset NAME`, as the line advises
    assert environment_overrides(environ) == ()
    assert environ == {"PATH": "/bin"}


def test_the_overrides_are_found_among_the_variables_of_a_real_environment() -> None:
    environ = {"PATH": "/bin", "HOME": "/home/me", "GOOGLE_CLOUD_QUOTA_PROJECT": "other-project"}
    assert environment_overrides(environ) == ("GOOGLE_CLOUD_QUOTA_PROJECT",)
    assert environment_overrides({}) == ()


@pytest.mark.parametrize(
    ("error", "advice"),
    [
        pytest.param(
            _api_error(genai_errors.ClientError, 401, "UNAUTHENTICATED"),
            "gcloud services enable aiplatform.googleapis.com --project demo-clinic",
            id="401",
        ),
        pytest.param(
            _api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"),
            "rate limiting project 'demo-clinic' (429 RESOURCE_EXHAUSTED): try again later",
            id="429",
        ),
        pytest.param(
            _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT"),
            "Vertex AI answered 400 INVALID_ARGUMENT",
            id="400",
        ),
        pytest.param(
            auth_errors.TransportError("the token endpoint did not answer"),
            "could not sign the request (TransportError)",
            id="transport-error-while-signing",
        ),
        pytest.param(
            httpx2.ConnectError("no route to host"),
            "Vertex AI cannot be reached (ConnectError): check the network",
            id="no-network",
        ),
        pytest.param(
            httpx2.ReadTimeout("no answer"),
            "Vertex AI cannot be reached (ReadTimeout): check the network",
            id="no-answer",
        ),
        pytest.param(
            TimeoutError("timed out"),
            "Vertex AI cannot be reached (TimeoutError): check the network",
            id="os-timeout",
        ),
    ],
)
async def test_the_other_failures_are_one_message_too(error: Exception, advice: str) -> None:
    with pytest.raises(RouteAccessError) as caught:
        await check_route_access(ROUTE, gcp_project="demo-clinic", client=_client(error))
    assert advice in str(caught.value)
    assert caught.value.__cause__ is error  # the cause stays attached, for whoever debugs it


async def test_a_message_never_carries_what_google_said_about_the_network() -> None:
    # httpx messages can hold URLs and hosts: only the type of a network error is shown.
    error = httpx2.ConnectError("could not reach https://aiplatform.eu.rep.googleapis.com/secret")
    with pytest.raises(RouteAccessError) as caught:
        await check_route_access(ROUTE, gcp_project="demo-clinic", client=_client(error))
    assert "secret" not in str(caught.value)


async def test_a_bug_in_the_check_is_not_dressed_up_as_a_network_problem() -> None:
    with pytest.raises(TypeError):
        await check_route_access(ROUTE, gcp_project="demo-clinic", client=_client(TypeError("bug")))


class _Closing:
    """A stand-in for the pinned client that records how it was built and whether it was closed."""

    def __init__(self) -> None:
        self.built: list[tuple[str, str]] = []
        self.closed = 0
        self.fail: Exception | None = None

    def client(self, route: Any, *, gcp_project: str) -> Client:
        self.built.append((route.route_id, gcp_project))

        async def count_tokens(**request: Any) -> object:
            if self.fail is not None:
                raise self.fail
            return SimpleNamespace(total_tokens=1)

        async def aclose() -> None:
            self.closed += 1

        models = SimpleNamespace(count_tokens=count_tokens)
        return cast(Client, SimpleNamespace(aio=SimpleNamespace(models=models, aclose=aclose)))


async def test_without_a_client_the_request_goes_through_the_pinned_one_and_it_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = _Closing()
    monkeypatch.setattr(access, "google_client", pinned.client)
    await check_route_access(ROUTE, gcp_project="demo-clinic")
    assert pinned.built == [("vertex-eu-gemini-flash-lite", "demo-clinic")]
    assert pinned.closed == 1


async def test_the_client_it_built_is_closed_when_the_request_fails_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = _Closing()
    pinned.fail = _api_error(genai_errors.ServerError, 503, "UNAVAILABLE")
    monkeypatch.setattr(access, "google_client", pinned.client)
    with pytest.raises(RouteAccessError, match="try again later"):
        await check_route_access(ROUTE, gcp_project="demo-clinic")
    assert pinned.closed == 1


async def test_a_client_it_was_given_is_left_open_for_its_owner() -> None:
    closed: list[str] = []

    async def count_tokens(**request: Any) -> object:
        return SimpleNamespace(total_tokens=1)

    async def aclose() -> None:
        closed.append("closed")

    models = SimpleNamespace(count_tokens=count_tokens)
    client = cast(Client, SimpleNamespace(aio=SimpleNamespace(models=models, aclose=aclose)))
    await check_route_access(ROUTE, gcp_project="demo-clinic", client=client)
    assert closed == []


async def test_a_client_that_cannot_be_closed_does_not_hide_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def count_tokens(**request: Any) -> object:
        return SimpleNamespace(total_tokens=1)

    async def aclose() -> None:
        raise RuntimeError("the transport is already gone")

    models = SimpleNamespace(count_tokens=count_tokens)
    broken = cast(Client, SimpleNamespace(aio=SimpleNamespace(models=models, aclose=aclose)))
    monkeypatch.setattr(access, "google_client", lambda route, *, gcp_project: broken)
    await check_route_access(
        ROUTE, gcp_project="demo-clinic"
    )  # the route works; that is the answer


@pytest.mark.parametrize(
    "route_id", ["vertex-eu-opus", "vertex-ew1-gemini-embedding-001", "offline-demo"]
)
async def test_a_route_that_is_not_a_gemini_route_cannot_be_proved_so_it_is_refused(
    route_id: str,
) -> None:
    # CountTokens on a Claude or an embedding model would fail for the wrong reason ("not served"),
    # and a run must not start on a route that nothing has proved.
    route = ROUTES[route_id]
    with pytest.raises(RouteAccessError, match=f"route '{route_id}' .* has no free access check"):
        await check_route_access(route, gcp_project="demo-clinic", client=_client(None))


# --- One explanation for every request to Vertex AI (fix round 1 of Task 9) -----------------------

FAILURES = [
    pytest.param(
        auth_errors.DefaultCredentialsError("no ADC"),
        "Google credentials are missing or expired: run `gcloud auth application-default login`, "
        "then try again",
        id="no credentials",
    ),
    pytest.param(
        auth_errors.RefreshError("expired"),
        "Google credentials are missing or expired: run `gcloud auth application-default login`, "
        "then try again",
        id="expired credentials",
    ),
    pytest.param(
        auth_errors.TransportError("the token endpoint did not answer"),
        "Google could not sign the request (TransportError): check the network, then run "
        "`gcloud auth application-default login` if it happens again",
        id="the token endpoint",
    ),
    pytest.param(
        _api_error(genai_errors.ClientError, 403, "PERMISSION_DENIED"),
        "Vertex AI refused project 'demo-clinic' (PERMISSION_DENIED: Vertex AI says no). "
        "Check that the API is enabled (`gcloud services enable aiplatform.googleapis.com "
        "--project demo-clinic`) and that your account may use it",
        id="403",
    ),
    pytest.param(
        _api_error(genai_errors.ClientError, 404, "NOT_FOUND"),
        "model 'gemini-3.1-flash-lite' is not served at https://aiplatform.eu.rep.googleapis.com",
        id="404",
    ),
    pytest.param(
        _api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"),
        "Vertex AI is rate limiting project 'demo-clinic' (429 RESOURCE_EXHAUSTED): try again later",
        id="429",
    ),
    pytest.param(
        _api_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT"),
        "Vertex AI answered 400 INVALID_ARGUMENT",
        id="400",
    ),
    pytest.param(
        _api_error(genai_errors.ServerError, 503, "UNAVAILABLE"),
        "Vertex AI is not answering (503 UNAVAILABLE): try again later",
        id="503",
    ),
    pytest.param(
        httpx2.ConnectError("no route to host"),
        "Vertex AI cannot be reached (ConnectError): check the network connection, then try again",
        id="no network",
    ),
    pytest.param(
        httpx2.ReadTimeout("no answer"),
        "Vertex AI cannot be reached (ReadTimeout): check the network connection, then try again",
        id="no answer",
    ),
    pytest.param(
        TimeoutError("timed out"),
        "Vertex AI cannot be reached (TimeoutError): check the network connection, then try again",
        id="a timeout of the socket layer",
    ),
]


@pytest.mark.parametrize(("error", "sentence"), FAILURES)
def test_a_failed_request_is_one_sentence_that_says_what_to_do(
    error: Exception, sentence: str
) -> None:
    explained = explain_failure(error, ROUTE, "demo-clinic")
    assert isinstance(explained, RouteAccessError)
    assert str(explained) == sentence


@pytest.mark.parametrize(("error", "sentence"), FAILURES)
async def test_the_access_check_raises_the_sentence_the_explanation_gives(
    error: Exception, sentence: str
) -> None:
    with pytest.raises(RouteAccessError) as caught:
        await check_route_access(ROUTE, gcp_project="demo-clinic", client=_client(error))
    assert str(caught.value) == sentence
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(TypeError("a bug"), id="a bug"),
        pytest.param(ValueError("a value"), id="a value"),
        pytest.param(BudgetError("the ledger is damaged"), id="the ledger"),
        pytest.param(BudgetExhaustedError("no room"), id="the budget"),
        pytest.param(KeyboardInterrupt(), id="Ctrl-C"),
    ],
)
def test_anything_that_is_not_a_failure_to_reach_vertex_ai_is_left_alone(
    error: BaseException,
) -> None:
    assert explain_failure(error, ROUTE, "demo-clinic") is None


def test_the_text_of_what_google_said_about_the_network_is_never_in_the_sentence() -> None:
    error = httpx2.ConnectError("could not reach https://aiplatform.eu.rep.googleapis.com/secret")
    explained = explain_failure(error, ROUTE, "demo-clinic")
    assert explained is not None
    assert "secret" not in str(explained)
