"""What the embedder puts on the wire: the real google-genai client that `google_client` builds,
answered by a transport inside this process. Nothing leaves it: the fake credentials need no
lookup, and every socket is refused while these tests run. It shows that the statuses and the
network failures the embedder retries are the ones the real client raises."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx2
import pytest
from google.genai import errors as genai_errors
from google.oauth2.credentials import Credentials

from frontdesk.model_gateway.access import RouteAccessError
from frontdesk.model_gateway.costs import Budget
from frontdesk.model_gateway.embeddings import EmbeddingRateLimitedError, VertexEmbedder
from frontdesk.model_gateway.routes import DataClass, require_route
from tests.model_gateway.attempt_helpers import ledger
from tests.model_gateway.budget_helpers import write_budget
from tests.model_gateway.embedding_helpers import Waits

ROUTE = require_route("vertex-ew1-gemini-embedding-001", DataClass.EVALUATION)
PROJECT = "demo-clinic"
HOST = "europe-west1-aiplatform.googleapis.com"
PATH = (
    f"/v1/projects/{PROJECT}/locations/europe-west1/publishers/google/models/"
    "gemini-embedding-001:predict"
)
# What would move the request elsewhere, or change how it is verified, if the client read it.
STRAY_ENVIRONMENT = {
    "GOOGLE_VERTEX_BASE_URL": "https://models.stray.example/",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "GOOGLE_CLOUD_PROJECT": "someone-elses-project",
    "GOOGLE_API_KEY": "not-a-real-key",
    "HTTPS_PROXY": "http://proxy.stray.example:3128",
    "ALL_PROXY": "http://proxy.stray.example:3128",
    "SSL_CERT_FILE": "/nonexistent/stray-ca.pem",
    "SSL_CERT_DIR": "/nonexistent/stray-certs",
    "GOOGLE_API_USE_CLIENT_CERTIFICATE": "true",  # the SDK's mutual-TLS switches: the clients
    "GOOGLE_API_USE_MTLS_ENDPOINT": "always",  # it is given are not theirs to change
}
Outcome = Callable[[httpx2.Request], httpx2.Response]
Wire = tuple[VertexEmbedder, list[httpx2.Request], Waits, Budget]


def ok(billed: int = 11) -> Outcome:
    def respond(request: httpx2.Request) -> httpx2.Response:
        embeddings = {"values": [0.25] * 3072, "statistics": {"truncated": False, "token_count": 3}}
        body = {
            "predictions": [{"embeddings": embeddings}],
            "metadata": {"billableCharacterCount": billed},
        }
        return httpx2.Response(200, json=body)

    return respond


def error(code: int) -> Outcome:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(code, json={"error": {"code": code, "status": "SCRIPTED"}})

    return respond


def read_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("no answer after the request went out", request=request)


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a wire test opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[Sequence[Outcome]], Wire]:
    """The embedder of the route on the real client, whose n-th request is answered by the n-th
    outcome, with the stray environment set and a ledger in `tmp_path`."""
    for name, value in STRAY_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    def build(outcomes: Sequence[Outcome]) -> Wire:
        budget, waits = write_budget(tmp_path, usd_per_eur=1.0), Waits()
        embedder = VertexEmbedder(
            ROUTE,
            gcp_project=PROJECT,
            task="RETRIEVAL_QUERY",
            budget=budget,
            run_id="run-1",
            sleep=waits,
        )
        sent: list[httpx2.Request] = []

        def answer(request: httpx2.Request) -> httpx2.Response:
            sent.append(request)
            return outcomes[len(sent) - 1](request)

        api = embedder._client._api_client
        api._httpx_client._transport = httpx2.MockTransport(answer)
        api._credentials = Credentials(token="fake-token")  # valid: no refresh, no lookup
        return embedder, sent, waits, budget

    return build


def test_a_text_goes_to_europe_west1_as_one_prediction(
    wire: Callable[[Sequence[Outcome]], Wire],
) -> None:
    embedder, sent, waits, budget = wire([ok(billed=11)])
    [vector] = embedder.embed(["ciao mondo"])
    assert len(vector) == 3072
    [request] = sent
    assert request.method == "POST"
    assert (request.url.scheme, request.url.host, request.url.path) == ("https", HOST, PATH)
    assert request.headers["authorization"] == "Bearer fake-token"
    assert json.loads(request.content) == {
        "instances": [{"content": "ciao mondo", "task_type": "RETRIEVAL_QUERY"}],
        "parameters": {"autoTruncate": False},
    }
    # Vertex AI counted 11 characters of a text of 10 bytes: the larger is billed.
    assert ledger(budget) == [("embedding", pytest.approx(11 * 0.5 / 1_000))]
    assert waits.waits == []


def test_the_client_the_embedder_builds_retries_what_the_real_sdk_raises(
    wire: Callable[[Sequence[Outcome]], Wire],
) -> None:
    # A 503 is an answer of an error status (unbilled); a read timeout went out and never came back
    # (recorded at the reserve of the 10 bytes); the third attempt is answered and billed.
    embedder, sent, waits, budget = wire([error(503), read_timeout, ok(billed=11)])
    embedder.embed(["ciao mondo"])
    assert len(sent) == 3
    assert waits.waits == [2.0, 4.0]
    assert ledger(budget) == [
        ("embedding-unconfirmed", pytest.approx(10 * 0.5 / 1_000)),
        ("embedding", pytest.approx(11 * 0.5 / 1_000)),
    ]


def test_a_429_that_outlives_the_attempts_stops_the_experiment_unbilled(
    wire: Callable[[Sequence[Outcome]], Wire],
) -> None:
    embedder, sent, waits, budget = wire([error(429)] * 4)
    with pytest.raises(EmbeddingRateLimitedError):
        embedder.embed(["ciao mondo"])
    assert len(sent) == 4
    assert waits.waits == [2.0, 4.0, 8.0]
    assert ledger(budget) == []


def test_a_refusal_the_client_raises_as_a_client_error_is_not_retried(
    wire: Callable[[Sequence[Outcome]], Wire],
) -> None:
    embedder, sent, waits, budget = wire([error(403)])
    with pytest.raises(RouteAccessError, match="Vertex AI refused project 'demo-clinic'") as caught:
        embedder.embed(["ciao mondo"])
    assert isinstance(caught.value.__cause__, genai_errors.ClientError)
    assert caught.value.__cause__.code == 403
    assert (len(sent), waits.waits, ledger(budget)) == (1, [], [])
