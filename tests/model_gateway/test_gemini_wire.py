"""What the Gemini route puts on the wire: the real stack (pydantic-ai, google-genai and the client
`build_live_model` builds) answered by a transport inside this process. Nothing leaves it: the
fake credentials need no lookup, and every socket is refused while these tests run."""

from __future__ import annotations

import json
import socket
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import httpx2
import pytest
from google.oauth2.credentials import Credentials
from pydantic_ai import models
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from sqlalchemy.orm import sessionmaker

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.classifier import ClassifierDeps, build_classifier
from frontdesk.agent.deps import ResponderDeps
from frontdesk.agent.responder import build_responder
from frontdesk.agent.tools import ToolServices, build_toolset
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.routes import DataClass, require_route
from frontdesk.tenancy.pack import Profile

GEMINI = "vertex-eu-gemini-flash-lite"
PROJECT = "demo-clinic"
EU_HOST = "aiplatform.eu.rep.googleapis.com"
PATH = (
    f"/v1/projects/{PROJECT}/locations/eu/publishers/google/models/"
    "gemini-3.1-flash-lite:generateContent"
)
# What would move the request elsewhere if the client read it.
STRAY_ENVIRONMENT = {
    "GOOGLE_VERTEX_BASE_URL": "https://models.stray.example/",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "GOOGLE_CLOUD_PROJECT": "someone-elses-project",
    "GOOGLE_API_KEY": "not-a-real-key",
}
SAMPLING = {"temperature", "topp", "topk", "seed"}

Run = Callable[[Model, ModelSettings], Awaitable[object]]


def _keys_alike(mapping: dict[str, Any]) -> dict[str, Any]:
    """The keys without underscores and in lower case: the SDK writes some keys in snake case and
    others in camel case, and the server takes either."""
    return {key.replace("_", "").lower(): value for key, value in mapping.items()}


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a wire test opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> tuple[Model, list[httpx2.Request]]:
    """The Gemini model of the route, whose requests are recorded and then refused."""
    for name, value in STRAY_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    # Only for this test: tests forbid model requests (conftest), and only the transport answers.
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    sent: list[httpx2.Request] = []

    def record_and_refuse(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        error = {"code": 400, "message": "the wire test stops here", "status": "INVALID_ARGUMENT"}
        return httpx2.Response(400, json={"error": error})

    model = build_live_model(require_route(GEMINI, DataClass.EVALUATION), gcp_project=PROJECT)
    api = model.client._api_client
    api._async_httpx_client._transport = httpx2.MockTransport(record_and_refuse)
    api._credentials = Credentials(token="fake-token")  # valid: no refresh, no credential lookup
    return model, sent


async def _classify(model: Model, settings: ModelSettings) -> object:
    classifier = build_classifier(
        model, Profile.RECEPTIONIST, model_settings=settings, instructions="Classify."
    )
    deps = ClassifierDeps(business_name="Clinica", language="it")
    return await classifier.run("Quanto costa la fisioterapia?", deps=deps)


async def _respond(model: Model, settings: ModelSettings) -> object:
    services = ToolServices(
        session_factory=sessionmaker(), embedder=HashEmbedder(), today=lambda: date(2026, 10, 10)
    )
    responder = build_responder(
        model,
        Profile.RECEPTIONIST,
        build_toolset(Profile.RECEPTIONIST, services),
        model_settings=settings,
        instructions="Reply.",
    )
    fields = ValidatedFields(category="administrative", intent="price", language="it")
    deps = ResponderDeps(
        business_id="demo-clinic", business_name="Clinica", language="it", fields=fields
    )
    return await responder.run("<request/>", deps=deps)


RUNS: dict[str, Run] = {"classifier": _classify, "responder": _respond}


@pytest.mark.parametrize("agent", ["classifier", "responder"])
async def test_each_agent_sends_minimal_thinking_and_its_own_cap_to_the_eu_host(
    gemini: tuple[Model, list[httpx2.Request]], agent: str
) -> None:
    model, sent = gemini
    route = require_route(GEMINI, DataClass.EVALUATION)
    cap = getattr(route.output_limits, agent)
    settings = live_model_settings(route, max_tokens=cap)

    with pytest.raises(ModelHTTPError):  # the transport answers 400 once it has recorded
        await RUNS[agent](model, settings)

    assert len(sent) == 1  # a 400 is not retried
    request = sent[0]
    assert request.method == "POST"
    assert (request.url.scheme, request.url.host) == ("https", EU_HOST)
    assert request.url.path == PATH
    body = _keys_alike(json.loads(request.content))
    generation = _keys_alike(body["generationconfig"])
    assert generation["maxoutputtokens"] == cap
    assert _keys_alike(generation["thinkingconfig"])["thinkinglevel"] == "MINIMAL"
    assert {*body, *generation}.isdisjoint(SAMPLING)
