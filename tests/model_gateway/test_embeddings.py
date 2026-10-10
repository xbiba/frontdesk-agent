"""The Vertex AI embedders of the experiment: pinned, metered, one text per request."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.genai import Client
from google.genai import errors as genai_errors

from frontdesk.model_gateway.access import RouteAccessError
from frontdesk.model_gateway.costs import BudgetError, BudgetExhaustedError, load_budget
from frontdesk.model_gateway.embeddings import EmbeddingRateLimitedError, VertexEmbedder
from frontdesk.model_gateway.routes import DataClass, require_route
from tests.model_gateway.budget_helpers import write_budget
from tests.model_gateway.embedding_helpers import Waits

ROUTE = require_route("vertex-ew1-gemini-embedding-001", DataClass.EVALUATION)


def _client(
    requests: list[dict[str, Any]], *, error: Exception | None = None, billed: int | None = 1_000
) -> Client:
    def embed_content(**request: Any) -> object:
        requests.append(request)
        if error is not None:
            raise error
        values = [0.5] * 3072
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=values)],
            metadata=SimpleNamespace(billable_character_count=billed),
        )

    return cast(Client, SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))


def _embedder(
    tmp_path: Path, client: Client, cap_eur: float = 10.0, waits: Waits | None = None
) -> VertexEmbedder:
    budget = write_budget(tmp_path, cap_eur=cap_eur, usd_per_eur=1.0)
    return VertexEmbedder(
        ROUTE,
        gcp_project="p",
        task="RETRIEVAL_QUERY",
        budget=budget,
        run_id="r",
        client=client,
        sleep=waits or Waits(),  # the tests never wait
    )


def test_each_text_is_one_request_billed_by_the_characters_vertex_counts(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []
    embedder = _embedder(tmp_path, _client(requests))
    vectors = embedder.embed(["primo testo", "secondo testo"])
    assert [len(vector) for vector in vectors] == [3072, 3072]
    assert [request["contents"] for request in requests] == [["primo testo"], ["secondo testo"]]
    assert requests[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert requests[0]["config"].auto_truncate is False
    # 1,000 billed characters at 0.5 USD per thousand, twice.
    assert embedder._budget.spent_usd() == pytest.approx(1.0)
    assert (embedder.dimension, embedder.identity) == (3072, "vertex:gemini-embedding-001")


def test_a_text_is_billed_on_its_bytes_when_vertex_counts_fewer(tmp_path: Path) -> None:
    # Whether Vertex bills characters or tokens, a text's UTF-8 bytes are at least as many.
    embedder = _embedder(tmp_path, _client([], billed=4))
    embedder.embed(["\N{LATIN SMALL LETTER E WITH DIAERESIS}" * 4])  # four characters, eight bytes
    assert embedder._budget.spent_usd() == pytest.approx(8 * 0.5 / 1_000)


def test_a_text_sent_without_an_answer_is_charged_its_reserve(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path, _client([], error=TimeoutError("no answer")))
    with pytest.raises(RouteAccessError, match=r"cannot be reached \(TimeoutError\)"):
        embedder.embed(["\N{LATIN SMALL LETTER E WITH DIAERESIS}" * 10])  # twenty bytes
    assert embedder._budget.spent_usd() == pytest.approx(20 * 0.5 / 1_000)


def test_a_text_the_budget_cannot_cover_is_never_sent(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []
    embedder = _embedder(tmp_path, _client(requests), cap_eur=0.001)
    with pytest.raises(BudgetExhaustedError):
        embedder.embed(["x" * 10_000])
    assert requests == []


def test_a_rate_limit_stops_the_experiment_and_costs_nothing(tmp_path: Path) -> None:
    # The embedder retries (the SDK makes one attempt): four 429 answers in a row end it, and an
    # error status is not billed.
    error = genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
    requests: list[dict[str, Any]] = []
    waits = Waits()
    embedder = _embedder(tmp_path, _client(requests, error=error), waits=waits)
    with pytest.raises(EmbeddingRateLimitedError, match="run the experiment again later") as caught:
        embedder.embed(["x"])
    assert "after 4 attempts" in str(caught.value)
    assert caught.value.__cause__ is error
    assert len(requests) == 4
    assert waits.waits == [2.0, 4.0, 8.0]
    assert embedder._budget.spent_usd() == 0.0


def test_a_chat_route_is_no_embedder(tmp_path: Path) -> None:
    chat = require_route("vertex-eu-gemini-flash-lite", DataClass.EVALUATION)
    with pytest.raises(ValueError, match="not an embedding route"):
        VertexEmbedder(
            chat,
            gcp_project="p",
            task="RETRIEVAL_QUERY",
            budget=write_budget(tmp_path),
            run_id="r",
            client=_client([]),
        )


def test_a_price_per_token_would_undercount_and_is_refused(tmp_path: Path) -> None:
    # The texts are counted in characters and bytes: a price per million tokens would make each
    # of them cost a thousandth of what it does.
    budget = write_budget(tmp_path)
    budget.path.write_text(
        budget.path.read_text(encoding="utf-8").replace("characters_1k", "tokens_1m"),
        encoding="utf-8",
    )
    with pytest.raises(BudgetError, match="priced per tokens_1m, and the embedder counts"):
        VertexEmbedder(
            ROUTE,
            gcp_project="p",
            task="RETRIEVAL_QUERY",
            budget=load_budget(budget.path),
            run_id="r",
            client=_client([]),
        )


def test_the_embedding_client_is_pinned_to_europe_west1(monkeypatch: pytest.MonkeyPatch) -> None:
    from frontdesk.model_gateway.factory import google_client

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    api = google_client(ROUTE, gcp_project="clinica-mcu")._api_client
    assert api._http_options.base_url == "https://europe-west1-aiplatform.googleapis.com/"
    assert api.location == "europe-west1"
    assert api._httpx_client.trust_env is False
