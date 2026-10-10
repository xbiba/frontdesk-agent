"""The embedder settles every attempt of a text on its own (M3 design §6, review of Task 6).

The SDK makes one attempt (`GOOGLE_RETRIES`), so the embedder does the retrying, as the meter does
for a chat request: before each attempt the budget must have room for the text's reserve, and
afterwards the attempt is recorded at what Vertex AI reported, at nothing for an answer of an error
status or a connection that was never made, and at the whole reserve for anything that was sent
without an answer. The table follows the one of `test_metering_attempts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest
from google.auth.exceptions import RefreshError
from google.genai import errors as genai_errors

from frontdesk.model_gateway.access import RouteAccessError
from frontdesk.model_gateway.costs import Budget, BudgetError, BudgetExhaustedError
from frontdesk.model_gateway.embeddings import EmbeddingRateLimitedError, VertexEmbedder
from frontdesk.model_gateway.routes import DataClass, require_route
from tests.model_gateway.attempt_helpers import (
    broken,
    connect_timeout,
    ledger,
    read_timeout,
    refused,
    reset,
    write_timeout,
)
from tests.model_gateway.budget_helpers import write_budget
from tests.model_gateway.embedding_helpers import (
    ScriptedEmbeddings,
    Step,
    Waits,
    answer,
    no_embedding,
    status,
)

ROUTE = require_route("vertex-ew1-gemini-embedding-001", DataClass.EVALUATION)
TEXT = "\N{LATIN SMALL LETTER E WITH DIAERESIS}" * 10  # ten characters, twenty bytes
RESERVE = 20 * 0.5 / 1_000  # a text is reserved at its bytes: 0.01 USD
USED = 1_000 * 0.5 / 1_000  # what `answer()` reports: 1,000 billed characters, 0.5 USD
EMBEDDING, UNCONFIRMED = "embedding", "embedding-unconfirmed"
BACKOFF = (2.0, 4.0, 8.0)


@dataclass(frozen=True)
class Row:
    script: tuple[Step, ...]  # how each attempt ends, in order
    ledger: tuple[tuple[str, float], ...]  # the lines the attempts leave: kind and cost
    waits: tuple[float, ...] = ()
    raises: type[BaseException] | None = None  # what the text ends with, if it failed
    says: str = ""  # a part of the one sentence it says, for an explained failure
    cause: type[BaseException] | None = None  # what that sentence was made from


def unconfirmed(count: int) -> tuple[tuple[str, float], ...]:
    return ((UNCONFIRMED, RESERVE),) * count


TABLE: dict[str, Row] = {
    "answered": Row((answer(),), ((EMBEDDING, USED),)),
    # Sent, and nothing came back: each attempt may have been billed in full.
    "four read timeouts": Row(
        (read_timeout,) * 4,
        unconfirmed(4),
        BACKOFF,
        RouteAccessError,
        "Vertex AI cannot be reached (ReadTimeout)",
        httpx2.ReadTimeout,
    ),
    "four write timeouts": Row(
        (write_timeout,) * 4,
        unconfirmed(4),
        BACKOFF,
        RouteAccessError,
        "Vertex AI cannot be reached (WriteTimeout)",
        httpx2.WriteTimeout,
    ),
    "two timeouts then an answer": Row(
        (read_timeout, read_timeout, answer()),
        (*unconfirmed(2), (EMBEDDING, USED)),
        (2.0, 4.0),
    ),
    # The attempts that cost nothing do not hide the one that may have cost everything.
    "a timeout then 429 three times": Row(
        (read_timeout, status(429), status(429), status(429)),
        unconfirmed(1),
        BACKOFF,
        EmbeddingRateLimitedError,
        "rate limited",
        genai_errors.ClientError,
    ),
    "a timeout then 503 three times": Row(
        (read_timeout, status(503), status(503), status(503)),
        unconfirmed(1),
        BACKOFF,
        RouteAccessError,
        "Vertex AI is not answering (503 SCRIPTED)",
        genai_errors.ServerError,
    ),
    "a timeout then a refused connection three times": Row(
        (read_timeout, refused, refused, refused),
        unconfirmed(1),
        BACKOFF,
        RouteAccessError,
        "Vertex AI cannot be reached (ConnectError)",
        httpx2.ConnectError,
    ),
    # Refused by status, or never made: nothing was billed.
    "429 four times": Row(
        (status(429),) * 4,
        (),
        BACKOFF,
        EmbeddingRateLimitedError,
        "answered 429 after 4 attempts",
        genai_errors.ClientError,
    ),
    "503 four times": Row(
        (status(503),) * 4,
        (),
        BACKOFF,
        RouteAccessError,
        "Vertex AI is not answering (503 SCRIPTED): try again later",
        genai_errors.ServerError,
    ),
    "refused connection four times": Row(
        (refused,) * 4,
        (),
        BACKOFF,
        RouteAccessError,
        "Vertex AI cannot be reached (ConnectError): check the network connection",
        httpx2.ConnectError,
    ),
    "connect timeout four times": Row(
        (connect_timeout,) * 4,
        (),
        BACKOFF,
        RouteAccessError,
        "Vertex AI cannot be reached (ConnectTimeout)",
        httpx2.ConnectTimeout,
    ),
    "503 then a good answer": Row((status(503), answer()), ((EMBEDDING, USED),), (2.0,)),
    "429 then a good answer": Row(
        (status(429), status(429), answer()), ((EMBEDDING, USED),), (2.0, 4.0)
    ),
    "a refused connection then a good answer": Row(
        (refused, answer()), ((EMBEDDING, USED),), (2.0,)
    ),
    "every status the engine retries": Row(
        (status(500), status(502), status(504), answer()), ((EMBEDDING, USED),), BACKOFF
    ),
    # Not retried: the SDK did not retry them either.
    "a 400": Row(
        (status(400),),
        (),
        (),
        RouteAccessError,
        "Vertex AI answered 400 SCRIPTED",
        genai_errors.ClientError,
    ),
    "a 403": Row(
        (status(403),),
        (),
        (),
        RouteAccessError,
        "Vertex AI refused project 'p' (SCRIPTED: scripted). Check that the API is enabled",
        genai_errors.ClientError,
    ),
    "a 404": Row(
        (status(404),),
        (),
        (),
        RouteAccessError,
        "model 'gemini-embedding-001' is not served at "
        "https://europe-west1-aiplatform.googleapis.com",
        genai_errors.ClientError,
    ),
    "a 408": Row(
        (status(408),),
        (),
        (),
        RouteAccessError,
        "Vertex AI answered 408 SCRIPTED",
        genai_errors.ClientError,
    ),
    "credentials that cannot be refreshed": Row(
        (lambda: RefreshError("expired"),),
        (),
        (),
        RouteAccessError,
        "Google credentials are missing or expired: run `gcloud auth application-default login`",
        RefreshError,
    ),
    "a connection reset": Row(
        (reset,),
        unconfirmed(1),
        (),
        RouteAccessError,
        "Vertex AI cannot be reached (ReadError)",
        httpx2.ReadError,
    ),
    "a broken protocol": Row(
        (broken,),
        unconfirmed(1),
        (),
        RouteAccessError,
        "Vertex AI cannot be reached (RemoteProtocolError)",
        httpx2.RemoteProtocolError,
    ),
    "Ctrl-C while the request is away": Row(
        (lambda: KeyboardInterrupt(),), unconfirmed(1), (), KeyboardInterrupt
    ),
    # A 200 that reports nothing may still have been billed for the whole text.
    "a 200 without usage": Row((answer(billed=None),), unconfirmed(1)),
    # Billed as reported (at least its bytes), though no vector came back: not sent again.
    "a 200 without a vector": Row((no_embedding,), ((EMBEDDING, RESERVE),), (), ValueError),
    # Billed as reported; a vector of another size would only fail later, in the database.
    "a vector of another size": Row((answer(dimension=768),), ((EMBEDDING, USED),), (), ValueError),
}


def _embedder(budget: Budget, scripted: ScriptedEmbeddings, waits: Waits) -> VertexEmbedder:
    return VertexEmbedder(
        ROUTE,
        gcp_project="p",
        task="RETRIEVAL_DOCUMENT",
        budget=budget,
        run_id="run-1",
        client=scripted.client,
        sleep=waits,
    )


@pytest.mark.parametrize("row", [pytest.param(row, id=name) for name, row in TABLE.items()])
def test_each_attempt_is_settled_on_its_own(tmp_path: Path, row: Row) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    waits, scripted = Waits(), ScriptedEmbeddings(*row.script)
    embedder = _embedder(budget, scripted, waits)
    if row.raises is None:
        embedder.embed([TEXT])
    else:
        with pytest.raises(row.raises) as caught:
            embedder.embed([TEXT])
        assert row.says in str(caught.value)
        if row.cause is not None:  # the sentence keeps what it was made from, for whoever debugs it
            assert isinstance(caught.value.__cause__, row.cause)
    assert len(scripted.requests) == len(row.script)  # no attempt more, none fewer
    assert ledger(budget) == [(kind, pytest.approx(usd)) for kind, usd in row.ledger]
    assert waits.waits == list(row.waits)
    assert {tuple(request["contents"]) for request in scripted.requests} <= {(TEXT,)}


def test_an_attempt_the_budget_cannot_cover_is_not_sent(tmp_path: Path) -> None:
    # Room for one reserve: the first attempt times out and is recorded at its reserve, so the
    # second no longer fits and is not sent, with the error of the refusal.
    budget = write_budget(tmp_path, cap_eur=RESERVE * 1.5, usd_per_eur=1.0)
    waits, scripted = Waits(), ScriptedEmbeddings(read_timeout, answer())
    with pytest.raises(BudgetExhaustedError):
        _embedder(budget, scripted, waits).embed([TEXT])
    assert len(scripted.requests) == 1
    assert ledger(budget) == [(UNCONFIRMED, pytest.approx(RESERVE))]
    assert budget.spent_usd() <= budget.cap_usd


def test_the_wait_starts_again_for_each_text(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, usd_per_eur=1.0)
    waits = Waits()
    scripted = ScriptedEmbeddings(status(503), answer(), status(503), answer())
    _embedder(budget, scripted, waits).embed(["primo", "secondo"])
    assert [request["contents"] for request in scripted.requests] == [
        ["primo"],
        ["primo"],
        ["secondo"],
        ["secondo"],
    ]
    assert waits.waits == [2.0, 2.0]


def test_a_text_that_cannot_be_recorded_is_not_sent_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first attempt timed out and its line cannot be written: another attempt would be paid
    # and uncounted, so the experiment stops at once, with the ledger named: not "cannot be
    # reached", which is what an OSError would be taken for.
    def cannot_write(*args: object, **kwargs: object) -> None:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Budget, "record", cannot_write)
    budget = write_budget(tmp_path)
    waits, scripted = Waits(), ScriptedEmbeddings(read_timeout, answer())
    with pytest.raises(BudgetError) as caught:
        _embedder(budget, scripted, waits).embed([TEXT])
    assert (
        str(caught.value) == f"{budget.ledger_path}: cannot be appended to (Read-only file system)"
    )
    assert isinstance(caught.value.__cause__, OSError)
    assert len(scripted.requests) == 1
    assert waits.waits == []


def test_a_ledger_that_cannot_be_read_is_named_and_no_text_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_read(*args: object, **kwargs: object) -> float:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Budget, "spent_usd", cannot_read)
    budget = write_budget(tmp_path)
    scripted = ScriptedEmbeddings(answer())
    with pytest.raises(BudgetError) as caught:
        _embedder(budget, scripted, Waits()).embed([TEXT])
    assert str(caught.value) == f"{budget.ledger_path}: cannot be read (Permission denied)"
    assert scripted.requests == []


def test_a_ledger_that_cannot_be_read_stops_the_text_before_it_is_sent(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("not a spend entry\n", encoding="utf-8")
    scripted = ScriptedEmbeddings(answer())
    with pytest.raises(BudgetError, match="line 1 is not a spend entry"):
        _embedder(budget, scripted, Waits()).embed([TEXT])
    assert scripted.requests == []
