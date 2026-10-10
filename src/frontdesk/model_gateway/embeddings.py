"""Vertex AI text embedding models for the embedder experiment (M3 design §4.5).

The candidates run on europe-west1, where Google serves them; their routes are approved for
evaluation data only. Each text is sent alone (gemini-embedding-001 takes one text per request).
The price page bills "count" per 1,000 without naming it: characters for the text models, perhaps
tokens for Gemini Embedding. A token covers at least one byte, and billable characters leave out
whitespace, so a text is reserved at its UTF-8 length in bytes and recorded at the larger of that
and the characters Vertex AI reports: an upper bound for either unit.

The SDK makes one attempt (`GOOGLE_RETRIES`), so the embedder does the retrying and settles every
attempt on its own, as `MeteredModel` does for a chat request (`frontdesk.model_gateway.metering`):

- before an attempt the budget must have room for the text's reserve, or the text is not sent;
- an attempt that was answered with usage costs the larger figure above;
- one that was answered with an error status, or that never left (no credentials, no connection),
  costs nothing;
- any other attempt that was sent (a timeout, a reset, Ctrl-C) may have been billed in full, and is
  recorded at its reserve; so is an answer that reports no usage.

429 and 5xx answers, timeouts and refused connections are tried again after 2, 4 and 8 seconds,
four attempts in all. A 429 that outlives them stops the experiment, which resumes later; any
other failure that outlives them is explained in one sentence that says what to do
(`frontdesk.model_gateway.access.explain_failure`, the sentences of the access check), because the
first embedding of a run is its first proof of the credentials, the API and the model. A ledger
that cannot be read or written is the budget's error, whatever the operating system called it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Final, Literal

import httpx2
from google.genai import Client
from google.genai import errors as genai_errors
from google.genai.types import EmbedContentConfig

from frontdesk.model_gateway.access import explain_failure
from frontdesk.model_gateway.costs import Budget, BudgetError
from frontdesk.model_gateway.factory import google_client
from frontdesk.model_gateway.metering import MAX_ATTEMPTS, NOT_SENT, RETRY_STATUSES, RETRY_WAITS
from frontdesk.model_gateway.routes import ModelRoute

EmbeddingTask = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
EMBEDDING_PROVIDER: Final = "google-vertex-embedding"
PRICE_UNIT: Final = "characters_1k"
# The ledger's kinds: what Vertex AI reported, and what is only the most the attempt may have cost.
KIND_ANSWERED: Final = "embedding"
KIND_UNCONFIRMED: Final = "embedding-unconfirmed"
# An error status is an answer, and Vertex AI bills no answer of the kind; nor a request that
# never left. Anything else that ends an attempt was sent.
UNBILLED: Final = (genai_errors.APIError, *NOT_SENT)


class EmbeddingRateLimitedError(RuntimeError):
    """Vertex AI answered 429 at every attempt: the embedder retries (the SDK makes one attempt),
    and a 429 that outlives its attempts stops the experiment, which resumes later."""


def _worth_retrying(error: Exception) -> bool:
    """What google-genai used to retry: a 429 or 5xx answer, a timeout, a refused connection."""
    if isinstance(error, genai_errors.APIError):
        return error.code in RETRY_STATUSES
    return isinstance(error, httpx2.TimeoutException | httpx2.ConnectError)


class VertexEmbedder:
    def __init__(
        self,
        route: ModelRoute,
        *,
        gcp_project: str,
        task: EmbeddingTask,
        budget: Budget,
        run_id: str,
        client: Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if route.provider != EMBEDDING_PROVIDER or route.dimension is None:
            raise ValueError(f"route '{route.route_id}' is not an embedding route")
        self._route = route
        self._task = task
        self._budget = budget
        self._price = budget.price(route.route_id)
        if self._price.unit != PRICE_UNIT:
            # The texts are counted in characters and bytes: a price per million tokens would
            # make each of them cost a thousandth of what it does.
            raise BudgetError(
                f"{budget.path}: route '{route.route_id}' is priced per {self._price.unit}, "
                "and the embedder counts characters"
            )
        self._run_id = run_id
        self._gcp_project = gcp_project
        self._client = client or google_client(route, gcp_project=gcp_project)
        self._sleep = sleep
        self.dimension = route.dimension
        self.identity = f"vertex:{route.model_name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """The text in up to `MAX_ATTEMPTS` attempts. The last ends it with what it ended with: a
        429 as `EmbeddingRateLimitedError`, a failure to reach the model as the `RouteAccessError`
        that explains it, anything else as it is."""
        size = len(text.encode("utf-8"))
        reserve = self._price.cost(input_units=size)
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._attempt(text, size, reserve)
            except Exception as error:
                if attempt < MAX_ATTEMPTS - 1 and _worth_retrying(error):
                    self._sleep(RETRY_WAITS[attempt])
                    continue
                if isinstance(error, genai_errors.APIError) and error.code == 429:
                    raise EmbeddingRateLimitedError(
                        f"{self._route.model_name} is rate limited (Vertex AI answered 429 after "
                        f"{MAX_ATTEMPTS} attempts): run the experiment again later"
                    ) from error
                explained = explain_failure(error, self._route, self._gcp_project)
                if explained is not None:
                    raise explained from error
                raise
        raise AssertionError("the last attempt returns or raises")  # pragma: no cover

    def _attempt(self, text: str, size: int, reserve: float) -> list[float]:
        """One attempt: the budget must have room for its reserve, and it is settled as soon as it
        ends. A ledger that cannot take the line ends the text with its error, never with a retry:
        another attempt would be paid and uncounted."""
        with self._ledger("read"):
            self._budget.ensure_room(reserve)
        try:
            response = self._client.models.embed_content(
                model=self._route.model_name,
                contents=[text],
                config=EmbedContentConfig(task_type=self._task, auto_truncate=False),
            )
        except UNBILLED:
            raise
        except BaseException:
            # Sent, but no answer came back: Vertex AI may have billed it.
            self._record(reserve, KIND_UNCONFIRMED)
            raise
        billed = response.metadata.billable_character_count if response.metadata else None
        if billed:
            self._record(self._price.cost(input_units=max(billed, size)), KIND_ANSWERED)
        else:  # an answer that reports no usage may still have been billed in full
            self._record(reserve, KIND_UNCONFIRMED)
        embeddings = response.embeddings or []
        if len(embeddings) != 1 or embeddings[0].values is None:
            raise ValueError(f"{self._route.model_name} returned no embedding")
        vector = [float(value) for value in embeddings[0].values]
        if len(vector) != self.dimension:
            raise ValueError(
                f"{self._route.model_name} returned {len(vector)} values, "
                f"not the {self.dimension} of its route"
            )
        return vector

    def _record(self, usd: float, kind: str) -> None:
        with self._ledger("appended to"):
            self._budget.record(
                run_id=self._run_id, route_id=self._route.route_id, kind=kind, usd=usd
            )

    @contextmanager
    def _ledger(self, action: str) -> Iterator[None]:
        """The ledger's own failures are the budget's errors, named with the file: an OSError from
        the ledger must never be taken for one from the network."""
        try:
            yield
        except OSError as error:
            raise BudgetError(
                f"{self._budget.ledger_path}: cannot be {action} ({error.strerror or error})"
            ) from error
