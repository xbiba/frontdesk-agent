"""Scripted Vertex AI embedding calls for the embedder's tests: a client that ends each request as
the test says, and a sleep that only writes the wait down. A step is a function, so that a failure
is a new exception each time it is raised."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.genai import Client
from google.genai import errors as genai_errors

Step = Callable[[], object]


class Waits:
    """Stands for `time.sleep`: it writes each wait down and does not wait."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class ScriptedEmbeddings:
    """Ends its n-th request as the n-th step says: a step that gives an exception raises it, any
    other gives the answer as it is. A request more than scripted is an error in the test."""

    def __init__(self, *script: Step) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def embed_content(self, **request: Any) -> object:
        step = self._script[len(self.requests)]
        self.requests.append(request)
        outcome = step()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def client(self) -> Client:
        return cast(Client, SimpleNamespace(models=self))


def answer(billed: int | None = 1_000, dimension: int = 3072) -> Step:
    """A good answer; `billed` is the characters Vertex AI says it counted (None: no metadata)."""

    def step() -> object:
        metadata = None if billed is None else SimpleNamespace(billable_character_count=billed)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.5] * dimension)], metadata=metadata
        )

    return step


def no_embedding() -> object:
    """A 200 answer that carries no vector."""
    return SimpleNamespace(embeddings=[], metadata=SimpleNamespace(billable_character_count=10))


def status(code: int) -> Step:
    """An answer of an error status, as the SDK raises it: 4xx a ClientError, 5xx a ServerError."""

    def step() -> BaseException:
        error = {"error": {"code": code, "status": "SCRIPTED", "message": "scripted"}}
        kind = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
        return kind(code, error)

    return step


def expired_credentials() -> BaseException:
    """The refresh of the access token fails: the credentials expired."""
    return RefreshError("Reauthentication is needed")


def no_credentials() -> BaseException:
    """There are no Application Default Credentials at all."""
    return DefaultCredentialsError("Your default credentials were not found.")
