"""Checks before a live run spends anything (M3 design §5, §6).

The model clients ignore proxy and certificate variables (final review P63), but Google's token
refresh still reads them, and `google.auth.default()` reads GOOGLE_CLOUD_QUOTA_PROJECT, which moves
the project a call is billed to; a live run therefore refuses to start while any is set. The access
check sends one CountTokens request, which Vertex AI does not charge, through the pinned client: it
proves the credentials, the API and the model at the pinned endpoint before the first paid call,
and turns each failure into what to do about it.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Final

import httpx2
from google.auth import exceptions as auth_errors
from google.genai import Client
from google.genai import errors as genai_errors

from frontdesk.model_gateway.factory import VERTEX_HOSTS, google_client, pinned_location
from frontdesk.model_gateway.routes import ModelRoute

ENVIRONMENT_OVERRIDES: Final = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
)
_LOGIN: Final = "`gcloud auth application-default login`"
# What the network can do to a request that nothing in Google's SDK wraps: refused or lost
# connections and timeouts, whether httpx or the socket layer says so.
_NETWORK: Final = (httpx2.HTTPError, OSError, TimeoutError)


class RouteAccessError(RuntimeError):
    """The live route cannot be used now; the message says what to do."""


def environment_overrides(environ: Mapping[str, str]) -> tuple[str, ...]:
    """The proxy and certificate variables that are set, in either case, each named as it is set
    (`https_proxy`, not `HTTPS_PROXY`), so that `unset NAME` is the advice that works. Sorted by
    name without regard to case."""
    found = [
        name
        for name, value in environ.items()
        if name.upper() in ENVIRONMENT_OVERRIDES and value.strip()
    ]
    return tuple(sorted(found, key=lambda name: (name.upper(), name)))


async def check_route_access(
    route: ModelRoute, *, gcp_project: str, client: Client | None = None
) -> None:
    if route.provider != "google-vertex":
        # The request is Gemini's: sent for another provider's model it would fail for a reason
        # that has nothing to do with the route, and a run must not start on a route it cannot prove.
        raise RouteAccessError(
            f"route '{route.route_id}' ({route.provider}) has no free access check: "
            "the CountTokens request is for Gemini routes on Vertex AI"
        )
    location = pinned_location(route)
    used = client if client is not None else google_client(route, gcp_project=gcp_project)
    try:
        await used.aio.models.count_tokens(model=route.model_name, contents="ping")
    except (auth_errors.DefaultCredentialsError, auth_errors.RefreshError) as error:
        raise RouteAccessError(
            f"Google credentials are missing or expired: run {_LOGIN}, then try again"
        ) from error
    except auth_errors.GoogleAuthError as error:
        raise RouteAccessError(
            f"Google could not sign the request ({type(error).__name__}): check the network, "
            f"then run {_LOGIN} if it happens again"
        ) from error
    except genai_errors.ClientError as error:
        raise _refused(error, route, gcp_project, location) from error
    except genai_errors.ServerError as error:
        raise RouteAccessError(
            f"Vertex AI is not answering ({error.code} {error.status}): try again later"
        ) from error
    except _NETWORK as error:
        # Only the type: an httpx message can carry the URL it tried.
        raise RouteAccessError(
            f"Vertex AI cannot be reached ({type(error).__name__}): check the network "
            "connection, then try again"
        ) from error
    finally:
        if client is None:  # the one this call built is this call's to close
            with suppress(Exception):
                await used.aio.aclose()


def _refused(
    error: genai_errors.ClientError, route: ModelRoute, gcp_project: str, location: str
) -> RouteAccessError:
    if error.code in {401, 403}:
        return RouteAccessError(
            f"Vertex AI refused project '{gcp_project}' ({error.status}: {error.message}). "
            "Check that the API is enabled (`gcloud services enable aiplatform.googleapis.com "
            f"--project {gcp_project}`) and that your account may use it"
        )
    if error.code == 404:
        return RouteAccessError(
            f"model '{route.model_name}' is not served at {VERTEX_HOSTS[location]}"
        )
    if error.code == 429:
        return RouteAccessError(
            f"Vertex AI is rate limiting project '{gcp_project}' ({error.code} {error.status}): "
            "try again later"
        )
    return RouteAccessError(f"Vertex AI answered {error.code} {error.status}")
