"""Gmail's access tokens for the email drafts (2026-11-07), two ways, neither with a downloaded key:

- domain-wide delegation, in production: the service account signs, through IAM's signJwt, a
  token request that names the mailbox as its subject; Google answers with a token for that
  mailbox, for the scopes the business's Workspace administrator granted the account (only
  gmail.modify: no scope that sends). The signing call goes with the service's own identity;
- a person's OAuth token file, in development (Albi's own Gmail, an app in "Testing"): the refresh
  token it holds is exchanged for an access token. `login_url` and `exchange_code` write that file
  once, through a page on this Mac (`frontdesk email login`).

No token, key, code or response body is ever part of an error.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode

import httpx2

from frontdesk.model_gateway.google_tokens import TokenSource

TOKEN_URI: Final = "https://oauth2.googleapis.com/token"  # noqa: S105 - a URL, not a secret
AUTH_URI: Final = "https://accounts.google.com/o/oauth2/v2/auth"
SIGN_JWT: Final = (
    "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{account}:signJwt"
)
LIFETIME_SECONDS: Final = 3600
EARLY_SECONDS: Final = 60  # a token is renewed this long before it expires
JWT_BEARER: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"


class MailTokenError(RuntimeError):
    """A Gmail token could not be had; the message says which step and the status."""


def _cached(fetch: Callable[[], tuple[str, float]], clock: Callable[[], float]) -> TokenSource:
    held: list[tuple[str, float]] = []

    def token() -> str:
        if not held or clock() >= held[0][1] - EARLY_SECONDS:
            held[:] = [fetch()]
        return held[0][0]

    return token


def _access_token(
    response: httpx2.Response, step: str, clock: Callable[[], float]
) -> tuple[str, float]:
    if response.status_code != 200:
        raise MailTokenError(f"{step}: status {response.status_code}")
    try:
        data = response.json()
        return str(data["access_token"]), clock() + float(data.get("expires_in", LIFETIME_SECONDS))
    except (KeyError, TypeError, ValueError):
        raise MailTokenError(f"{step}: malformed answer") from None


def delegated_token_source(
    scopes: Sequence[str],
    *,
    service_account: str,
    subject: str,
    signer_token: TokenSource,
    client: httpx2.Client,
    clock: Callable[[], float] = time.time,
) -> TokenSource:
    """A token for `subject`'s mailbox, through domain-wide delegation to `service_account`.
    `signer_token` is a cloud-platform token of an identity that may sign as the account (the
    account itself on Cloud Run, with the Service Account Token Creator role on itself)."""

    def fetch() -> tuple[str, float]:
        now = int(clock())
        claims = {
            "iss": service_account,
            "sub": subject,
            "scope": " ".join(scopes),
            "aud": TOKEN_URI,
            "iat": now,
            "exp": now + LIFETIME_SECONDS,
        }
        try:
            signed = client.post(
                SIGN_JWT.format(account=service_account),
                headers={"Authorization": f"Bearer {signer_token()}"},
                json={"payload": json.dumps(claims)},
            )
            if signed.status_code != 200:
                raise MailTokenError(f"signJwt: status {signed.status_code}")
            assertion = str(signed.json()["signedJwt"])
            answer = client.post(TOKEN_URI, data={"grant_type": JWT_BEARER, "assertion": assertion})
        except httpx2.HTTPError as error:
            raise MailTokenError(f"delegated token: {type(error).__name__}") from None
        except (KeyError, TypeError, ValueError):
            raise MailTokenError("signJwt: malformed answer") from None
        return _access_token(answer, "delegated token", clock)

    return _cached(fetch, clock)


def user_token_source(
    token_file: Path, *, client: httpx2.Client, clock: Callable[[], float] = time.time
) -> TokenSource:
    """A token from a person's OAuth token file (client_id, client_secret, refresh_token)."""

    def fetch() -> tuple[str, float]:
        try:
            saved = json.loads(token_file.read_text(encoding="utf-8"))
            form = {
                "grant_type": "refresh_token",
                "client_id": saved["client_id"],
                "client_secret": saved["client_secret"],
                "refresh_token": saved["refresh_token"],
            }
        except (OSError, KeyError, TypeError, ValueError):
            raise MailTokenError(
                f"{token_file.name}: not a token file of `frontdesk email login`"
            ) from None
        try:
            answer = client.post(TOKEN_URI, data=form)
        except httpx2.HTTPError as error:
            raise MailTokenError(f"refresh token: {type(error).__name__}") from None
        return _access_token(answer, "refresh token", clock)

    return _cached(fetch, clock)


@dataclass(frozen=True)
class Login:
    """One login in progress: the page to open and what the answer is checked with."""

    url: str
    state: str
    verifier: str
    redirect_uri: str
    client_id: str
    client_secret: str


def read_client(path: Path) -> tuple[str, str]:
    """The client ID and secret of an "installed app" client file from Google Cloud's console."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        client = data.get("installed") or data.get("web") or data
        return str(client["client_id"]), str(client["client_secret"])
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        raise MailTokenError(f"{path.name}: not an OAuth client file") from None


def login_url(
    client_id: str, client_secret: str, scopes: Sequence[str], redirect_uri: str
) -> Login:
    """The consent page for `scopes`, with PKCE and a state, offline access for a refresh token."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    state = secrets.token_urlsafe(24)
    query = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "code_challenge": challenge.decode(),
        "code_challenge_method": "S256",
    }
    return Login(
        f"{AUTH_URI}?{urlencode(query)}", state, verifier, redirect_uri, client_id, client_secret
    )


def exchange_code(
    login: Login, code: str, state: str, token_file: Path, client: httpx2.Client
) -> None:
    """Exchange the consent's code for a refresh token and write the token file, readable by its
    owner only."""
    if not secrets.compare_digest(state, login.state):
        raise MailTokenError("login: the answer's state does not match")
    try:
        answer = client.post(
            TOKEN_URI,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": login.client_id,
                "client_secret": login.client_secret,
                "redirect_uri": login.redirect_uri,
                "code_verifier": login.verifier,
            },
        )
    except httpx2.HTTPError as error:
        raise MailTokenError(f"login: {type(error).__name__}") from None
    if answer.status_code != 200:
        raise MailTokenError(f"login: status {answer.status_code}")
    refresh = answer.json().get("refresh_token")
    if not refresh:
        raise MailTokenError("login: Google gave no refresh token")
    saved = {
        "type": "authorized_user",
        "client_id": login.client_id,
        "client_secret": login.client_secret,
        "refresh_token": refresh,
    }
    token_file.touch(mode=0o600, exist_ok=True)
    token_file.chmod(0o600)
    token_file.write_text(json.dumps(saved), encoding="utf-8")
