"""Secrets (Albi, 2026-10-13, before Instagram): cal.com's API key, and Instagram's token and app
secret when M4 brings them.

In production (FRONTDESK_SECRETS_BACKEND=gcp) they come from Google Secret Manager, read at start
with the service's own identity, never from a file: a key also found in the environment is refused,
since an env file is where it would come from. In development they come from the environment
(FRONTDESK_CALCOM_API_KEY, kept outside the repositories). A secret that cannot be read is named in
the error, never its value.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from typing import Final

import httpx2
from pydantic import SecretStr

from frontdesk.settings import SecretsBackend, Settings

CLOUD_SCOPE: Final = "https://www.googleapis.com/auth/cloud-platform"
SECRET_IDS: Final = {
    "calcom_api_key": "frontdesk-calcom-api-key",
    "instagram_token": "frontdesk-instagram-token",
    "instagram_app_secret": "frontdesk-instagram-app-secret",
    "instagram_verify_token": "frontdesk-instagram-verify-token",
    "instagram_pairing_code": "frontdesk-instagram-pairing-code",
    "whatsapp_token": "frontdesk-whatsapp-token",
    "whatsapp_app_secret": "frontdesk-whatsapp-app-secret",
    "whatsapp_verify_token": "frontdesk-whatsapp-verify-token",
    "whatsapp_staff": "frontdesk-whatsapp-staff",
}
_API: Final = "https://secretmanager.googleapis.com/v1"


class SecretUnavailableError(RuntimeError):
    """A secret could not be read from Secret Manager."""


def read_secret(
    settings: Settings,
    name: str,
    *,
    token: Callable[[], str] | None = None,
    client: httpx2.Client | None = None,
) -> str | None:
    """The secret `name` (a key of SECRET_IDS), or None when development has none."""
    found = getattr(settings, name, None)
    in_environment = found.get_secret_value() if isinstance(found, SecretStr) else None
    if settings.secrets_backend is SecretsBackend.ENV:
        return in_environment
    if in_environment is not None:
        raise ValueError(
            f"{name} is set in the environment: with Secret Manager it is never in the environment"
        )
    project = settings.secrets_project or settings.gcp_project
    if not project:
        raise ValueError("FRONTDESK_SECRETS_BACKEND=gcp needs FRONTDESK_SECRETS_PROJECT")
    if token is None or client is None:
        raise ValueError("reading Secret Manager needs a token source and an HTTP client")
    secret_id = SECRET_IDS[name]
    url = f"{_API}/projects/{project}/secrets/{secret_id}/versions/latest:access"
    try:
        response = client.get(url, headers={"Authorization": f"Bearer {token()}"})
    except httpx2.HTTPError as error:
        raise SecretUnavailableError(f"{secret_id}: {type(error).__name__}") from None
    if response.status_code != 200:
        raise SecretUnavailableError(f"{secret_id}: status {response.status_code}")
    try:
        data = response.json()["payload"]["data"]
        return base64.b64decode(data).decode("utf-8").strip()
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise SecretUnavailableError(f"{secret_id}: malformed answer") from None


def add_secret_version(
    settings: Settings,
    name: str,
    value: str,
    *,
    token: Callable[[], str],
    client: httpx2.Client,
) -> None:
    """Store `value` as the newest version of the secret `name`: the Instagram token, renewed every
    week (M4 design D3). The program never writes a development secret."""
    if settings.secrets_backend is SecretsBackend.ENV:
        raise ValueError(f"{name} lives in the environment: the program cannot renew it")
    project = settings.secrets_project or settings.gcp_project
    if not project:
        raise ValueError("FRONTDESK_SECRETS_BACKEND=gcp needs FRONTDESK_SECRETS_PROJECT")
    secret_id = SECRET_IDS[name]
    data = base64.b64encode(value.encode("utf-8")).decode("ascii")
    try:
        response = client.post(
            f"{_API}/projects/{project}/secrets/{secret_id}:addVersion",
            json={"payload": {"data": data}},
            headers={"Authorization": f"Bearer {token()}"},
        )
    except httpx2.HTTPError as error:
        raise SecretUnavailableError(f"{secret_id}: {type(error).__name__}") from None
    if response.status_code != 200:
        raise SecretUnavailableError(f"{secret_id}: status {response.status_code}")
