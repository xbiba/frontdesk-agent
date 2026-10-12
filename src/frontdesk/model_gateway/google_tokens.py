"""Google access tokens for the engine's other Google APIs: today the calendar reader (ADR-0003).

Only the model gateway reads Google credentials. Application Default Credentials, impersonating a
service account when one is named (on the Mac); on Cloud Run, the service's own identity. No JSON
key anywhere. The refresh goes through a session that ignores proxy and certificate variables and
verifies with certifi, like the model clients (final review P63).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import certifi
import google.auth
import requests
from google.auth import impersonated_credentials
from google.auth.transport.requests import Request

TokenSource = Callable[[], str]


def google_token_source(scopes: Sequence[str], *, impersonate: str | None) -> TokenSource:
    credentials, _ = google.auth.default(scopes=list(scopes))
    if impersonate:
        credentials = impersonated_credentials.Credentials(  # type: ignore[no-untyped-call]
            source_credentials=credentials,
            target_principal=impersonate,
            target_scopes=list(scopes),
        )
    session = requests.Session()
    session.trust_env = False
    session.verify = certifi.where()
    request = Request(session=session)

    def token() -> str:
        if not credentials.valid:
            credentials.refresh(request)  # type: ignore[no-untyped-call]
        return str(credentials.token)

    return token
