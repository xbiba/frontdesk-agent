"""Google tokens for the calendar reader: ADC, impersonated when an account is named."""

from typing import Any

import pytest

from frontdesk.model_gateway import google_tokens


class FakeCredentials:
    def __init__(self) -> None:
        self.valid = False
        self.token: str | None = None
        self.refreshes = 0

    def refresh(self, request: Any) -> None:
        self.refreshes += 1
        self.valid, self.token = True, f"token-{self.refreshes}"


def test_adc_tokens_refresh_only_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = FakeCredentials()
    monkeypatch.setattr(google_tokens.google.auth, "default", lambda scopes: (credentials, "p"))
    token = google_tokens.google_token_source(["scope"], impersonate=None)
    assert (token(), token()) == ("token-1", "token-1")
    assert credentials.refreshes == 1


def test_a_named_account_is_impersonated(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = FakeCredentials(), FakeCredentials()
    seen: dict[str, Any] = {}

    def impersonate(**kwargs: Any) -> FakeCredentials:
        seen.update(kwargs)
        return target

    monkeypatch.setattr(google_tokens.google.auth, "default", lambda scopes: (source, "p"))
    monkeypatch.setattr(google_tokens.impersonated_credentials, "Credentials", impersonate)
    token = google_tokens.google_token_source(["scope"], impersonate="sa@p.iam.gserviceaccount.com")
    assert token() == "token-1"
    assert seen == {
        "source_credentials": source,
        "target_principal": "sa@p.iam.gserviceaccount.com",
        "target_scopes": ["scope"],
    }
    assert source.refreshes == 0
