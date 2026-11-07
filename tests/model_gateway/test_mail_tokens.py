"""Gmail's tokens: domain-wide delegation through IAM's signJwt, and a person's token file in
development. No key is downloaded, and no token reaches an error."""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest

from frontdesk.model_gateway import mail_tokens
from frontdesk.model_gateway.mail_tokens import (
    MailTokenError,
    delegated_token_source,
    exchange_code,
    login_url,
    read_client,
    user_token_source,
)

SCOPE = "https://www.googleapis.com/auth/gmail.modify"
ACCOUNT = "frontdesk-email@demo.iam.gserviceaccount.com"


def _client(handler):  # type: ignore[no-untyped-def]
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_delegation_signs_with_iam_and_names_the_mailbox() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if "signJwt" in str(request.url):
            return httpx2.Response(200, json={"signedJwt": "signed.jwt"})
        return httpx2.Response(200, json={"access_token": "gmail-token", "expires_in": 3600})

    clock = [1000.0]
    token = delegated_token_source(
        [SCOPE],
        service_account=ACCOUNT,
        subject="info@example.com",
        allowed_subjects=frozenset({"info@example.com"}),
        signer_token=lambda: "cloud-token",
        client=_client(handler),
        clock=lambda: clock[0],
    )
    assert token() == "gmail-token"
    assert token() == "gmail-token"
    assert len(seen) == 2  # the second call used the held token
    sign, grant = seen
    assert sign.headers["Authorization"] == "Bearer cloud-token"
    claims = json.loads(json.loads(sign.content)["payload"])
    assert claims["sub"] == "info@example.com"
    assert claims["scope"] == SCOPE
    assert claims["iss"] == ACCOUNT
    assert parse_qs(grant.content.decode())["assertion"] == ["signed.jwt"]
    clock[0] += 3600
    token()
    assert len(seen) == 4  # renewed before it expired


@pytest.mark.parametrize(("status", "step"), [(403, "signJwt"), (200, "delegated token")])
def test_a_refused_step_is_named_without_its_answer(status: int, step: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if "signJwt" in str(request.url):
            return httpx2.Response(status, json={"signedJwt": "s", "error": "secret detail"})
        return httpx2.Response(401, json={"error": "secret detail"})

    token = delegated_token_source(
        [SCOPE],
        service_account=ACCOUNT,
        subject="info@example.com",
        allowed_subjects=frozenset({"info@example.com"}),
        signer_token=lambda: "t",
        client=_client(handler),
    )
    with pytest.raises(MailTokenError, match=step) as raised:
        token()
    assert "secret" not in str(raised.value)


def test_a_token_file_is_refreshed(tmp_path: Path) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps({"client_id": "c", "client_secret": "s", "refresh_token": "r"}), encoding="utf-8"
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        form = parse_qs(request.content.decode())
        assert form["refresh_token"] == ["r"]
        return httpx2.Response(200, json={"access_token": "user-token"})

    assert user_token_source(token_file, client=_client(handler))() == "user-token"
    token_file.write_text("{}", encoding="utf-8")
    with pytest.raises(MailTokenError, match="not a token file"):
        user_token_source(token_file, client=_client(handler))()


def test_login_writes_a_private_token_file(tmp_path: Path) -> None:
    client_file = tmp_path / "client.json"
    client_file.write_text(json.dumps({"installed": {"client_id": "c", "client_secret": "s"}}))
    client_id, secret = read_client(client_file)
    login = login_url(client_id, secret, [SCOPE], "http://127.0.0.1:8765/")
    query = parse_qs(urlparse(login.url).query)
    assert query["scope"] == [SCOPE]
    assert query["code_challenge_method"] == ["S256"]

    def handler(request: httpx2.Request) -> httpx2.Response:
        form = parse_qs(request.content.decode())
        assert form["code_verifier"] == [login.verifier]
        return httpx2.Response(200, json={"refresh_token": "r", "access_token": "a"})

    token_file = tmp_path / "token.json"
    with pytest.raises(MailTokenError, match="state"):
        exchange_code(login, "code", "other", token_file, _client(handler))
    exchange_code(login, "code", login.state, token_file, _client(handler))
    assert json.loads(token_file.read_text())["refresh_token"] == "r"
    assert token_file.stat().st_mode & 0o077 == 0
    with pytest.raises(MailTokenError, match="not an OAuth client"):
        read_client(token_file.with_name("missing.json"))


def test_no_scope_that_sends_is_ever_asked() -> None:
    from frontdesk.email import gmail

    assert gmail.SCOPES == ("https://www.googleapis.com/auth/gmail.modify",)
    assert "send" not in " ".join(gmail.SCOPES)
    assert mail_tokens.TOKEN_URI.startswith("https://oauth2.googleapis.com/")


def test_delegation_refuses_any_mailbox_but_the_configured_one() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request may leave")

    with pytest.raises(MailTokenError, match="only the configured mailbox"):
        delegated_token_source(
            [SCOPE],
            service_account=ACCOUNT,
            subject="ceo@example.com",
            allowed_subjects=frozenset({"info@example.com"}),
            signer_token=lambda: "t",
            client=_client(handler),
        )
