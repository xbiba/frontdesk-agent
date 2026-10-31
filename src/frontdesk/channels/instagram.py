"""Instagram (M4): the adapter of the Instagram API with Facebook Login (ADR-0010).

The account is linked to a Facebook Page, and the token is the business's system-user token. It
checks the signature Meta puts on every webhook delivery (`X-Hub-Signature-256`, an HMAC-SHA256 of
the raw body with the app secret), reads the deliveries' events and sends texts with the Send API,
through the Page and with the Page's token. Nothing here logs a message, a person's ID or a token.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, Final

import httpx2

from frontdesk.channels.messages import Inbound, InboundKind, SendOutcome, SendStatus

INSTAGRAM: Final = "instagram"
GRAPH: Final = "https://graph.facebook.com"
MAX_MESSAGE_BYTES: Final = 1000  # Meta: the text is "UTF-8 and 1000 bytes or less"
MEDIA: Final = frozenset({"image", "video", "audio", "file"})
CONNECT_RETRIES: Final = 2  # D9: a request that never reached Meta is tried twice more
TOKEN_ERROR: Final = 190  # Meta's code for an expired or revoked token
_PREFIX: Final = "sha256="


def signature_ok(app_secret: str, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith(_PREFIX):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header[len(_PREFIX) :].encode("utf-8"), expected.encode("ascii"))


class InstagramWebhook:
    """The receiving side (a `WebhookReader`)."""

    name = INSTAGRAM

    def __init__(self, app_secret: str) -> None:
        self._secret = app_secret

    def signature_ok(self, body: bytes, signature: str | None) -> bool:
        return signature_ok(self._secret, body, signature)

    def parse(self, body: bytes) -> list[Inbound]:
        return parse_events(body)


def parse_events(body: bytes) -> list[Inbound]:
    """The events of one delivery, in order. A delivery that is not JSON, or not for Instagram,
    raises ValueError; an event of a shape this code does not know is IGNORED."""
    try:
        payload = json.loads(body)
    except ValueError as error:  # bad UTF-8 or bad JSON
        raise ValueError("the delivery is not JSON") from error
    if not isinstance(payload, dict) or payload.get("object") != INSTAGRAM:
        raise ValueError("the delivery is not for an Instagram account")
    return [
        _event(str(entry.get("id", "")), item)
        for entry in _mappings(payload.get("entry"))
        for item in _mappings(entry.get("messaging"))
    ]


def _event(account: str, item: Mapping[str, Any]) -> Inbound:
    sender, recipient = _id(item.get("sender")), _id(item.get("recipient"))
    sent_at = _time(item.get("timestamp"))
    message = item.get("message")
    if not isinstance(message, Mapping):
        return Inbound(account, sender, InboundKind.IGNORED, sent_at=sent_at)  # reactions, reads
    mid = message.get("mid") if isinstance(message.get("mid"), str) else None
    text = _text(message)
    if message.get("is_echo"):
        return Inbound(account, recipient, InboundKind.ECHO, mid, text, sent_at)
    if message.get("is_deleted"):
        return Inbound(account, sender, InboundKind.DELETED, mid, sent_at=sent_at)
    if text is not None:
        return Inbound(account, sender, InboundKind.TEXT, mid, text, sent_at)
    kinds = {attachment.get("type") for attachment in _mappings(message.get("attachments"))}
    kind = InboundKind.ATTACHMENT if kinds & MEDIA else InboundKind.IGNORED
    return Inbound(account, sender, kind, mid, sent_at=sent_at)


def _mappings(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, Mapping))


def _id(value: object) -> str:
    found = value.get("id") if isinstance(value, Mapping) else None
    return str(found) if isinstance(found, str | int) and not isinstance(found, bool) else ""


def _text(message: Mapping[str, Any]) -> str | None:
    """The message's text, without NUL characters: Postgres refuses them, and a delivery it
    refused came back from Meta for 36 hours (final review of M4, M11)."""
    text = message.get("text")
    if not isinstance(text, str):
        return None
    kept = text.replace("\x00", "").strip()
    return kept or None


def _time(value: object) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, UTC)
    except (OverflowError, OSError, ValueError):
        return None


class PageToken:
    """The Page's token, asked of Meta once with the system user's: the Send API and the Page's
    subscription take only a Page token. A system user's Page token does not expire."""

    def __init__(
        self, client: httpx2.AsyncClient, system_token: str, *, page_id: str, api_version: str
    ) -> None:
        self._client = client
        self._system_token = system_token
        self._url = f"{GRAPH}/{api_version}/{page_id}"
        self._value: str | None = None

    async def get(self) -> str:
        if self._value is None:
            response = await self._client.get(
                self._url, params={"fields": "access_token"}, headers=_bearer(self._system_token)
            )
            value = _json(response).get("access_token") if response.status_code == 200 else None
            if not isinstance(value, str) or not value:
                raise ValueError(f"Meta gave no Page token (status {response.status_code})")
            self._value = value
        return self._value


class InstagramSender:
    """The sending side (a `Sender`): the Send API, through the Page, with the Page's token."""

    name = INSTAGRAM
    max_message_bytes = MAX_MESSAGE_BYTES

    def __init__(
        self,
        token: Callable[[], Awaitable[str]],
        client: httpx2.AsyncClient,
        *,
        page_id: str,
        api_version: str,
    ) -> None:
        self._token = token
        self._client = client
        self._url = f"{GRAPH}/{api_version}/{page_id}/messages"

    async def send(self, user_id: str, text: str, kind: str = "reply") -> SendOutcome:
        try:
            headers = _bearer(await self._token())
        except (ValueError, httpx2.HTTPError):
            return SendOutcome(SendStatus.FAILED, reason="page_token")  # nothing was sent
        body = {"recipient": {"id": user_id}, "message": {"text": text}}
        for _ in range(CONNECT_RETRIES + 1):
            try:
                response = await self._client.post(self._url, json=body, headers=headers)
            except (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout):
                continue  # the request never left: it is safe to try again
            except httpx2.HTTPError:
                return SendOutcome(SendStatus.UNCERTAIN, reason="transport")
            return _outcome(response)
        return SendOutcome(SendStatus.FAILED, reason="connect")


def _outcome(response: httpx2.Response) -> SendOutcome:
    answer = _json(response)
    if response.status_code == 200:
        mid = answer.get("message_id")
        if isinstance(mid, str) and mid:
            return SendOutcome(SendStatus.SENT, message_id=mid)
        return SendOutcome(SendStatus.UNCERTAIN, reason="no_message_id")
    if response.status_code >= 500:
        return SendOutcome(SendStatus.UNCERTAIN, reason=f"status_{response.status_code}")
    error = answer.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if code == TOKEN_ERROR:
        return SendOutcome(SendStatus.FAILED, reason="token")
    suffix = f"_code_{code}" if isinstance(code, int) else ""
    return SendOutcome(SendStatus.FAILED, reason=f"status_{response.status_code}{suffix}")


def _json(response: httpx2.Response) -> Mapping[str, Any]:
    try:
        answer = response.json()
    except ValueError:
        return {}
    return answer if isinstance(answer, Mapping) else {}


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def linked_accounts(
    client: httpx2.AsyncClient, token: str, api_version: str
) -> list[tuple[str, str, str]]:
    """The Pages the token reaches that have an Instagram account: the Page's ID, the ID the
    account's webhooks carry, and the account's username."""
    response = await client.get(
        f"{GRAPH}/{api_version}/me/accounts",
        params={"fields": "id,instagram_business_account{id,username}"},
        headers=_bearer(token),
    )
    if response.status_code != 200:
        raise ValueError(f"Meta listed no Pages (status {response.status_code})")
    found = []
    for page in _mappings(_json(response).get("data")):
        account = page.get("instagram_business_account")
        if isinstance(account, Mapping) and _id(page) and _id(account):
            found.append((_id(page), _id(account), str(account.get("username") or "")))
    return found


async def subscribe(
    client: httpx2.AsyncClient, page_token: str, page_id: str, api_version: str
) -> None:
    """Subscribe the Page, and so its Instagram account, to this app's `messages` webhooks."""
    response = await client.post(
        f"{GRAPH}/{api_version}/{page_id}/subscribed_apps",
        params={"subscribed_fields": "messages"},
        headers=_bearer(page_token),
    )
    if response.status_code != 200 or _json(response).get("success") is not True:
        raise ValueError(f"Meta did not subscribe the Page (status {response.status_code})")
