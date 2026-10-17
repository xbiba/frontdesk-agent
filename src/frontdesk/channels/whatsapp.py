"""WhatsApp (W1): the adapter of the WhatsApp Cloud API on Meta's Graph, for the reception's staff.

It checks the signature Meta puts on every webhook delivery (Instagram's, with the same app
secret), reads the deliveries' messages and sends texts and templates. Nothing here logs a
message, a phone number or the token.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import httpx2

from frontdesk.channels.instagram import CONNECT_RETRIES, TOKEN_ERROR, signature_ok
from frontdesk.channels.messages import Inbound, InboundKind, SendOutcome, SendStatus

WHATSAPP: Final = "whatsapp"
GRAPH: Final = "https://graph.facebook.com"
TEMPLATE: Final = "template"  # the outbox kind of a template part
MAX_MESSAGE_BYTES: Final = 4096  # Meta: a text's body is at most 4096 characters
MEDIA: Final = frozenset({"audio", "image", "video", "document", "sticker", "location"})
_OBJECT: Final = "whatsapp_business_account"


class WhatsAppWebhook:
    """The receiving side (a `WebhookReader`)."""

    name = WHATSAPP

    def __init__(self, app_secret: str) -> None:
        self._secret = app_secret

    def signature_ok(self, body: bytes, signature: str | None) -> bool:
        return signature_ok(self._secret, body, signature)

    def parse(self, body: bytes) -> list[Inbound]:
        return parse_events(body)


def parse_events(body: bytes) -> list[Inbound]:
    """The messages of one delivery, in order. A delivery that is not JSON, or not for WhatsApp,
    raises ValueError; statuses and every other change give no event."""
    try:
        payload = json.loads(body)
    except ValueError as error:  # bad UTF-8 or bad JSON
        raise ValueError("the delivery is not JSON") from error
    if not isinstance(payload, dict) or payload.get("object") != _OBJECT:
        raise ValueError("the delivery is not for a WhatsApp Business account")
    events: list[Inbound] = []
    for entry in _mappings(payload.get("entry")):
        for change in _mappings(entry.get("changes")):
            value = change.get("value")
            if change.get("field") != "messages" or not isinstance(value, Mapping):
                continue
            account = _string(value.get("metadata"), "phone_number_id") or ""
            events.extend(_event(account, message) for message in _mappings(value.get("messages")))
    return events


def _event(account: str, message: Mapping[str, Any]) -> Inbound:
    sender = message.get("from")
    user = sender.lstrip("+") if isinstance(sender, str) else ""
    mid = message.get("id") if isinstance(message.get("id"), str) else None
    sent_at = _time(message.get("timestamp"))
    kind = message.get("type")
    if not user:  # a business-scoped user ID only: no number to match the staff list
        return Inbound(account, "", InboundKind.IGNORED, mid, sent_at=sent_at)
    if kind == "text":
        text = _clean(_string(message.get("text"), "body"))
        found = InboundKind.TEXT if text else InboundKind.IGNORED
        return Inbound(account, user, found, mid, text, sent_at)
    if kind in {"button", "interactive"}:
        payload = _payload(message)
        found = InboundKind.BUTTON if payload else InboundKind.IGNORED
        return Inbound(account, user, found, mid, payload, sent_at)
    found = InboundKind.ATTACHMENT if kind in MEDIA else InboundKind.IGNORED
    return Inbound(account, user, found, mid, sent_at=sent_at)


def _payload(message: Mapping[str, Any]) -> str | None:
    """A template's quick reply carries `button.payload`; an interactive reply button its `id`."""
    if message.get("type") == "button":
        return _clean(_string(message.get("button"), "payload"))
    interactive = message.get("interactive")
    reply = interactive.get("button_reply") if isinstance(interactive, Mapping) else None
    return _clean(_string(reply, "id"))


def _mappings(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, Mapping))


def _string(value: object, key: str) -> str | None:
    found = value.get(key) if isinstance(value, Mapping) else None
    return found if isinstance(found, str) else None


def _clean(text: str | None) -> str | None:
    """Without NUL characters, which Postgres refuses (as for Instagram)."""
    kept = text.replace("\x00", "").strip() if text is not None else ""
    return kept or None


def _time(value: object) -> datetime | None:
    """Meta writes Unix seconds as a string."""
    if not isinstance(value, str) or not value.isdigit():
        return None
    try:
        return datetime.fromtimestamp(int(value), UTC)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class Template:
    """A template message as the outbox keeps it: the template's name and language, its body's
    parameters in order, and the payload of each quick reply button, in the template's order."""

    name: str
    language: str
    params: tuple[str, ...]
    payloads: tuple[str, ...]

    def to_text(self) -> str:
        fields = {
            "name": self.name,
            "language": self.language,
            "params": list(self.params),
            "payloads": list(self.payloads),
        }
        return json.dumps(fields, ensure_ascii=False)

    @classmethod
    def from_text(cls, text: str) -> Template:
        fields = json.loads(text)
        return cls(
            name=str(fields["name"]),
            language=str(fields["language"]),
            params=tuple(str(param) for param in fields["params"]),
            payloads=tuple(str(payload) for payload in fields["payloads"]),
        )


class WhatsAppSender:
    """The sending side (a `Sender`): the Cloud API's messages, from the bot's number."""

    name = WHATSAPP
    max_message_bytes = MAX_MESSAGE_BYTES

    def __init__(
        self,
        token: Callable[[], str],
        client: httpx2.AsyncClient,
        *,
        phone_number_id: str,
        api_version: str,
    ) -> None:
        self._token = token
        self._client = client
        self._url = f"{GRAPH}/{api_version}/{phone_number_id}/messages"

    async def send(self, user_id: str, text: str, kind: str = "reply") -> SendOutcome:
        body = _template_body(user_id, text) if kind == TEMPLATE else _text_body(user_id, text)
        for _ in range(CONNECT_RETRIES + 1):
            try:
                response = await self._client.post(
                    self._url, json=body, headers={"Authorization": f"Bearer {self._token()}"}
                )
            except (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout):
                continue  # the request never left: it is safe to try again
            except httpx2.HTTPError:
                return SendOutcome(SendStatus.UNCERTAIN, reason="transport")
            return _outcome(response)
        return SendOutcome(SendStatus.FAILED, reason="connect")


def _text_body(user_id: str, text: str) -> dict[str, Any]:
    return {**_to(user_id), "type": "text", "text": {"preview_url": False, "body": text}}


def _template_body(user_id: str, text: str) -> dict[str, Any]:
    template = Template.from_text(text)
    body = {"type": "body", "parameters": [{"type": "text", "text": p} for p in template.params]}
    buttons = [
        {
            "type": "button",
            "sub_type": "quick_reply",
            "index": str(index),
            "parameters": [{"type": "payload", "payload": payload}],
        }
        for index, payload in enumerate(template.payloads)
    ]
    return {
        **_to(user_id),
        "type": "template",
        "template": {
            "name": template.name,
            "language": {"code": template.language},
            "components": [body, *buttons],
        },
    }


def _to(user_id: str) -> dict[str, str]:
    return {"messaging_product": "whatsapp", "recipient_type": "individual", "to": user_id}


def _outcome(response: httpx2.Response) -> SendOutcome:
    answer = _json(response)
    if response.status_code == 200:
        messages = answer.get("messages")
        first = messages[0] if isinstance(messages, list) and messages else None
        mid = _string(first, "id")
        if mid:
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
