"""The Gmail adapter (design §3): the Gmail API with a token for the business's mailbox, through
domain-wide delegation in production or a person's OAuth token in development
(frontdesk.model_gateway.mail_tokens).

The one scope is gmail.modify: read, create drafts and labels. It cannot send, and this client
refuses any request to a path that sends. No message content is logged, and an error carries a
status, never a response body.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from email.message import EmailMessage
from html.parser import HTMLParser
from typing import Any, Final
from urllib.parse import quote

import httpx2

from frontdesk.email.mailbox import Draft, MailMessage, MailThread
from frontdesk.model_gateway.factory import tls_context
from frontdesk.model_gateway.google_tokens import TokenSource

API: Final = "https://gmail.googleapis.com/gmail/v1/users/me"
SCOPES: Final = ("https://www.googleapis.com/auth/gmail.modify",)
TIMEOUT_SECONDS: Final = 15.0
_SCREENED_HEADERS: Final = frozenset({"list-unsubscribe", "auto-submitted", "precedence"})
MAX_BODY_CHARS: Final = 50_000  # what is read of a body at most; the pass reads 6,000 of it
MAX_DECODED_CHARS: Final = 4 * MAX_BODY_CHARS  # what all the parts of a message decode at most


class MailboxError(RuntimeError):
    """Gmail refused or failed a request; the message names it and its status."""


def gmail_http_client() -> httpx2.Client:
    """No proxy from the environment and certifi's certificates, as the model gateway's clients."""
    return httpx2.Client(
        trust_env=False, verify=tls_context(), timeout=httpx2.Timeout(TIMEOUT_SECONDS, connect=5.0)
    )


class GmailMailbox:
    def __init__(self, token: TokenSource, client: httpx2.Client) -> None:
        self._token = token
        self._client = client
        self._labels: dict[str, str] | None = None

    def thread_ids(self, query: str, limit: int) -> list[str]:
        data = self._call("GET", "/threads", params={"q": query, "maxResults": str(limit)})
        return [str(item["id"]) for item in data.get("threads", [])][:limit]

    def thread(self, thread_id: str) -> MailThread:
        data = self._call("GET", f"/threads/{quote(thread_id, safe='')}", params={"format": "full"})
        return MailThread(
            id=str(data["id"]), messages=tuple(_message(item) for item in data.get("messages", []))
        )

    def create_draft(self, draft: Draft) -> str:
        raw = base64.urlsafe_b64encode(mime(draft)).decode("ascii")
        data = self._call(
            "POST", "/drafts", json={"message": {"threadId": draft.thread_id, "raw": raw}}
        )
        return str(data.get("id", ""))

    def add_labels(self, thread_id: str, names: Sequence[str]) -> None:
        ids = [self._label_id(name) for name in names]
        self._call(
            "POST", f"/threads/{quote(thread_id, safe='')}/modify", json={"addLabelIds": ids}
        )

    def _label_id(self, name: str) -> str:
        if self._labels is None:
            data = self._call("GET", "/labels")
            self._labels = {str(item["name"]): str(item["id"]) for item in data.get("labels", [])}
        if name not in self._labels:
            made = self._call(
                "POST",
                "/labels",
                json={
                    "name": name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            self._labels[name] = str(made["id"])
        return self._labels[name]

    def _call(self, method: str, path: str, **options: Any) -> dict[str, Any]:
        if re.search(r"(?:^|/)send(?:$|[/?])|:send\b", path):
            raise MailboxError("refused: the email drafts never send")  # no request leaves
        try:
            response = self._client.request(
                method,
                f"{API}{path}",
                headers={"Authorization": f"Bearer {self._token()}"},
                **options,
            )
        except httpx2.HTTPError as error:
            raise MailboxError(
                f"gmail {method} {path.split('/')[1]}: {type(error).__name__}"
            ) from None
        if response.status_code != 200:
            raise MailboxError(
                f"gmail {method} {path.split('/')[1]}: status {response.status_code}"
            )
        try:
            data = response.json()
        except ValueError:
            raise MailboxError(f"gmail {method}: malformed answer") from None
        return data if isinstance(data, dict) else {}


def mime(draft: Draft) -> bytes:
    """The reply as an RFC 5322 message, in its conversation's thread."""
    message = EmailMessage()
    message["From"] = draft.sender
    message["To"] = draft.to
    message["Subject"] = draft.subject
    if draft.in_reply_to:
        message["In-Reply-To"] = draft.in_reply_to
    if draft.references:
        message["References"] = draft.references
    message.set_content(draft.body)
    return message.as_bytes()


def _message(item: Mapping[str, Any]) -> MailMessage:
    payload = item.get("payload", {})
    headers = {
        str(h.get("name", "")).lower(): str(h.get("value", "")) for h in payload.get("headers", [])
    }
    plain: list[str] = []
    rich: list[str] = []
    attachments = 0
    left = MAX_DECODED_CHARS  # what all the parts of a message may decode, together
    for part in _parts(payload):
        if part.get("filename"):
            attachments += 1
            continue
        kind = part.get("mimeType")
        if kind not in {"text/plain", "text/html"} or left <= 0:
            continue
        text = _decode(part, left)  # capped before anything parses it (security review)
        left -= len(text)
        if kind == "text/plain":
            plain.append(text)
        else:
            rich.append(_strip_html(text))
    internal = item.get("internalDate")
    return MailMessage(
        id=str(item["id"]),
        sender=headers.get("from", ""),
        subject=headers.get("subject", ""),
        body=("\n".join(plain) if plain else "\n".join(rich)).strip()[:MAX_BODY_CHARS],
        message_id=headers.get("message-id", ""),
        references=headers.get("references", ""),
        reply_to=headers.get("reply-to", ""),
        sent_at=datetime.fromtimestamp(int(internal) / 1000, UTC) if internal else None,
        headers={name: value for name, value in headers.items() if name in _SCREENED_HEADERS},
        labels=frozenset(str(label) for label in item.get("labelIds", [])),
        attachments=attachments,
    )


def _parts(part: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield part
    for child in part.get("parts", []) or []:
        yield from _parts(child)


def _decode(part: Mapping[str, Any], limit: int) -> str:
    """At most `limit` characters of the part's text: only as much base64 as that needs is
    decoded."""
    data = str(part.get("body", {}).get("data", ""))[: (limit // 3 + 1) * 4]
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except ValueError:
        return ""
    return raw.decode("utf-8", errors="replace")[:limit]


class _Text(HTMLParser):
    """The text of an HTML part, without its scripts and styles: one linear pass."""

    _SKIPPED: Final = frozenset({"script", "style", "head", "title"})
    _BREAKS: Final = frozenset({"br", "p", "div", "li", "tr"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIPPED:
            self._skipping += 1
        elif tag in self._BREAKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def _strip_html(text: str) -> str:
    parser = _Text()
    parser.feed(text)
    parser.close()
    return re.sub(r"[ \t]+", " ", "".join(parser.parts))
