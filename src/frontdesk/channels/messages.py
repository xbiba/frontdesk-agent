"""What a channel's adapter gives the inbox, and what it does for it (ADR-0010)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class InboundKind(StrEnum):
    TEXT = "text"  # a message with text, story replies included
    ATTACHMENT = "attachment"  # a photo, video, audio or file: never opened (M4 design D11)
    ECHO = "echo"  # a message the business account sent: the bot's or the staff's (D6)
    DELETED = "deleted"  # the person unsent a message
    IGNORED = "ignored"  # reactions, reads, story mentions, shared posts
    BUTTON = "button"  # a tap on a business button: the text is its payload, never a model's


@dataclass(frozen=True)
class Inbound:
    """One event of a webhook delivery. `user_id` is the person on the other side of the thread:
    the sender of a message, the recipient of an echo."""

    account_id: str
    user_id: str
    kind: InboundKind
    message_id: str | None = None
    text: str | None = None
    sent_at: datetime | None = None


class SendStatus(StrEnum):
    SENT = "sent"
    FAILED = "failed"  # refused before delivery: it was not delivered
    UNCERTAIN = "uncertain"  # it may have been delivered: never sent again (D9)


@dataclass(frozen=True)
class SendOutcome:
    status: SendStatus
    message_id: str | None = None
    reason: str | None = None  # a code for the logs, never the provider's message


class WebhookReader(Protocol):
    """The receiving side of a channel: it checks a delivery's signature and reads its events."""

    name: str

    def signature_ok(self, body: bytes, signature: str | None) -> bool: ...

    def parse(self, body: bytes) -> list[Inbound]: ...


class Sender(Protocol):
    """The sending side of a channel. It never composes or checks a reply: the engine does."""

    name: str
    max_message_bytes: int

    async def send(self, user_id: str, text: str, kind: str = "reply") -> SendOutcome:
        """`kind` is the outbox part's: a channel with templates sends a "template" part as one."""
        ...
