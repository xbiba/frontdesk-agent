"""The chat on the business's site (design of 2026-11-07, ADR-0010: one adapter more).

The widget posts a message with its session and its own message ID; the webhook stores it in the
inbox like an Instagram message, and the worker runs the turn. Nobody is called to deliver the
reply: the web Sender leaves it in the inbox, and the widget reads it from there. The session is a
random ID the widget keeps in the tab's sessionStorage: no cookie, no login, no IP address kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import IN, OUT, InStatus, OutStatus, conversation_key
from frontdesk.channels.messages import Inbound, InboundKind, SendOutcome, SendStatus
from frontdesk.db.engine import business_session
from frontdesk.db.models import ChannelMessage

WEB: Final = "web"
# Design §3: per session, the identical message in a row, the channel's model turns of a day.
MESSAGES_PER_SESSION: Final = 30
REPEATS: Final = 2  # the third identical message in a row gets the handoff text
TURNS_A_DAY: Final = 300
MAX_TEXT_CHARS: Final = 2000  # the engine's (the widget stops at 500)
MAX_REPLY_BYTES: Final = 16_000  # a reply is never split: it is read, not sent
_SESSION: Final = re.compile(r"^[A-Za-z0-9-]{16,64}$")
_MESSAGE_ID: Final = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def valid_session(value: str) -> bool:
    return _SESSION.fullmatch(value) is not None


def valid_message_id(value: str) -> bool:
    return _MESSAGE_ID.fullmatch(value) is not None


def inbound(account_id: str, session: str, message_id: str, text: str) -> Inbound:
    """The widget's message as the inbox reads it. Its ID is the session's: a retry of the same
    message is stored once, and two sessions never share an ID."""
    return Inbound(
        account_id=account_id,
        user_id=session,
        kind=InboundKind.TEXT,
        message_id=f"{session}:{message_id}",
        text=text,
    )


def cursor_of(moment: datetime) -> int:
    """A reply's place in the conversation, in microseconds: the widget asks for what follows."""
    delta = moment - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class WebSender:
    """The reply is already in the inbox, recorded before this send: it is ready to be read."""

    name = WEB
    max_message_bytes = MAX_REPLY_BYTES

    async def send(self, user_id: str, text: str, kind: str = "reply") -> SendOutcome:
        return SendOutcome(SendStatus.SENT)


@dataclass(frozen=True)
class Reply:
    cursor: int
    text: str


@dataclass(frozen=True)
class Replies:
    replies: tuple[Reply, ...]
    waiting: bool  # a message still waits for its turn, or a reply to be ready


class WebReplies:
    """The replies of a session ready to be read. A reply's text is kept a day (D8)."""

    def __init__(self, session_factory: sessionmaker[Session], *, business_id: str) -> None:
        self._factory = session_factory
        self._business = business_id

    def after(self, session: str, cursor: int) -> Replies:
        key = conversation_key(self._business, WEB, session)
        mine = (
            ChannelMessage.business_id == self._business,
            ChannelMessage.channel == WEB,
            ChannelMessage.conversation_id == key,
        )
        with business_session(self._factory, self._business) as db:
            rows = db.execute(
                select(ChannelMessage.created_at, ChannelMessage.text)
                .where(*mine, ChannelMessage.direction == OUT)
                .where(ChannelMessage.status == OutStatus.SENT)
                .where(ChannelMessage.text.is_not(None))
                .where(ChannelMessage.created_at > _EPOCH + timedelta(microseconds=cursor))
                .order_by(ChannelMessage.created_at)
            ).all()
            waiting = db.scalar(
                select(ChannelMessage.id)
                .where(*mine)
                .where(
                    or_(
                        and_(
                            ChannelMessage.direction == IN,
                            ChannelMessage.status == InStatus.PENDING,
                        ),
                        and_(
                            ChannelMessage.direction == OUT,
                            ChannelMessage.status.in_([OutStatus.QUEUED, OutStatus.SENDING]),
                        ),
                    )
                )
                .limit(1)
            )
        replies = tuple(Reply(cursor_of(created), text) for created, text in rows if text)
        return Replies(replies=replies, waiting=waiting is not None)
