"""The channel inbox (ADR-0011, M4 design §4): Postgres keeps every message a person sends, once,
and the conversation it belongs to, until a turn reads it.

`accept` runs in the webhook, in one transaction per delivery:
- a message already stored (Meta retries a delivery for up to 36 hours) is not stored again;
- an echo of the bot's own message is recognised by its ID, or by its text when it comes before
  the send returns (D6);
- an echo of the staff's pauses the conversation until 24 hours pass without messages (D6);
- in testers mode, the message of someone who is not a tester is dropped before anything is
  stored, unless it is the pairing code (D4).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.db.engine import business_session
from frontdesk.db.models import ChannelConversation, ChannelMessage, ChannelTester
from frontdesk.observability.logs import log_event

logger = logging.getLogger(__name__)

IN: Final = "in"
OUT: Final = "out"
BURST: Final = timedelta(seconds=3)  # D12: a turn starts 3 seconds after the last message
PAUSE: Final = timedelta(hours=24)  # D6: the staff keeps the thread until 24 hours of silence
ECHO_WINDOW: Final = timedelta(minutes=2)  # D6: an echo can come before the send returns
FORGET_STATE: Final = timedelta(days=7)  # D8: after a week of silence the conversation starts over
TESTER_WELCOME: Final = (
    "Modalità di prova attiva: l'assistente risponde a questo account. "
    "Test mode on: the assistant answers this account."
)
_KEY: Final = re.compile(r"^[a-z]{2,16}-[0-9a-f]{40}$")


class InStatus(StrEnum):
    PENDING = "pending"
    READ = "read"
    DISCARDED = "discarded"  # unsent by the person, or written while the staff had the thread
    FAILED = "failed"  # three turns failed on it, or it waited a whole day


class OutStatus(StrEnum):
    QUEUED = "queued"  # recorded, not tried yet
    SENDING = "sending"  # handed to the channel: if nothing comes back, it becomes uncertain
    SENT = "sent"
    UNCERTAIN = "uncertain"  # it may have been delivered: never sent again (D9)
    FAILED = "failed"
    DROPPED = "dropped"  # not sent: the switch, the staff, the reply window, a failed part before


def conversation_key(business_id: str, channel: str, user_id: str) -> str:
    """The conversation's ID: the same person on the same channel always gets the same one (D8),
    and it does not say who they are."""
    digest = hashlib.sha256(f"{business_id}\x1f{channel}\x1f{user_id}".encode()).hexdigest()
    return f"{channel}-{digest[:40]}"


def valid_key(value: str) -> bool:
    return _KEY.match(value) is not None


def queue_reply(
    session: Session,
    *,
    business_id: str,
    channel: str,
    conversation_id: str,
    parts: Sequence[tuple[str, str]],
    now: datetime,
) -> None:
    """Record replies to send, in order. Each part is (kind, text), kind "reply" or "fixed". A
    reply sorts after the messages that came in at the same instant."""
    for offset, (kind, text) in enumerate(parts, start=1):
        session.add(
            ChannelMessage(
                id=uuid.uuid4(),
                business_id=business_id,
                conversation_id=conversation_id,
                channel=channel,
                direction=OUT,
                kind=kind,
                text=text,
                status=OutStatus.QUEUED,
                attempts=0,
                created_at=now + timedelta(microseconds=offset),
                updated_at=now,
            )
        )


def _normal(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Accepted:
    """The conversations a delivery gave work to: the worker is called for each."""

    work: tuple[str, ...] = ()


class Inbox:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        business_id: str,
        channel: str,
        account_id: str,
        testers_only: bool,
        pairing_code: str | None,
        now: Callable[[], datetime],
    ) -> None:
        self._factory = session_factory
        self._business = business_id
        self._channel = channel
        self._account = account_id
        self._testers_only = testers_only
        self._pairing = pairing_code
        self._now = now

    def accept(self, events: Sequence[Inbound]) -> Accepted:
        now = self._now()
        work: dict[str, None] = {}
        others: set[str] = set()
        strangers = 0
        with business_session(self._factory, self._business) as session:
            for event in events:
                if event.account_id != self._account:
                    others.add(event.account_id)
                    continue
                if not event.user_id or event.kind is InboundKind.IGNORED:
                    continue
                if event.kind is InboundKind.ECHO:
                    self._echo(session, event, now)
                    continue
                if self._testers_only and not self._is_tester(session, event.user_id):
                    paired = self._pair(session, event, now)
                    if paired is None:
                        strangers += 1
                    else:
                        work[paired] = None
                    continue
                if event.kind is InboundKind.DELETED:
                    self._unsent(session, event, now)
                    continue
                key = self._message(session, event, now)
                if key is not None:
                    work[key] = None
        for account in sorted(others):
            log_event(logger, "channel.other_account", channel=self._channel, account_id=account)
        if strangers:
            log_event(logger, "channel.not_a_tester", channel=self._channel, count=strangers)
        return Accepted(tuple(work))

    def count_testers(self) -> int:
        with business_session(self._factory, self._business) as session:
            count = session.scalar(
                select(func.count())
                .select_from(ChannelTester)
                .where(ChannelTester.business_id == self._business)
                .where(ChannelTester.channel == self._channel)
            )
        return count or 0

    def clear_testers(self) -> int:
        with business_session(self._factory, self._business) as session:
            removed = session.scalars(
                delete(ChannelTester)
                .where(ChannelTester.business_id == self._business)
                .where(ChannelTester.channel == self._channel)
                .returning(ChannelTester.user_id)
            ).all()
        return len(removed)

    def _is_tester(self, session: Session, user_id: str) -> bool:
        return session.get(ChannelTester, (self._business, self._channel, user_id)) is not None

    def _pair(self, session: Session, event: Inbound, now: datetime) -> str | None:
        """Someone who is not a tester: dropped, unless the message is the pairing code (D4)."""
        if self._pairing is None or event.kind is not InboundKind.TEXT or event.text is None:
            return None
        if not hmac.compare_digest(event.text.strip().encode(), self._pairing.encode()):
            return None
        session.execute(
            insert(ChannelTester)
            .values(
                business_id=self._business,
                channel=self._channel,
                user_id=event.user_id,
                created_at=now,
            )
            .on_conflict_do_nothing()
        )
        key, _ = self._open(session, event.user_id, now)
        # The code is never a turn: it is stored as read, so a delivery of it again finds it.
        self._insert(session, key, event, now, status=InStatus.READ, text=None)
        queue_reply(
            session,
            business_id=self._business,
            channel=self._channel,
            conversation_id=key,
            parts=[("fixed", TESTER_WELCOME)],
            now=now,
        )
        log_event(logger, "channel.tester_added", channel=self._channel, conversation_id=key)
        return key

    def _message(self, session: Session, event: Inbound, now: datetime) -> str | None:
        key, paused = self._open(session, event.user_id, now)
        if paused is not None and paused > now:
            self._touch(session, key, now, pause=True)  # D6: 24 hours without messages
            return None  # the staff has the thread: nothing stored, nothing for the model
        text = event.text if event.kind is InboundKind.TEXT else None
        if self._insert(session, key, event, now, status=InStatus.PENDING, text=text):
            self._touch(session, key, now)
            return key
        waiting = session.scalar(
            select(ChannelMessage.status)
            .where(ChannelMessage.business_id == self._business)
            .where(ChannelMessage.channel == self._channel)
            .where(ChannelMessage.external_id == event.message_id)
        )
        return key if waiting == InStatus.PENDING else None

    def _unsent(self, session: Session, event: Inbound, now: datetime) -> None:
        if event.message_id is None:
            return
        session.execute(
            update(ChannelMessage)
            .where(ChannelMessage.business_id == self._business)
            .where(ChannelMessage.channel == self._channel)
            .where(ChannelMessage.external_id == event.message_id)
            .where(ChannelMessage.direction == IN)
            .where(ChannelMessage.status == InStatus.PENDING)
            .values(status=InStatus.DISCARDED, text=None, updated_at=now)
        )

    def _echo(self, session: Session, event: Inbound, now: datetime) -> None:
        key = conversation_key(self._business, self._channel, event.user_id)
        known = session.scalar(
            select(ChannelConversation.conversation_id)
            .where(ChannelConversation.business_id == self._business)
            .where(ChannelConversation.conversation_id == key)
        )
        if known is None or self._ours(session, key, event, now):
            return  # a thread the bot is not in, or the bot's own message
        self._touch(session, key, now, pause=True)
        log_event(logger, "channel.staff_reply", channel=self._channel, conversation_id=key)

    def _ours(self, session: Session, key: str, event: Inbound, now: datetime) -> bool:
        if event.message_id is not None:
            sent = session.scalar(
                select(ChannelMessage.id)
                .where(ChannelMessage.business_id == self._business)
                .where(ChannelMessage.channel == self._channel)
                .where(ChannelMessage.external_id == event.message_id)
                .where(ChannelMessage.direction == OUT)
            )
            if sent is not None:
                return True
        if event.text is None:
            return False
        recent = session.scalars(
            select(ChannelMessage.text)
            .where(ChannelMessage.business_id == self._business)
            .where(ChannelMessage.conversation_id == key)
            .where(ChannelMessage.direction == OUT)
            .where(ChannelMessage.status.in_([OutStatus.SENDING, OutStatus.SENT]))
            .where(ChannelMessage.updated_at >= now - ECHO_WINDOW)
        ).all()
        echoed = _normal(event.text)
        return any(text is not None and _normal(text) == echoed for text in recent)

    def _open(self, session: Session, user_id: str, now: datetime) -> tuple[str, datetime | None]:
        """The person's conversation, created with their first message: its key and its pause.
        The row stays locked until the delivery's transaction ends."""
        key = conversation_key(self._business, self._channel, user_id)
        session.execute(
            insert(ChannelConversation)
            .values(
                business_id=self._business,
                conversation_id=key,
                channel=self._channel,
                user_id=user_id,
                last_message_at=now,
            )
            .on_conflict_do_nothing(index_elements=["business_id", "conversation_id"])
        )
        paused = session.scalar(
            select(ChannelConversation.paused_until)
            .where(ChannelConversation.business_id == self._business)
            .where(ChannelConversation.conversation_id == key)
            .with_for_update()
        )
        return key, paused

    def _touch(self, session: Session, key: str, now: datetime, *, pause: bool = False) -> None:
        values: dict[str, datetime] = {"last_message_at": now}
        if pause:
            values["paused_until"] = now + PAUSE
        session.execute(
            update(ChannelConversation)
            .where(ChannelConversation.business_id == self._business)
            .where(ChannelConversation.conversation_id == key)
            .values(**values)
        )

    def _insert(
        self,
        session: Session,
        key: str,
        event: Inbound,
        now: datetime,
        *,
        status: InStatus,
        text: str | None,
    ) -> bool:
        stored = session.execute(
            insert(ChannelMessage)
            .values(
                id=uuid.uuid4(),
                business_id=self._business,
                conversation_id=key,
                channel=self._channel,
                direction=IN,
                kind=event.kind.value,
                external_id=event.message_id,
                text=text,
                status=status.value,
                attempts=0,
                sent_at=event.sent_at,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["business_id", "channel", "external_id"])
            .returning(ChannelMessage.id)
        ).first()
        return stored is not None
