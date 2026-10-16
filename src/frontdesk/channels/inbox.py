"""The channel inbox (ADR-0011, M4 design §4): Postgres keeps every message a person sends, once,
and the conversation it belongs to, until a turn reads it.

`accept` runs in the webhook, in one transaction per delivery:
- a message already stored (Meta retries a delivery for up to 36 hours) is not stored again;
- an echo of the bot's own message is recognised by its ID, or by its text when it comes before
  the send returns (D6);
- an echo of the staff's pauses the conversation until 24 hours pass without messages (D6);
- in testers mode, the message of someone who is not a tester is dropped before anything is
  stored, unless it is the pairing code (D4).

`sweep` and `purge` run on Cloud Scheduler: the first gives back lost work, the second keeps what
the design keeps (D8).
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

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.db.engine import business_session
from frontdesk.db.models import ChannelConversation, ChannelMessage, ChannelTester
from frontdesk.observability.logs import log_alert, log_event

logger = logging.getLogger(__name__)

IN: Final = "in"
OUT: Final = "out"
BURST: Final = timedelta(seconds=3)  # D12: a turn starts 3 seconds after the last message
PAUSE: Final = timedelta(hours=24)  # D6: the staff keeps the thread until 24 hours of silence
ECHO_WINDOW: Final = timedelta(minutes=2)  # D6: an echo can come before the send returns
FORGET_STATE: Final = timedelta(days=7)  # D8: after a week of silence the conversation starts over
STALE: Final = timedelta(minutes=2)  # work older than this lost its task, a send its answer
KEEP_IDS: Final = timedelta(days=7)  # D8: Meta retries for 36 hours
KEEP_SENT_TEXT: Final = timedelta(days=1)  # D8: the text of a reply, for its echo
FORGET_CONVERSATION: Final = timedelta(days=30)  # D8
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


@dataclass(frozen=True)
class Swept:
    work: tuple[str, ...]  # conversations to give back to the worker
    uncertain: int  # sends that never came back


@dataclass(frozen=True)
class Purged:
    messages: int
    conversations: int


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

    def sweep(self) -> Swept:
        """Every 10 minutes (ADR-0011): a send that never came back becomes uncertain and is never
        tried again (D9); a conversation whose work lost its task is given back."""
        now = self._now()
        with business_session(self._factory, self._business) as session:
            mine = (
                ChannelMessage.business_id == self._business,
                ChannelMessage.channel == self._channel,
            )
            stuck = session.scalars(
                update(ChannelMessage)
                .where(*mine, ChannelMessage.direction == OUT)
                .where(ChannelMessage.status == OutStatus.SENDING)
                .where(ChannelMessage.updated_at < now - STALE)
                .values(status=OutStatus.UNCERTAIN, reason="no_answer", updated_at=now)
                .returning(ChannelMessage.conversation_id)
            ).all()
            if stuck:
                # The rest of a reply whose part may never have arrived is never sent, as inside a
                # run (final review of M4, M5).
                session.execute(
                    update(ChannelMessage)
                    .where(*mine, ChannelMessage.direction == OUT)
                    .where(ChannelMessage.status == OutStatus.QUEUED)
                    .where(ChannelMessage.conversation_id.in_(stuck))
                    .values(status=OutStatus.DROPPED, reason="previous_uncertain", updated_at=now)
                )
            waiting = session.scalars(
                select(ChannelMessage.conversation_id)
                .distinct()
                .where(*mine, ChannelMessage.created_at < now - STALE)
                .where(
                    or_(
                        and_(
                            ChannelMessage.direction == IN,
                            ChannelMessage.status == InStatus.PENDING,
                        ),
                        and_(
                            ChannelMessage.direction == OUT,
                            ChannelMessage.status == OutStatus.QUEUED,
                        ),
                    )
                )
            ).all()
        for key in stuck:
            log_alert(
                logger,
                "channel.send_uncertain",
                channel=self._channel,
                conversation_id=key,
                reason="no_answer",
            )
        return Swept(work=tuple(sorted(set(waiting))), uncertain=len(stuck))

    def purge(self) -> Purged:
        """Every night: what the design keeps, and no longer (D8)."""
        now = self._now()
        day_old = now - KEEP_SENT_TEXT
        with business_session(self._factory, self._business) as session:
            mine = (
                ChannelMessage.business_id == self._business,
                ChannelMessage.channel == self._channel,
            )
            session.execute(
                update(ChannelMessage)
                .where(*mine, ChannelMessage.direction == OUT)
                .where(ChannelMessage.status == OutStatus.QUEUED)
                .where(ChannelMessage.created_at < day_old)
                .values(status=OutStatus.DROPPED, reason="expired", updated_at=now)
            )
            session.execute(
                update(ChannelMessage)
                .where(*mine, ChannelMessage.direction == OUT)
                .where(ChannelMessage.text.is_not(None))
                .where(ChannelMessage.created_at < day_old)
                .values(text=None)
            )
            session.execute(
                update(ChannelMessage)
                .where(*mine, ChannelMessage.direction == IN)
                .where(ChannelMessage.status == InStatus.PENDING)
                .where(ChannelMessage.created_at < day_old)
                .values(status=InStatus.FAILED, reason="expired", text=None, updated_at=now)
            )
            messages = session.scalars(
                delete(ChannelMessage)
                .where(*mine, ChannelMessage.created_at < now - KEEP_IDS)
                .returning(ChannelMessage.id)
            ).all()
            talks = (
                ChannelConversation.business_id == self._business,
                ChannelConversation.channel == self._channel,
            )
            session.execute(
                update(ChannelConversation)
                .where(*talks, ChannelConversation.state.is_not(None))
                .where(ChannelConversation.last_message_at < now - FORGET_STATE)
                .values(state=None)
            )
            gone = session.scalars(
                delete(ChannelConversation)
                .where(*talks, ChannelConversation.last_message_at < now - FORGET_CONVERSATION)
                .returning(ChannelConversation.conversation_id)
            ).all()
        log_event(logger, "channel.purged", channel=self._channel, count=len(messages) + len(gone))
        return Purged(messages=len(messages), conversations=len(gone))

    def _is_tester(self, session: Session, user_id: str) -> bool:
        return session.get(ChannelTester, (self._business, self._channel, user_id)) is not None

    def _is_code(self, event: Inbound) -> bool:
        """The message is the pairing code (D4), a secret."""
        if self._pairing is None or event.kind is not InboundKind.TEXT or event.text is None:
            return False
        return hmac.compare_digest(event.text.strip().encode(), self._pairing.encode())

    def _pair(self, session: Session, event: Inbound, now: datetime) -> str | None:
        """Someone who is not a tester: dropped, unless the message is the pairing code (D4)."""
        if not self._is_code(event):
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
        if self._is_code(event):
            # A tester who sends the code again: stored as read with no text, like the first
            # time, never a turn (final review of M4, M7).
            self._insert(session, key, event, now, status=InStatus.READ, text=None)
            return None
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
            # An uncertain send may have been delivered: its echo is the bot's too.
            .where(
                ChannelMessage.status.in_([OutStatus.SENDING, OutStatus.SENT, OutStatus.UNCERTAIN])
            )
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
