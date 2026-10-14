"""The worker's side of the inbox (M4 design §4 steps 4-8): one turn of a conversation, then its
replies, each sent at most once (D9).

A conversation runs under a lease, one turn at a time. Messages younger than the burst window wait
for the task of the newest one (D12). The attempt is counted before the turn: a turn that fails
three times gives the person reception's contact instead of silence. Every reply is recorded before
it is sent, and a reply that went out without an answer is never sent again.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Final

from pydantic import ValidationError
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import (
    BURST,
    FORGET_STATE,
    IN,
    OUT,
    InStatus,
    OutStatus,
    queue_reply,
)
from frontdesk.channels.messages import InboundKind, Sender, SendStatus
from frontdesk.channels.split import split_message
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import RATE_WINDOW, TurnResult
from frontdesk.db.engine import business_session
from frontdesk.db.models import ChannelConversation, ChannelMessage
from frontdesk.observability.logs import log_alert, log_event
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey

logger = logging.getLogger(__name__)

type Turn = Callable[[str, ConversationState], Awaitable[tuple[TurnResult, ConversationState]]]

LEASE: Final = timedelta(seconds=120)  # longer than a turn (45 s) and its sends
MAX_ATTEMPTS: Final = 3  # spec §4 step 8
MAX_REPLIES_A_DAY: Final = 60  # D10
REPLY_WINDOW: Final = timedelta(hours=24)  # Meta's window after the person's last message
RATE_LIMIT_RULE: Final = "rate_limit"  # the engine's rule past 10 messages a minute
_LIVE: Final = (OutStatus.QUEUED, OutStatus.SENDING, OutStatus.SENT, OutStatus.UNCERTAIN)


class Outcome(StrEnum):
    DONE = "done"  # a turn ran, or fixed texts went out
    IDLE = "idle"  # nothing waited
    WAIT = "wait"  # a message is younger than the burst window: its own task comes later
    BUSY = "busy"  # another worker holds the conversation: Cloud Tasks tries again
    PAUSED = "paused"  # the staff has the thread
    FAILED = "failed"  # three turns failed: the handoff text went out


@dataclass(frozen=True)
class _Conversation:
    user_id: str
    state: dict[str, Any] | None
    active_at: datetime | None
    paused_until: datetime | None
    slowed_at: datetime | None


@dataclass(frozen=True)
class _Pending:
    id: uuid.UUID
    kind: str
    text: str | None
    created_at: datetime
    attempts: int


def _mark(session: Session, pending: Sequence[_Pending], status: InStatus, now: datetime) -> None:
    session.execute(
        update(ChannelMessage)
        .where(ChannelMessage.id.in_([message.id for message in pending]))
        .values(status=status, text=None, updated_at=now)
    )


class Runner:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        pack: BusinessPack,
        channel: str,
        sender: Sender,
        turn: Turn,
        now: Callable[[], datetime],
        off: bool = False,
    ) -> None:
        self._factory = session_factory
        self._pack = pack
        self._channel = channel
        self._sender = sender
        self._turn = turn
        self._now = now
        self._off = off

    async def process(self, conversation_id: str) -> Outcome:
        """One run for a conversation: its waiting messages become a turn, then every reply
        recorded and not yet tried goes out."""
        claimed = self._claim(conversation_id, self._now())
        if isinstance(claimed, Outcome):
            return claimed
        try:
            outcome = await self._run(conversation_id, claimed, self._now())
            await self._deliver(conversation_id, claimed.user_id)
        finally:
            self._release(conversation_id)
        log_event(
            logger,
            "channel.processed",
            channel=self._channel,
            conversation_id=conversation_id,
            outcome=outcome,
        )
        return outcome

    def _session(self) -> AbstractContextManager[Session]:
        return business_session(self._factory, self._pack.id)

    def _claim(self, key: str, now: datetime) -> _Conversation | Outcome:
        with self._session() as session:
            row = session.execute(
                update(ChannelConversation)
                .where(ChannelConversation.business_id == self._pack.id)
                .where(ChannelConversation.conversation_id == key)
                .where(
                    or_(
                        ChannelConversation.lease_until.is_(None),
                        ChannelConversation.lease_until < now,
                    )
                )
                .values(lease_until=now + LEASE)
                .returning(
                    ChannelConversation.user_id,
                    ChannelConversation.state,
                    ChannelConversation.active_at,
                    ChannelConversation.paused_until,
                    ChannelConversation.slowed_at,
                )
            ).first()
            if row is not None:
                return _Conversation(*row)
            exists = session.get(ChannelConversation, (self._pack.id, key)) is not None
        return Outcome.BUSY if exists else Outcome.IDLE

    def _release(self, key: str) -> None:
        with self._session() as session:
            session.execute(
                update(ChannelConversation)
                .where(ChannelConversation.business_id == self._pack.id)
                .where(ChannelConversation.conversation_id == key)
                .values(lease_until=None)
            )

    async def _run(self, key: str, conversation: _Conversation, now: datetime) -> Outcome:
        pending = self._pending(key)
        if not pending:
            return Outcome.IDLE
        if now - max(message.created_at for message in pending) < BURST:
            return Outcome.WAIT
        if conversation.paused_until is not None and conversation.paused_until > now:
            with self._session() as session:
                _mark(session, pending, InStatus.DISCARDED, now)
            return Outcome.PAUSED
        if max(message.attempts for message in pending) >= MAX_ATTEMPTS:
            self._give_up(key, pending, conversation, now)
            return Outcome.FAILED
        self._count_attempt(pending, now)  # committed first: a turn that crashes still counts
        state = self._state(key, conversation, now)
        texts = [m.text for m in pending if m.kind == InboundKind.TEXT and m.text]
        replies: list[tuple[str, str]] = []
        slowed = conversation.slowed_at
        following = state
        if texts:
            result, following = await self._turn("\n".join(texts), state)
            if result.trace.rule != RATE_LIMIT_RULE:
                replies.append(("reply", result.text))
            elif slowed is None or now - slowed >= RATE_WINDOW:
                replies.append(("reply", result.text))  # D10: one slow-down a minute, then quiet
                slowed = now
        if any(m.kind == InboundKind.ATTACHMENT for m in pending):
            replies.append(("fixed", self._fixed(following, FixedTextKey.ATTACHMENT)))
        if replies and self._replies_today(key, now) >= MAX_REPLIES_A_DAY:
            log_alert(
                logger,
                "channel.reply_cap",
                channel=self._channel,
                conversation_id=key,
                count=MAX_REPLIES_A_DAY,
            )
            replies = []
        self._finish(key, pending, following, replies, slowed, now)
        return Outcome.DONE

    def _pending(self, key: str) -> list[_Pending]:
        with self._session() as session:
            rows = session.execute(
                select(
                    ChannelMessage.id,
                    ChannelMessage.kind,
                    ChannelMessage.text,
                    ChannelMessage.created_at,
                    ChannelMessage.attempts,
                )
                .where(ChannelMessage.business_id == self._pack.id)
                .where(ChannelMessage.conversation_id == key)
                .where(ChannelMessage.direction == IN)
                .where(ChannelMessage.status == InStatus.PENDING)
                .order_by(
                    ChannelMessage.sent_at.asc().nulls_last(),
                    ChannelMessage.created_at,
                    ChannelMessage.id,
                )
            ).all()
        return [_Pending(*row) for row in rows]

    def _state(self, key: str, conversation: _Conversation, now: datetime) -> ConversationState:
        """The engine's state, or a new one after the staff's pause or a week of silence (D6,
        D8): the new conversation starts with the line that says who answers."""
        fresh = ConversationState(conversation_id=key)
        if conversation.state is None or conversation.active_at is None:
            return fresh
        if conversation.paused_until is not None or now - conversation.active_at > FORGET_STATE:
            return fresh
        try:
            return ConversationState.model_validate(conversation.state)
        except ValidationError:
            log_event(logger, "channel.state_reset", channel=self._channel, conversation_id=key)
            return fresh

    def _fixed(self, state: ConversationState, text: FixedTextKey) -> str:
        return self._pack.fixed_texts[state.language or self._pack.default_language][text]

    def _count_attempt(self, pending: Sequence[_Pending], now: datetime) -> None:
        with self._session() as session:
            session.execute(
                update(ChannelMessage)
                .where(ChannelMessage.id.in_([message.id for message in pending]))
                .values(attempts=ChannelMessage.attempts + 1, updated_at=now)
            )

    def _replies_today(self, key: str, now: datetime) -> int:
        with self._session() as session:
            count = session.scalar(
                select(func.count())
                .select_from(ChannelMessage)
                .where(ChannelMessage.business_id == self._pack.id)
                .where(ChannelMessage.conversation_id == key)
                .where(ChannelMessage.direction == OUT)
                .where(ChannelMessage.status.in_(_LIVE))
                .where(ChannelMessage.created_at >= now - timedelta(days=1))
            )
        return count or 0

    def _finish(
        self,
        key: str,
        pending: Sequence[_Pending],
        state: ConversationState,
        replies: Sequence[tuple[str, str]],
        slowed: datetime | None,
        now: datetime,
    ) -> None:
        size = self._sender.max_message_bytes
        parts = [(kind, part) for kind, text in replies for part in split_message(text, size)]
        with self._session() as session:
            session.execute(
                update(ChannelConversation)
                .where(ChannelConversation.business_id == self._pack.id)
                .where(ChannelConversation.conversation_id == key)
                .values(
                    state=state.model_dump(mode="json"),
                    active_at=now,
                    paused_until=None,
                    slowed_at=slowed,
                )
            )
            _mark(session, pending, InStatus.READ, now)
            queue_reply(
                session,
                business_id=self._pack.id,
                channel=self._channel,
                conversation_id=key,
                parts=parts,
                now=now,
            )

    def _give_up(
        self, key: str, pending: Sequence[_Pending], conversation: _Conversation, now: datetime
    ) -> None:
        text = self._fixed(self._state(key, conversation, now), FixedTextKey.HANDOFF)
        parts = [("fixed", part) for part in split_message(text, self._sender.max_message_bytes)]
        with self._session() as session:
            _mark(session, pending, InStatus.FAILED, now)
            queue_reply(
                session,
                business_id=self._pack.id,
                channel=self._channel,
                conversation_id=key,
                parts=parts,
                now=now,
            )
        log_alert(
            logger,
            "channel.turn_failed",
            channel=self._channel,
            conversation_id=key,
            count=len(pending),
        )

    async def _deliver(self, key: str, user_id: str) -> None:
        for message_id, text in self._queued(key):
            blocked = self._blocked(key, text)
            if blocked is not None or text is None:
                self._drop_queued(key, blocked or "expired")
                return
            self._set(message_id, status=OutStatus.SENDING)
            outcome = await self._sender.send(user_id, text)
            self._set(
                message_id,
                status=OutStatus(outcome.status.value),
                external_id=outcome.message_id,
                reason=outcome.reason,
            )
            if outcome.status is not SendStatus.SENT:
                log_alert(
                    logger,
                    f"channel.send_{outcome.status.value}",
                    channel=self._channel,
                    conversation_id=key,
                    reason=outcome.reason,
                )
                self._drop_queued(key, "previous_failed")
                return

    def _queued(self, key: str) -> list[tuple[uuid.UUID, str | None]]:
        with self._session() as session:
            rows = session.execute(
                select(ChannelMessage.id, ChannelMessage.text)
                .where(ChannelMessage.business_id == self._pack.id)
                .where(ChannelMessage.conversation_id == key)
                .where(ChannelMessage.direction == OUT)
                .where(ChannelMessage.status == OutStatus.QUEUED)
                .order_by(ChannelMessage.created_at)
            ).all()
        return [(row[0], row[1]) for row in rows]

    def _blocked(self, key: str, text: str | None) -> str | None:
        """Why a reply must not go now (D9, D13), or None."""
        if self._off:
            return "switch"
        if text is None:
            return "expired"
        now = self._now()
        with self._session() as session:
            paused = session.scalar(
                select(ChannelConversation.paused_until)
                .where(ChannelConversation.business_id == self._pack.id)
                .where(ChannelConversation.conversation_id == key)
            )
            last = session.scalar(
                select(func.max(ChannelMessage.created_at))
                .where(ChannelMessage.business_id == self._pack.id)
                .where(ChannelMessage.conversation_id == key)
                .where(ChannelMessage.direction == IN)
            )
        if paused is not None and paused > now:
            return "staff"
        if last is None or now - last > REPLY_WINDOW:
            return "window"
        return None

    def _drop_queued(self, key: str, reason: str) -> None:
        with self._session() as session:
            session.execute(
                update(ChannelMessage)
                .where(ChannelMessage.business_id == self._pack.id)
                .where(ChannelMessage.conversation_id == key)
                .where(ChannelMessage.direction == OUT)
                .where(ChannelMessage.status == OutStatus.QUEUED)
                .values(status=OutStatus.DROPPED, reason=reason, updated_at=self._now())
            )
        if reason == "window":
            log_alert(logger, "channel.window_closed", channel=self._channel, conversation_id=key)
        else:
            log_event(
                logger, "channel.dropped", channel=self._channel, conversation_id=key, reason=reason
            )

    def _set(self, message_id: uuid.UUID, **values: Any) -> None:
        with self._session() as session:
            session.execute(
                update(ChannelMessage)
                .where(ChannelMessage.id == message_id)
                .values(updated_at=self._now(), **values)
            )
