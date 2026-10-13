"""Conversation state between turns: metadata, and the last reply as context. Only a conversation
the reception agent leads keeps its last messages (Albi, 2026-10-12: the model must read the
conversation), with every phone-like number masked."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.observability.tracing import redact
from frontdesk.tenancy.pack import Language

MAX_REMEMBERED_INTENTS: Final = 5
# The assistant's last reply, which the models read the next message with (Albi's live chat,
# 2026-10-12: "Friday is fine" answers the times just offered). Never the person's message.
MAX_REPLY_CHARS: Final = 600
# The messages the reception agent reads back, each cut to a length.
MAX_HISTORY: Final = 12
MAX_HISTORY_CHARS: Final = 800
_PHONE_LIKE: Final = re.compile(r"\+?\d[\d\s\-./()]{5,}\d")
TOPIC_TURNS: Final = 3  # turns a topic survives without being mentioned again
# The key of the order lookup attempts is a String(64) column: a longer id would fail every order
# turn at the database.
MAX_CONVERSATION_ID_CHARS: Final = 64


class BookingStage(StrEnum):
    CHOOSE = "choose"  # the free slots were offered
    NAME = "name"
    CONFIRM = "confirm"


class BookingStep(BaseModel):
    """Where a booking from the chat stands (M5). Metadata only: the name and the phone live on the
    hold in Postgres, never here."""

    model_config = ConfigDict(frozen=True)

    stage: BookingStage
    doctor_id: str
    service_id: str
    offered: tuple[datetime, ...] = ()
    shown: tuple[datetime, ...] = ()  # every time offered so far: "other times" are the rest
    hold_id: uuid.UUID | None = None
    # The reception agent: the turn whose reply showed the summary; only the next message may say yes.
    summary_turn: int | None = None


class HistoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["patient", "assistant"]
    text: str


def remembered(role: Literal["patient", "assistant"], text: str) -> HistoryEntry:
    """A message as the conversation keeps it: email addresses and phone-like numbers masked, cut
    to a length."""
    masked = _PHONE_LIKE.sub("[…]", redact(text))  # addresses, then anything phone-like
    return HistoryEntry(role=role, text=masked[:MAX_HISTORY_CHARS])


class ConversationState(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=MAX_CONVERSATION_ID_CHARS
    )
    language: Language | None = None
    turns: int = 0
    previous_intents: tuple[Intent, ...] = ()
    topic_service_ids: tuple[str, ...] = ()
    topic_doctor_ids: tuple[str, ...] = ()
    topic_age: int = 0
    pending_clarification: ClarifyField | None = None
    next_step_shown: bool = False  # the pack's next step ends one answer per conversation
    booking: BookingStep | None = None  # a booking under way; any other turn ends it
    last_reply: str | None = None  # never one that holds a name or a phone
    history: tuple[HistoryEntry, ...] = ()  # the reception agent's conversation only

    def advanced(
        self,
        *,
        language: Language,
        intent: Intent | None,
        category: Category | None,
        service_ids: Sequence[str],
        doctor_ids: Sequence[str],
        clarify: ClarifyField | None,
        next_step_shown: bool = False,
        booking: BookingStep | None = None,
        reply: str | None = None,
        exchange: tuple[str, str] | None = None,
    ) -> ConversationState:
        """`exchange`: the patient's message and the reply, kept when the reception agent leads."""
        history = self.history
        if exchange is not None:
            message, answer = exchange
            kept = (remembered("patient", message), remembered("assistant", answer))
            history = (*history, *kept)[-MAX_HISTORY:]
        intents = self.previous_intents
        if intent is not None:
            intents = (*intents, intent)[-MAX_REMEMBERED_INTENTS:]
        services, doctors, age = self.topic_service_ids, self.topic_doctor_ids, self.topic_age + 1
        if category is Category.EMERGENCY or age >= TOPIC_TURNS:
            services, doctors, age = (), (), 0
        if service_ids or doctor_ids:
            services, doctors, age = tuple(service_ids), tuple(doctor_ids), 0
        return self.model_copy(
            update={
                "language": language,
                "turns": self.turns + 1,
                "previous_intents": intents,
                "topic_service_ids": services,
                "topic_doctor_ids": doctors,
                "topic_age": age,
                "pending_clarification": clarify,
                "next_step_shown": self.next_step_shown or next_step_shown,
                "booking": booking,
                "last_reply": reply[:MAX_REPLY_CHARS] if reply else None,
                "history": history,
            }
        )
