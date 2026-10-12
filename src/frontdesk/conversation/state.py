"""Conversation state between turns: metadata only, never message text."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.tenancy.pack import Language

MAX_REMEMBERED_INTENTS: Final = 5
TOPIC_TURNS: Final = 3  # turns a topic survives without being mentioned again
# The key of the order lookup attempts is a String(64) column: a longer id would fail every order
# turn at the database.
MAX_CONVERSATION_ID_CHARS: Final = 64


class BookingStage(StrEnum):
    CHOOSE = "choose"  # the free slots were offered
    NAME = "name"
    PHONE = "phone"
    CONFIRM = "confirm"


class BookingStep(BaseModel):
    """Where a booking from the chat stands (M5). Metadata only: the name and the phone live on the
    hold in Postgres, never here."""

    model_config = ConfigDict(frozen=True)

    stage: BookingStage
    doctor_id: str
    service_id: str
    offered: tuple[datetime, ...] = ()
    hold_id: uuid.UUID | None = None
    retries: int = 0


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
    ) -> ConversationState:
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
            }
        )
