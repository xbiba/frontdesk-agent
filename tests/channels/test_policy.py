"""A channel's policy narrows the business pack (ADR-0010, M4 design D5): it switches features
off and never on, and a booking switched off by the channel cannot be reached."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.channels.policy import narrow
from frontdesk.conversation.reception import build_reception
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import handle_message
from frontdesk.db.engine import business_session
from frontdesk.db.models import BookingHold
from frontdesk.tenancy.pack import (
    BusinessPack,
    CalendarIntegration,
    ChannelPolicy,
    Channels,
    FixedTextKey,
    Integrations,
)
from tests.conversation.helpers import classifier_turns, document_responder, make_runtime

TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 22, 8, tzinfo=TIRANE)
CALENDAR = """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 10:00"}
  busy: []
"""
REFUSED = "Bookings are made by reception"


def _bookable(clinic_pack: BusinessPack) -> BusinessPack:
    texts = {
        language: {**texts, FixedTextKey.BOOKING_PRIVACY: "Informazioni e privacy: www.example.com"}
        for language, texts in clinic_pack.fixed_texts.items()
    }
    calendar = CalendarIntegration(
        credentials_ref="adc", calendars={"dr-arben-demo": "cal-arben"}, chat_booking=True
    )
    return clinic_pack.model_copy(
        update={"fixed_texts": texts, "integrations": Integrations(calendar=calendar)}
    )


def test_a_policy_switches_features_off(clinic_pack: BusinessPack) -> None:
    pack = _bookable(clinic_pack)
    narrowed = narrow(pack, ChannelPolicy(chat_booking=False, orientation=False))
    assert narrowed.integrations.calendar is not None
    assert narrowed.integrations.calendar.chat_booking is False
    assert narrowed.features.orientation is False
    assert pack.integrations.calendar is not None
    assert pack.integrations.calendar.chat_booking is True  # the pack it came from is untouched
    assert pack.features.orientation is True


def test_a_policy_cannot_switch_a_feature_on(clinic_pack: BusinessPack) -> None:
    # The demo clinic has no order status and no calendar: an open policy leaves it as it is.
    assert narrow(clinic_pack, ChannelPolicy()) == clinic_pack


def test_the_pack_holds_the_instagram_policy(clinic_pack: BusinessPack) -> None:
    raw = clinic_pack.model_dump(mode="json")
    raw["channels"] = {"instagram": {"chat_booking": False}}
    pack = BusinessPack.model_validate(raw)
    assert pack.channels == Channels(instagram=ChannelPolicy(chat_booking=False))
    assert clinic_pack.channels.instagram is None


def _model(seen: list[str]) -> FunctionModel:
    """The reception agent asks for free times, then answers; it keeps what the tool said."""
    steps: list[Any] = [
        [("free_times", {})],
        {"text": "La reception la aiuta a prenotare.", "kind": "answer"},
    ]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        if isinstance(last, ModelRequest):
            seen.extend(str(p.content) for p in last.parts if isinstance(p, ToolReturnPart))
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, dict):
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, step)])
        return ModelResponse(parts=[ToolCallPart(name, args) for name, args in step])

    return FunctionModel(respond)


@pytest.mark.db
@pytest.mark.parametrize("booking", [True, False])
async def test_bookings_switched_off_by_the_channel_cannot_be_reached(
    activate_pack: Callable[..., sessionmaker[Session]],
    clinic_pack: BusinessPack,
    tmp_path: Path,
    booking: bool,
) -> None:
    pack = narrow(_bookable(clinic_pack), ChannelPolicy(chat_booking=booking))
    factory = activate_pack(pack)
    calendar = tmp_path / "calendar.yaml"
    calendar.write_text(CALENDAR, encoding="utf-8")
    seen: list[str] = []
    runtime = replace(
        make_runtime(pack, factory, classifier_turns([], []), document_responder()),
        scheduler=CalendarScheduler(FixtureCalendar(calendar)),
        now=lambda: NOW,
        reception=build_reception(_model(seen), pack.profile),
    )
    state = ConversationState(conversation_id="c1", turns=1)
    await handle_message("Vorrei prenotare una visita ortopedica", state, runtime)
    assert seen, "the model called free_times"
    assert (REFUSED in seen[0]) is (not booking)
    with business_session(factory, pack.id) as session:
        assert session.scalar(select(func.count()).select_from(BookingHold)) == 0
