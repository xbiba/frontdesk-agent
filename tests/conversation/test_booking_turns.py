"""A booking question gets the first free slots of the doctor's calendar, written by code (M4)."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.vocab import ClarifyField
from frontdesk.booking.calendar import FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, handle_message
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, FixedTextKey, Integrations
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 22, 8, tzinfo=TIRANE)
STEP = "Per prenotare una visita, chiama la reception al +355 00 000 0000."
NO_SLOTS = "Al momento non ci sono orari liberi da proporre."
BOOK = {"category": "administrative", "intent": "booking", "doctor_ids": ["dr-arben-demo"]}
CALENDAR = """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 11:00"}
  busy:
    - {start: "2026-10-23 09:30", end: "2026-10-23 10:00"}
cal-elira: {blocks: [], busy: []}
cal-empty: {blocks: [], busy: []}
"""
SLOTS = (
    "Primi orari liberi con Dr. Arben Demo (Visita ortopedica, 30 minuti):\n"
    "- venerdì 23 ottobre, 09:00\n- venerdì 23 ottobre, 10:00\n- venerdì 23 ottobre, 10:30"
)


def _pack(clinic_pack: BusinessPack, **calendars: str) -> BusinessPack:
    texts = {
        language: {**texts, FixedTextKey.NEXT_STEP: STEP, FixedTextKey.NO_SLOTS: NO_SLOTS}
        for language, texts in clinic_pack.fixed_texts.items()
    }
    calendar = CalendarIntegration(
        credentials_ref="adc", calendars=calendars or {"dr-arben-demo": "cal-arben"}
    )
    return clinic_pack.model_copy(
        update={"fixed_texts": texts, "integrations": Integrations(calendar=calendar)}
    )


def _runtime(
    pack: BusinessPack,
    activate: Activate,
    args: dict[str, Any],
    tmp_path: Path,
    *,
    calendar: bool = True,
) -> TurnRuntime:
    path = tmp_path / "calendar.yaml"
    path.write_text(CALENDAR, encoding="utf-8")
    runtime = make_runtime(pack, activate(pack), classifier_model(args), document_responder())
    scheduler = CalendarScheduler(FixtureCalendar(path)) if calendar else None
    return replace(runtime, scheduler=scheduler, now=lambda: NOW)


async def _ask(
    runtime: TurnRuntime, message: str = "Vorrei prenotare"
) -> tuple[TurnResult, ConversationState]:
    return await handle_message(message, ConversationState(turns=1), runtime)


async def test_a_booking_question_gets_the_first_free_slots_and_how_to_book(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    runtime = _runtime(_pack(clinic_pack), activate_pack, BOOK, tmp_path)
    result, state = await _ask(runtime)
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text == f"{SLOTS}\n\n{STEP}"
    assert result.trace.sources == ("calendar:dr-arben-demo",)
    assert state.next_step_shown


async def test_asking_when_the_doctor_is_in_gets_the_free_slots(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    args = {**BOOK, "intent": "doctor_info"}
    runtime = _runtime(_pack(clinic_pack), activate_pack, args, tmp_path)
    result, _ = await _ask(runtime, "Quando c'è il dottor Arben?")
    assert result.text.startswith(SLOTS)


async def test_the_only_bookable_doctor_needs_no_name(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    args = {"category": "administrative", "intent": "booking"}
    runtime = _runtime(_pack(clinic_pack), activate_pack, args, tmp_path)
    result, _ = await _ask(runtime)
    assert result.text.startswith(SLOTS)


async def test_no_free_slot_says_so_and_how_to_book(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    pack = _pack(clinic_pack, **{"dr-arben-demo": "cal-empty"})
    result, _ = await _ask(_runtime(pack, activate_pack, BOOK, tmp_path))
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text == f"{NO_SLOTS}\n\n{STEP}"


async def test_a_calendar_that_cannot_be_read_says_how_to_book(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    pack = _pack(clinic_pack, **{"dr-arben-demo": "cal-unknown"})
    result, _ = await _ask(_runtime(pack, activate_pack, BOOK, tmp_path))
    assert (result.outcome, result.text) == (TurnOutcome.HANDOFF, STEP)
    assert result.trace.rejection == "calendar_unavailable"


async def test_two_bookable_doctors_and_none_named_asks_which(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    pack = _pack(clinic_pack, **{"dr-arben-demo": "cal-arben", "ft-elira-shembull": "cal-elira"})
    args = {"category": "administrative", "intent": "booking"}
    result, _ = await _ask(_runtime(pack, activate_pack, args, tmp_path))
    assert result.outcome is TurnOutcome.ASKED
    assert result.trace.clarify is ClarifyField.DOCTOR


async def test_a_booking_without_a_calendar_says_how_to_book(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    for pack, reply in (
        (_pack(clinic_pack), STEP),
        # A pack without a next step keeps the handoff of M3.
        (clinic_pack, clinic_pack.fixed_texts["it"][FixedTextKey.HANDOFF]),  # type: ignore[index]
    ):
        runtime = make_runtime(pack, factory, classifier_model(BOOK), document_responder())
        result, _ = await _ask(replace(runtime, now=lambda: NOW))
        assert (result.outcome, result.text) == (TurnOutcome.HANDOFF, reply)


async def test_a_booking_with_symptoms_gets_the_clinical_notice(
    activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path
) -> None:
    args = {**BOOK, "category": "clinical"}
    result, _ = await _ask(_runtime(_pack(clinic_pack), activate_pack, args, tmp_path))
    clinical = clinic_pack.fixed_texts["it"][FixedTextKey.CLINICAL]  # type: ignore[index]
    assert result.outcome is TurnOutcome.HANDOFF
    assert result.text.startswith(clinical)
    assert "Primi orari" not in result.text
