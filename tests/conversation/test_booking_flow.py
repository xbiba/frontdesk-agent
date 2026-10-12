"""A booking from the chat (M5): slots, a number, the name, the phone, the summary, yes."""

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classifier import build_classifier
from frontdesk.booking.calendar import CalendarUnavailableError, FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.booking.slots import Interval
from frontdesk.conversation.evaluation import TurnCase, run_turn_eval
from frontdesk.conversation.state import BookingStage, ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, handle_message
from frontdesk.db.engine import business_session
from frontdesk.db.models import BookingHold
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, FixedTextKey, Integrations
from tests.conversation.helpers import classifier_turns, document_responder, make_runtime

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 22, 8, tzinfo=TIRANE)
STEP = "Per prenotare una visita, chiama la reception al +355 00 000 0000."
PRIVACY = "Informazioni e privacy: www.example.com"
BOOK = {"category": "administrative", "intent": "booking", "doctor_ids": ["dr-arben-demo"]}
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["orthopedics-visit"]}
CALENDAR = """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 11:00"}
  busy:
    - {start: "2026-10-23 09:30", end: "2026-10-23 10:00"}
"""
OFFER = (
    "Primi orari liberi con Dr. Arben Demo (Visita ortopedica, 30 minuti):\n"
    "1) venerdì 23 ottobre, 09:00\n2) venerdì 23 ottobre, 10:00\n3) venerdì 23 ottobre, 10:30\n\n"
    "Per prenotare, scriva il numero dell'orario che preferisce."
)
TEN = datetime(2026, 10, 23, 10, tzinfo=TIRANE)


def _pack(clinic_pack: BusinessPack) -> BusinessPack:
    texts = {
        language: {**texts, FixedTextKey.NEXT_STEP: STEP, FixedTextKey.BOOKING_PRIVACY: PRIVACY}
        for language, texts in clinic_pack.fixed_texts.items()
    }
    calendar = CalendarIntegration(
        credentials_ref="adc", calendars={"dr-arben-demo": "cal-arben"}, chat_booking=True
    )
    return clinic_pack.model_copy(
        update={"fixed_texts": texts, "integrations": Integrations(calendar=calendar)}
    )


class Chat:
    """One conversation: the classifier answers `outputs` in turn and must not be asked again."""

    def __init__(
        self, runtime: TurnRuntime, outputs: list[dict[str, Any]], conversation: str = "c1"
    ) -> None:
        self.runtime = replace(runtime, classifier=_classifier(runtime, outputs))
        self.state = ConversationState(conversation_id=conversation, turns=1)

    async def say(self, message: str) -> TurnResult:
        result, self.state = await handle_message(message, self.state, self.runtime)
        return result


def _classifier(runtime: TurnRuntime, outputs: list[dict[str, Any]]) -> Any:
    return build_classifier(classifier_turns(list(outputs), []), runtime.pack.profile)


@pytest.fixture
def calendar(tmp_path: Path) -> FixtureCalendar:
    path = tmp_path / "calendar.yaml"
    path.write_text(CALENDAR, encoding="utf-8")
    return FixtureCalendar(path)


@pytest.fixture
def runtime(
    activate_pack: Activate, clinic_pack: BusinessPack, calendar: FixtureCalendar
) -> TurnRuntime:
    pack = _pack(clinic_pack)
    base = make_runtime(pack, activate_pack(pack), classifier_turns([], []), document_responder())
    return replace(base, scheduler=CalendarScheduler(calendar), now=lambda: NOW)


def _holds(runtime: TurnRuntime) -> list[BookingHold]:
    with business_session(runtime.session_factory, runtime.pack.id) as session:
        return list(session.scalars(select(BookingHold)))


async def _to_summary(chat: Chat, choice: str = "2") -> TurnResult:
    assert (await chat.say("Vorrei prenotare")).text == OFFER
    await chat.say(choice)
    await chat.say("Mario Rossi")
    return await chat.say("+355 69 123 4567")


async def test_a_patient_books_from_the_chat(
    runtime: TurnRuntime, calendar: FixtureCalendar, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    chat = Chat(runtime, [BOOK])
    offered = await chat.say("Vorrei prenotare")
    assert (offered.outcome, offered.text) == (TurnOutcome.REPLIED, OFFER)
    assert chat.state.booking is not None
    assert chat.state.booking.stage is BookingStage.CHOOSE
    asked_name = await chat.say("2")
    assert asked_name.text == "Mi scrive il suo nome e cognome?"
    assert (await chat.say("Mario Rossi")).text == "E il suo numero di telefono?"
    summary = await chat.say("+355 69 123 4567")
    assert summary.text == (
        "Riepilogo: venerdì 23 ottobre, 10:00, Visita ortopedica con Dr. Arben Demo.\n"
        "Nome: Mario Rossi\nTelefono: +355691234567\n"
        "Confermo la prenotazione? Risponda sì o no."
    )
    booked = await chat.say("sì")
    assert booked.outcome is TurnOutcome.REPLIED
    assert booked.text == (
        "Prenotazione confermata: venerdì 23 ottobre, 10:00, Visita ortopedica con Dr. Arben Demo. "
        "Per annullarla o spostarla, chiami la reception al +355 00 000 0000.\n\n"
        f"{PRIVACY}"
    )
    assert chat.state.booking is None
    (event,) = calendar.booked["cal-arben"].values()
    assert event == Interval(TEN, TEN + timedelta(minutes=30))
    (row,) = _holds(runtime)
    assert (row.status, row.name, row.phone) == ("booked", None, None)
    # The name and the phone reach no state, trace or log: only the hold, until the booking.
    assert "Rossi" not in repr(chat.state)
    assert all("Rossi" not in repr(result.trace) for result in (summary, booked))
    assert "Rossi" not in caplog.text
    assert "1234567" not in caplog.text


async def test_a_slot_taken_in_the_calendar_meanwhile_gets_fresh_slots(
    runtime: TurnRuntime, calendar: FixtureCalendar
) -> None:
    chat = Chat(runtime, [BOOK])
    await chat.say("Vorrei prenotare")
    calendar.book(
        "cal-arben", TEN, TEN + timedelta(minutes=30), event_id="phone", title="", description=""
    )
    taken = await chat.say("2")
    assert taken.text.startswith("Questo orario non è più libero.\n\nPrimi orari liberi")
    assert "10:00" not in taken.text
    assert chat.state.booking is not None
    assert chat.state.booking.stage is BookingStage.CHOOSE


async def test_a_slot_taken_before_the_yes_is_not_written(
    runtime: TurnRuntime, calendar: FixtureCalendar
) -> None:
    chat = Chat(runtime, [BOOK])
    await _to_summary(chat)
    calendar.book(
        "cal-arben", TEN, TEN + timedelta(minutes=30), event_id="phone", title="", description=""
    )
    taken = await chat.say("sì")
    assert taken.text.startswith("Questo orario non è più libero.")
    assert list(calendar.booked["cal-arben"]) == ["phone"]
    assert _holds(runtime) == []  # the hold went with the name and the phone


async def test_an_expired_hold_books_nothing(
    runtime: TurnRuntime, calendar: FixtureCalendar
) -> None:
    chat = Chat(runtime, [BOOK])
    await _to_summary(chat)
    later = NOW + timedelta(minutes=16)
    chat.runtime = replace(chat.runtime, now=lambda: later)
    expired = await chat.say("sì")
    assert expired.text.startswith("Il tempo per prenotare è scaduto.\n\nPrimi orari liberi")
    assert calendar.booked == {}


async def test_two_patients_on_one_slot(runtime: TurnRuntime) -> None:
    first, second = Chat(runtime, [BOOK], "c1"), Chat(runtime, [BOOK], "c2")
    await first.say("Vorrei prenotare")
    await second.say("Vorrei prenotare")
    assert (await first.say("2")).text == "Mi scrive il suo nome e cognome?"
    assert (await second.say("2")).text.startswith("Questo orario non è più libero.")


async def test_an_emergency_in_the_middle_of_a_booking_gets_its_text(
    runtime: TurnRuntime,
) -> None:
    chat = Chat(runtime, [BOOK])
    await chat.say("Vorrei prenotare")
    await chat.say("2")
    emergency = await chat.say("Ho un forte dolore al petto e non respiro")
    assert emergency.text.startswith(runtime.pack.fixed_texts["it"][FixedTextKey.EMERGENCY])  # type: ignore[index]
    assert chat.state.booking is None


async def test_a_question_at_the_choice_goes_through_the_pipeline(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK, PRICE])
    await chat.say("Vorrei prenotare")
    price = await chat.say("Quanto costa?")
    assert price.text.startswith("Visita ortopedica: 40 EUR")
    assert chat.state.booking is None


async def test_a_wrong_name_twice_ends_with_how_to_book(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK])
    await chat.say("Vorrei prenotare")
    await chat.say("2")
    assert (await chat.say("12345")).text == "Mi scriva solo nome e cognome, per favore."
    ended = await chat.say("4 5 6")
    assert (ended.outcome, ended.text) == (TurnOutcome.HANDOFF, STEP)
    assert _holds(runtime) == []


async def test_no_at_the_summary_books_nothing(
    runtime: TurnRuntime, calendar: FixtureCalendar
) -> None:
    chat = Chat(runtime, [BOOK])
    await _to_summary(chat)
    assert (await chat.say("no")).text == "Va bene, non ho prenotato nulla."
    assert calendar.booked == {}
    assert _holds(runtime) == []


async def test_a_calendar_that_refuses_the_write_says_how_to_book(
    runtime: TurnRuntime, calendar: FixtureCalendar, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise CalendarUnavailableError("status 403")

    chat = Chat(runtime, [BOOK])
    await _to_summary(chat)
    monkeypatch.setattr(calendar, "book", refuse)
    failed = await chat.say("sì")
    assert (failed.outcome, failed.text) == (
        TurnOutcome.HANDOFF,
        f"Non sono riuscito a completare la prenotazione.\n\n{STEP}",
    )
    assert _holds(runtime) == []


async def test_booking_cases_can_be_run_again_and_again(runtime: TurnRuntime) -> None:
    # Each case gets a fresh fixture calendar and leaves no hold: the same booking twice books twice.
    case = TurnCase.model_validate(
        {
            "id": "book",
            "messages": ["Vorrei prenotare", "2", "Mario Rossi", "+355 69 123 4567", "sì"],
            "expect": {"outcome": "replied", "contains": ["Prenotazione confermata"]},
        }
    )
    twice = replace(runtime, classifier=_classifier(runtime, [BOOK, BOOK]))
    report = await run_turn_eval(
        twice, [case, case.model_copy(update={"id": "again"})], route="test"
    )
    assert [result.failures for result in report.results] == [(), ()]
    assert _holds(runtime) == []
