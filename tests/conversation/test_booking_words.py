"""A booking in the patient's own words (Albi's live chat, 2026-10-12): "Friday is fine", "what
times are there?", "in the afternoon", "on Monday". The classifier reads them against the times just
offered and the previous reply; code checks every time against the calendar."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classifier import build_classifier
from frontdesk.booking.calendar import FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.conversation.state import BookingStage
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.tenancy.pack import BusinessPack
from tests.conversation.helpers import classifier_turns, document_responder, make_runtime
from tests.conversation.test_booking_flow import BOOK, NOW, TIRANE, Chat, _holds, _pack

pytestmark = pytest.mark.db
Activate = Callable[..., sessionmaker[Session]]
CALENDAR = """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 10:00"}
    - {start: "2026-10-26 09:00", end: "2026-10-26 10:00"}
    - {start: "2026-10-26 15:00", end: "2026-10-26 16:00"}
    - {start: "2026-10-27 14:00", end: "2026-10-27 15:00"}
  busy: []
"""
ABOUT = "(Visita ortopedica, 30 minuti):"
CHOOSE = "Quale preferisce? Può scrivere il numero o il giorno."
FIRST = (
    f"Primi orari liberi con Dr. Arben Demo {ABOUT}\n"
    "1) venerdì 23 ottobre, 09:00\n2) lunedì 26 ottobre, 09:00\n3) martedì 27 ottobre, 14:00\n\n"
    f"{CHOOSE}"
)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TIRANE)


@pytest.fixture
def runtime(activate_pack: Activate, clinic_pack: BusinessPack, tmp_path: Path) -> TurnRuntime:
    path = tmp_path / "calendar.yaml"
    path.write_text(CALENDAR, encoding="utf-8")
    pack = _pack(clinic_pack)
    base = make_runtime(pack, activate_pack(pack), classifier_turns([], []), document_responder())
    return replace(base, scheduler=CalendarScheduler(FixtureCalendar(path)), now=lambda: NOW)


def _seeing(runtime: TurnRuntime, outputs: list[dict[str, Any]], seen: list[str]) -> Chat:
    chat = Chat(runtime, [])
    classifier = build_classifier(classifier_turns(outputs, seen), runtime.pack.profile)
    chat.runtime = replace(chat.runtime, classifier=classifier)
    return chat


async def test_friday_is_fine_takes_the_friday_time(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK, {**BOOK, "slot": 1}])
    assert (await chat.say("Vorrei prenotare")).text == FIRST
    assert (await chat.say("venerdì va bene")).text == "Mi scrive il suo nome e cognome?"
    (row,) = _holds(runtime)
    assert row.during.lower == at(23, 9)


async def test_other_times_are_the_ones_not_offered_yet(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK, BOOK])
    await chat.say("Vorrei prenotare")
    more = await chat.say("che orari ci sono?")
    assert more.text == (
        f"Altri orari liberi con Dr. Arben Demo {ABOUT}\n1) venerdì 23 ottobre, 09:30\n"
        f"2) lunedì 26 ottobre, 09:30\n3) martedì 27 ottobre, 14:30\n\n{CHOOSE}"
    )
    assert chat.state.booking is not None
    assert chat.state.booking.offered == (at(23, 9, 30), at(26, 9, 30), at(27, 14, 30))


async def test_the_afternoon(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK, {**BOOK, "slot_part": "afternoon"}])
    await chat.say("Vorrei prenotare")
    afternoon = await chat.say("avete qualcosa di pomeriggio?")
    assert afternoon.text == (
        f"Orari liberi con Dr. Arben Demo di pomeriggio {ABOUT}\n1) lunedì 26 ottobre, 15:00\n"
        f"2) lunedì 26 ottobre, 15:30\n3) martedì 27 ottobre, 14:00\n\n{CHOOSE}"
    )


async def test_a_day(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK, {**BOOK, "slot_day": "2026-10-26"}])
    await chat.say("Vorrei prenotare")
    monday = await chat.say("lunedì?")
    assert monday.text == (
        f"Orari liberi con Dr. Arben Demo lunedì 26 ottobre {ABOUT}\n1) lunedì 26 ottobre, 09:00\n"
        f"2) lunedì 26 ottobre, 09:30\n3) lunedì 26 ottobre, 15:00\n\n{CHOOSE}"
    )
    assert (await chat.say("2")).text == "Mi scrive il suo nome e cognome?"
    (row,) = _holds(runtime)
    assert row.during.lower == at(26, 9, 30)


async def test_a_day_without_free_times_says_so_then_offers_the_first(
    runtime: TurnRuntime,
) -> None:
    chat = Chat(runtime, [BOOK, {**BOOK, "slot_day": "2026-10-24"}])
    await chat.say("Vorrei prenotare")
    saturday = await chat.say("sabato?")
    assert saturday.text == f"Non ci sono orari liberi sabato 24 ottobre.\n\n{FIRST}"
    assert chat.state.booking is not None
    assert chat.state.booking.stage is BookingStage.CHOOSE


async def test_a_first_request_can_name_the_day_and_the_part(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [{**BOOK, "slot_day": "2026-10-26", "slot_part": "afternoon"}])
    offered = await chat.say("Vorrei prenotare lunedì pomeriggio")
    assert offered.text == (
        f"Orari liberi con Dr. Arben Demo lunedì 26 ottobre, di pomeriggio {ABOUT}\n"
        f"1) lunedì 26 ottobre, 15:00\n2) lunedì 26 ottobre, 15:30\n\n{CHOOSE}"
    )


async def test_when_every_time_was_offered_the_first_come_again(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK] * 4)
    await chat.say("Vorrei prenotare")
    await chat.say("altri orari?")
    await chat.say("altri ancora?")  # Monday 15:00 and 15:30 are the last ones
    last = await chat.say("e poi?")
    assert last.text.startswith("Non ci sono altri orari liberi nei prossimi 14 giorni.\n\n")
    assert last.text.endswith(FIRST)


async def test_the_classifier_reads_the_times_offered_and_the_previous_reply(
    runtime: TurnRuntime,
) -> None:
    seen: list[str] = []
    chat = _seeing(runtime, [BOOK, {**BOOK, "slot": 2}], seen)
    await chat.say("Vorrei prenotare")
    assert (await chat.say("quello di lunedì")).text == "Mi scrive il suo nome e cognome?"
    assert "Today: Thursday 2026-10-22." in seen[0]
    assert "The assistant's previous message:\n<<<" not in seen[0]
    assert (
        "Times the assistant has just offered: 1) Friday 2026-10-23 09:00; "
        "2) Monday 2026-10-26 09:00; 3) Tuesday 2026-10-27 14:00." in seen[1]
    )
    assert f"The assistant's previous message:\n<<<\n{FIRST}\n>>>" in seen[1]


async def test_the_state_keeps_the_last_reply_but_never_the_summary(runtime: TurnRuntime) -> None:
    chat = Chat(runtime, [BOOK])
    offered = await chat.say("Vorrei prenotare")
    assert chat.state.last_reply == offered.text
    await chat.say("1")
    await chat.say("Mario Rossi")
    summary = await chat.say("+355 69 123 4567")
    assert "Mario Rossi" in summary.text
    assert chat.state.last_reply is None
