"""The reception agent leads (Albi, 2026-10-12): a scripted model calls the tools as Gemini would;
code checks the calendar, the hold, the patient's yes and every number of the reply."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.calendar import FixtureCalendar
from frontdesk.booking.scheduler import CalendarScheduler
from frontdesk.conversation.reception import build_reception
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnRuntime, handle_message
from frontdesk.db.engine import business_session
from frontdesk.db.models import BookingHold
from frontdesk.tenancy.pack import BusinessPack, CalendarIntegration, FixedTextKey, Integrations
from tests.conversation.helpers import classifier_turns, document_responder, make_runtime

pytestmark = pytest.mark.db
TIRANE = ZoneInfo("Europe/Tirane")
NOW = datetime(2026, 10, 22, 8, tzinfo=TIRANE)
STEP = "Per prenotare una visita, chiama la reception al +355 00 000 0000."
PRIVACY = "Informazioni e privacy: www.example.com"
Activate = Callable[..., sessionmaker[Session]]
Step = list[tuple[str, dict[str, Any]]] | dict[str, Any]
CALENDAR = """
cal-arben:
  blocks:
    - {start: "2026-10-23 09:00", end: "2026-10-23 10:00"}
    - {start: "2026-10-26 09:00", end: "2026-10-26 10:00"}
  busy: []
"""
# The agent writes the times in its own words; code checks each one against what the tools gave.
LIST = "1) venerdì 23: alle 9 e alle 9:30\n2) lunedì 26: alle 9 e alle 9:30"
FRIDAY = "Venerdì 23 ho posto alle 9 e alle 9:30: quale preferisce?"
SUMMARY = (
    "Riepilogo: venerdì 23 ottobre, 09:30, Visita ortopedica con Dr. Arben Demo.\n"
    "Nome: Mario Rossi\nTelefono: +355691234567\n"
    "Confermo la prenotazione? Risponda sì o no."
)
BOOKED = (
    "Prenotazione confermata: venerdì 23 ottobre, 09:30, Visita ortopedica con Dr. Arben Demo. "
    "Per annullarla o spostarla, chiami la reception al +355 00 000 0000."
)


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


class Script:
    """The model's answers in order: tool calls, or the final reply. Keeps what each call saw."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[list[ModelMessage]] = []

    def model(self) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.calls.append(list(messages))
            step = self.steps.pop(0)
            if isinstance(step, dict):
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, step)])
            return ModelResponse(parts=[ToolCallPart(name, args) for name, args in step])

        return FunctionModel(respond)

    def tool_results(self) -> list[str]:
        return [
            str(part.content)
            for messages in self.calls
            for message in messages[-1:]
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]


@pytest.fixture
def calendar(tmp_path: Path) -> FixtureCalendar:
    path = tmp_path / "calendar.yaml"
    path.write_text(CALENDAR, encoding="utf-8")
    return FixtureCalendar(path)


@pytest.fixture
def base(activate_pack: Activate, clinic_pack: BusinessPack, calendar: FixtureCalendar) -> Any:
    pack = _pack(clinic_pack)
    runtime = make_runtime(
        pack, activate_pack(pack), classifier_turns([], []), document_responder()
    )
    return replace(runtime, scheduler=CalendarScheduler(calendar), now=lambda: NOW)


class Chat:
    def __init__(self, base: TurnRuntime, script: Script) -> None:
        self.runtime = replace(base, reception=build_reception(script.model(), base.pack.profile))
        self.state = ConversationState(conversation_id="c1", turns=1)

    async def say(self, message: str) -> TurnResult:
        result, self.state = await handle_message(message, self.state, self.runtime)
        return result


def _final(text: str, kind: str = "answer") -> dict[str, Any]:
    return {"text": text, "kind": kind}


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=TIRANE)


async def test_a_booking_the_agent_leads_is_made_by_the_tools(
    base: TurnRuntime, calendar: FixtureCalendar
) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"Ecco i primi orari liberi:\n{LIST}\nQuale preferisce?", "answer"),
            [("free_times", {"day": "2026-10-23"})],
            _final(FRIDAY),
            [("hold_time", {"day": "2026-10-23", "time": "09:30"})],
            _final("Le tengo le 09:30. Mi scrive nome, cognome e telefono?", "answer"),
            [("prepare_booking", {"name": "Mario Rossi", "phone": "+355 69 123 4567"})],
            _final(SUMMARY, "answer"),
            [("confirm_booking", {})],
            _final(BOOKED),
        ]
    )
    chat = Chat(base, script)
    assert (await chat.say("Vorrei prenotare")).text.endswith("Quale preferisce?")
    assert (await chat.say("venerdì che orari ci sono?")).text == FRIDAY
    await chat.say("alle 9:30")
    assert (await chat.say("Mario Rossi, +355 69 123 4567")).text == SUMMARY
    booked = await chat.say("sì, confermo")
    assert booked.outcome is TurnOutcome.REPLIED
    assert booked.text == f"{BOOKED}\n\n{PRIVACY}"  # the privacy link ends every booking
    (event,) = calendar.booked["cal-arben"].values()
    assert event.start == at(23, 9, 30)
    with business_session(base.session_factory, base.pack.id) as session:
        (row,) = session.scalars(select(BookingHold))
        assert (row.status, row.name, row.phone) == ("booked", None, None)
    assert chat.state.booking is None
    # The conversation is kept for the agent, its phone numbers masked.
    texts = [entry.text for entry in chat.state.history]
    assert "Mario Rossi, [phone]" in texts
    assert not any("123 4567" in text or "691234567" in text for text in texts)


async def test_without_the_patient_s_yes_nothing_is_booked(
    base: TurnRuntime, calendar: FixtureCalendar
) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?", "answer"),
            [("hold_time", {"day": "2026-10-23", "time": "09:00"})],
            _final("Nome, cognome e telefono?", "answer"),
            [("prepare_booking", {"name": "Mario Rossi", "phone": "+355 69 123 4567"})],
            _final(SUMMARY, "answer"),
            [("confirm_booking", {})],  # the model calls it on a question
            _final("Il prezzo è 40 EUR. Conferma la prenotazione?", "answer"),
        ]
    )
    chat = Chat(base, script)
    for message in ("Vorrei prenotare", "il primo", "Mario Rossi, +355 69 123 4567"):
        await chat.say(message)
    reply = await chat.say("quanto costa?")
    assert "has not said yes" in script.tool_results()[-1]
    assert reply.text == "Il prezzo è 40 EUR. Conferma la prenotazione?"
    assert calendar.booked == {}


async def test_a_number_from_nowhere_is_sent_back_once(base: TurnRuntime) -> None:
    script = Script(
        [_final("La visita ortopedica costa 55 EUR."), _final("La visita ortopedica costa 40 EUR.")]
    )
    reply = await Chat(base, script).say("Quanto costa la visita ortopedica?")
    assert reply.text == "La visita ortopedica costa 40 EUR."
    retry = script.calls[1][-1]
    assert isinstance(retry, ModelRequest)
    assert "55" in str(retry.parts[0])  # the model is told the value
    # Traces and logs get the kind only, never a value of the reply (security review).
    assert reply.trace.check_reasons == ("sent_back:numbers",)


async def test_a_booking_no_tool_made_is_never_announced(base: TurnRuntime) -> None:
    script = Script([_final("Prenotazione confermata!"), _final("Prenotazione confermata!")])
    reply = await Chat(base, script).say("Vorrei prenotare venerdì")
    assert reply.outcome is TurnOutcome.HANDOFF
    assert reply.trace.rejection == "reception_error"
    assert "confermata" not in reply.text


async def test_the_agent_reads_the_conversation_so_far(base: TurnRuntime) -> None:
    script = Script([_final("Certo, mi dica.", "answer"), _final("Il prezzo è 40 EUR.")])
    chat = Chat(base, script)
    await chat.say("Avrei una domanda sulla visita ortopedica")
    await chat.say("quanto costa?")
    earlier = [
        part.content
        for message in script.calls[1]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert earlier == ["Avrei una domanda sulla visita ortopedica", "quanto costa?"]


async def test_an_emergency_the_agent_sees_gets_the_fixed_text(base: TurnRuntime) -> None:
    script = Script([_final("Chiami subito il 127.", "emergency")])
    reply = await Chat(base, script).say("Mio figlio ha ingoiato qualcosa")
    assert len(script.calls) == 1  # the rules let it through; the agent saw it
    emergency = base.pack.fixed_texts["it"][FixedTextKey.EMERGENCY]  # type: ignore[index]
    assert reply.text.startswith(emergency)
    assert reply.trace.category is not None
    assert reply.trace.category.value == "emergency"


async def test_the_rules_still_answer_before_the_agent(base: TurnRuntime) -> None:
    script = Script([])
    reply = await Chat(base, script).say("Ho un forte dolore al petto e non respiro")
    assert reply.trace.rule == "emergency"
    assert script.calls == []


async def test_the_agent_reads_the_clinic_s_facts(base: TurnRuntime) -> None:
    script = Script([_final("Il prezzo è 40 EUR.")])
    await Chat(base, script).say("Quanto costa la visita ortopedica?")
    request = script.calls[0][-1]
    assert isinstance(request, ModelRequest)
    instructions = request.instructions or ""
    assert "Visita ortopedica: 40 EUR" in instructions
    assert "Dr. Arben Demo" in instructions
    assert "Bookable here in the chat" in instructions
    assert "Now: Thursday 2026-10-22 08:00 (Europe/Tirane)" in instructions
    # "Domani" is the next weekday by name: asked on a Friday, it is a Saturday (review, 2026-10-13).
    assert "Tomorrow: Friday 2026-10-23." in instructions


async def test_the_reply_is_in_the_language_the_model_reads_in_the_message(
    base: TurnRuntime,
) -> None:
    # No listed word in "Fate trattamenti osteopatici?": code took it for Albanian, and the agent
    # answered in Albanian (run of 2026-10-13). The model names the language it writes in, and
    # code's own texts follow it.
    reply = {"text": "Sì, ne parli con la reception.", "kind": "handoff", "language": "it"}
    chat = Chat(base, Script([reply]))
    chat.state = ConversationState(conversation_id="c1", language="sq")
    result = await chat.say("Fate trattamenti osteopatici?")
    texts = base.pack.fixed_texts["it"]
    assert result.language == "it"
    assert result.text.startswith(texts[FixedTextKey.DISCLOSURE])  # the first message
    assert result.text.endswith(texts[FixedTextKey.HANDOFF])
    assert chat.state.language == "it"


async def test_a_tool_never_holds_a_time_that_was_not_shown(base: TurnRuntime) -> None:
    script = Script(
        [[("hold_time", {"day": "2026-10-23", "time": "09:00"})], _final("Le mostro gli orari?")]
    )
    await Chat(base, script).say("prendo il primo")
    assert script.tool_results() == ["No times were shown yet: call free_times first."]
    with business_session(base.session_factory, base.pack.id) as session:
        assert list(session.scalars(select(BookingHold))) == []


# --- The security review of the agent (2026-10-12) --------------------------------------------------


async def test_personal_data_outside_a_held_time_is_turned_away_and_not_kept(
    base: TurnRuntime,
) -> None:
    script = Script([])
    chat = Chat(base, script)
    reply = await chat.say("Il mio numero è +355 69 123 4567, mi richiamate?")
    assert reply.text == base.pack.fixed_texts["it"][FixedTextKey.PERSONAL_DATA]  # type: ignore[index]
    assert script.calls == []
    assert chat.state.history == ()


async def test_only_the_name_and_phone_pass_while_a_time_is_held(base: TurnRuntime) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?", "answer"),
            [("hold_time", {"day": "2026-10-23", "time": "09:00"})],
            _final("Nome, cognome e telefono?", "answer"),
        ]
    )
    chat = Chat(base, script)
    await chat.say("Vorrei prenotare")
    await chat.say("il primo")
    reply = await chat.say("Mario Rossi, mario.rossi@example.com")
    assert reply.text == base.pack.fixed_texts["it"][FixedTextKey.PERSONAL_DATA]  # type: ignore[index]
    assert len(script.calls) == 4  # the agent never saw the address
    assert not any("example.com" in entry.text for entry in chat.state.history)


async def test_a_yes_that_does_not_answer_the_summary_books_nothing(
    base: TurnRuntime, calendar: FixtureCalendar
) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?", "answer"),
            [("hold_time", {"day": "2026-10-23", "time": "09:00"})],
            _final("Nome, cognome e telefono?", "answer"),
            [("prepare_booking", {"name": "Mario Rossi", "phone": "+355 69 123 4567"})],
            _final(SUMMARY, "answer"),
            _final("Si paga in clinica. Vuole sapere altro?", "answer"),
            [("confirm_booking", {})],  # a yes to another question
            _final("Mi conferma la prenotazione?", "answer"),
        ]
    )
    chat = Chat(base, script)
    for message in ("Vorrei prenotare", "il primo", "Mario Rossi, +355 69 123 4567", "e pago lì?"):
        await chat.say(message)
    await chat.say("sì")
    assert "The summary was not your last message" in script.tool_results()[-1]
    assert calendar.booked == {}


async def test_the_summary_is_shown_as_code_wrote_it(base: TurnRuntime) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?", "answer"),
            [("hold_time", {"day": "2026-10-23", "time": "09:30"})],
            _final("Nome, cognome e telefono?", "answer"),
            [("prepare_booking", {"name": "Mario Rossi", "phone": "+355 69 123 4567"})],
            _final("Perfetto Mario, confermiamo?", "answer"),
        ]
    )
    chat = Chat(base, script)
    await chat.say("Vorrei prenotare")
    await chat.say("il secondo")
    assert (await chat.say("Mario Rossi, +355 69 123 4567")).text == SUMMARY


# --- Albi, 2026-10-12: one-line introduction, a person's number and WhatsApp ------------------------


async def test_a_first_greeting_gets_the_introduction_alone(base: TurnRuntime) -> None:
    chat = Chat(base, Script([_final("", "greeting")]))
    chat.state = ConversationState(conversation_id="c1")  # the first message
    reply = await chat.say("Buongiorno")
    assert reply.text == base.pack.fixed_texts["it"][FixedTextKey.DISCLOSURE]  # type: ignore[index]


async def test_a_handoff_without_words_is_the_pack_s_own(base: TurnRuntime) -> None:
    reply = await Chat(base, Script([_final("", "handoff")])).say("Vorrei parlare con una persona")
    assert (reply.outcome, reply.text) == (
        TurnOutcome.HANDOFF,
        base.pack.fixed_texts["it"][FixedTextKey.HANDOFF],  # type: ignore[index]
    )


async def test_the_facts_give_reception_s_whatsapp(
    activate_pack: Activate, clinic_pack: BusinessPack, calendar: FixtureCalendar
) -> None:
    location = clinic_pack.location.model_copy(update={"whatsapp": "+355 00 000 0001"})
    pack = _pack(clinic_pack.model_copy(update={"location": location}))
    runtime = make_runtime(
        pack, activate_pack(pack), classifier_turns([], []), document_responder()
    )
    script = Script([_final("Può scriverci su WhatsApp al +355 00 000 0001.", "handoff")])
    chat = Chat(replace(runtime, scheduler=CalendarScheduler(calendar), now=lambda: NOW), script)
    reply = await chat.say("Posso scrivervi su WhatsApp?")
    request = script.calls[0][-1]
    assert isinstance(request, ModelRequest)
    assert "WhatsApp: +355 00 000 0001" in (request.instructions or "")
    assert reply.text.startswith("Può scriverci su WhatsApp al +355 00 000 0001.")


async def test_a_question_back_is_a_clarification(base: TurnRuntime) -> None:
    script = Script(
        [_final("Gli orari di apertura, o altri orari per un appuntamento?", "clarify")]
    )
    reply = await Chat(base, script).say("vorrei sapere gli orari")
    assert reply.outcome is TurnOutcome.ASKED


async def test_a_booking_made_before_the_model_fails_is_still_told(
    base: TurnRuntime, calendar: FixtureCalendar
) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?"),
            [("hold_time", {"day": "2026-10-23", "time": "09:00"})],
            _final("Nome, cognome e telefono?"),
            [("prepare_booking", {"name": "Mario Rossi", "phone": "+355 69 123 4567"})],
            _final(SUMMARY),
            [("confirm_booking", {})],
            # then the model answers no more: two refused replies end the run
            _final("Prenotazione confermata per il 31 ottobre!"),
            _final("Prenotazione confermata per il 31 ottobre!"),
        ]
    )
    chat = Chat(base, script)
    for message in ("Vorrei prenotare", "il primo", "Mario Rossi, +355 69 123 4567"):
        await chat.say(message)
    reply = await chat.say("sì")
    assert reply.text.startswith("Prenotazione confermata: venerdì 23 ottobre, 09:00")
    assert reply.text.endswith(PRIVACY)
    assert len(calendar.booked["cal-arben"]) == 1


async def test_a_confirmation_in_other_words_is_shown_as_code_wrote_it(
    base: TurnRuntime, calendar: FixtureCalendar
) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?"),
            [("hold_time", {"day": "2026-10-23", "time": "09:00"})],
            _final("Nome, cognome e telefono?"),
            [("prepare_booking", {"name": "Mario Rossi", "phone": "+355 69 123 4567"})],
            _final(SUMMARY),
            [("confirm_booking", {})],
            _final("Fatto! Prenotazione confermata per il 30 ottobre."),  # 30 is in the facts
        ]
    )
    chat = Chat(base, script)
    for message in ("Vorrei prenotare", "il primo", "Mario Rossi, +355 69 123 4567"):
        await chat.say(message)
    reply = await chat.say("sì")
    assert reply.text.startswith("Prenotazione confermata: venerdì 23 ottobre, 09:00")
    assert "30 ottobre" not in reply.text
    assert reply.text.count(PRIVACY) == 1


async def test_a_search_reads_the_conversation_s_language_first(base: TurnRuntime) -> None:
    script = Script(
        [[("search_information", {"question": "Come si prepara la visita?"})], _final("Ecco.")]
    )
    await Chat(base, script).say("Come mi preparo alla visita?")
    (found,) = script.tool_results()
    assert found  # the demo clinic's documents
    assert len(found) <= 2500 + 400  # the passages, with their titles


async def test_a_practical_answer_to_a_message_with_symptoms_gets_the_notice(
    base: TurnRuntime,
) -> None:
    script = Script([_final("Apriamo alle 09:00.", "notice")])
    reply = await Chat(base, script).say("Ho la febbre, a che ora aprite domani?")
    texts = base.pack.fixed_texts["it"]
    assert reply.text == (
        f"Apriamo alle 09:00.\n\n{texts[FixedTextKey.CLINICAL]}\n\n{texts[FixedTextKey.HANDOFF]}"  # type: ignore[index]
    )
    assert reply.trace.category is not None
    assert reply.trace.category.value == "clinical"


async def test_every_booking_step_names_the_calendar(base: TurnRuntime) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?"),
            [("hold_time", {"day": "2026-10-23", "time": "09:00"})],
            _final("Nome, cognome e telefono?"),
        ]
    )
    chat = Chat(base, script)
    await chat.say("Vorrei prenotare")
    held = await chat.say("il primo")
    assert held.trace.sources == ("calendar:dr-arben-demo",)


async def test_a_notice_answer_may_not_talk_about_the_problem(base: TurnRuntime) -> None:
    script = Script(
        [
            _final("Con la febbre è meglio riposare; apriamo alle 09:00.", "notice"),
            _final("Apriamo alle 09:00.", "notice"),
        ]
    )
    reply = await Chat(base, script).say("Ho la febbre, a che ora aprite domani?")
    assert reply.text.startswith("Apriamo alle 09:00.\n\n")
    retry = script.calls[1][-1]
    assert isinstance(retry, ModelRequest)
    assert "health problem" in str(retry.parts[0])


# --- Albi's chat of 2026-10-12: "solo questi?", the times in the agent's words -----------------------


async def test_free_times_gives_every_time_of_each_day_and_whether_other_days_have_some(
    base: TurnRuntime,
) -> None:
    script = Script(
        [
            [("free_times", {})],
            _final(f"{LIST}\nQuale preferisce?"),
            [("free_times", {"day": "2026-10-23"})],
            _final(FRIDAY),
        ]
    )
    chat = Chat(base, script)
    await chat.say("Vorrei prenotare")
    await chat.say("venerdì che orari ci sono?")
    first, friday = script.tool_results()
    assert "- 2026-10-23 (venerdì 23 ottobre): 09:00, 09:30" in first
    assert "- 2026-10-26 (lunedì 26 ottobre): 09:00, 09:30" in first
    assert "Every free time of each day below is listed" in first
    assert "Other days with free times: none in the calendar's reach." in first
    # A day asked for: all of it, and the other days are said to exist.
    assert "- 2026-10-23 (venerdì 23 ottobre): 09:00, 09:30" in friday
    assert "2026-10-26" not in friday.splitlines()[1]
    assert "Other days with free times: yes, from 2026-10-26" in friday


async def test_a_time_no_tool_gave_is_sent_back(base: TurnRuntime) -> None:
    script = Script(
        [
            [("free_times", {"day": "2026-10-23"})],
            _final("Venerdì 23 ho posto alle 9, alle 9:30 e alle 11."),  # 11:00 is not free
            _final(FRIDAY),
        ]
    )
    reply = await Chat(base, script).say("venerdì che orari ci sono?")
    assert reply.text == FRIDAY
    assert reply.trace.check_reasons == ("sent_back:times",)


async def test_hold_time_needs_a_time_that_was_shown(base: TurnRuntime) -> None:
    script = Script(
        [
            [("free_times", {"day": "2026-10-23"})],
            _final(FRIDAY),
            [("hold_time", {"day": "2026-10-23", "time": "11:00"})],
            _final("Le mostro gli orari di venerdì?"),
        ]
    )
    chat = Chat(base, script)
    await chat.say("venerdì che orari ci sono?")
    await chat.say("alle 11")
    assert "is not among the times shown" in script.tool_results()[-1]
    with business_session(base.session_factory, base.pack.id) as session:
        assert list(session.scalars(select(BookingHold))) == []
