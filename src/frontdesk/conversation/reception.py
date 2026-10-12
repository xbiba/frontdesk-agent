"""The reception agent (Albi, 2026-10-12: "troppo stupido"): a model leads the conversation with the
clinic's facts and tools; code keeps the facts and the bookings.

The rules still run first (frontdesk.conversation.turn): emergency, clinical advice, personal data
outside a booking. The agent reads the conversation so far, the clinic's approved facts and the
date. It may search the approved documents, and book through tools that check each request against
the calendar, the hold and the patient's own words: no time it was not shown, no booking without a
summary and a yes. Its reply is checked (frontdesk.agent.reply_check) and sent back once when it
fails. When it judges a message an emergency or a request for medical advice, the pack's fixed text
answers instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classification import check_free_text
from frontdesk.agent.prompts import instructions_for
from frontdesk.agent.reply_check import reply_problems
from frontdesk.agent.vocab import ClarifyField, SlotPart
from frontdesk.booking.flow import affirms, phrase, valid_name, valid_phone
from frontdesk.booking.holds import attach, held
from frontdesk.booking.reply import SlotWish, booking_target, slot_time
from frontdesk.booking.scheduler import Scheduler
from frontdesk.conversation.booking_turn import (
    BookingContext,
    StepReply,
    cancel,
    confirm,
    country_code,
    take,
)
from frontdesk.conversation.booking_turn import offer as booking_offer
from frontdesk.conversation.state import BookingStage, BookingStep, HistoryEntry
from frontdesk.db.engine import business_session
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.evidence import neutralized
from frontdesk.knowledge.facts import Catalog
from frontdesk.knowledge.retrieval import RetrievedChunk, chunk_evidence, hybrid_search
from frontdesk.safety.rules import contains_clinical_terms
from frontdesk.tenancy.pack import BusinessPack, Language, Profile

RECEPTION_LIMITS: Final = UsageLimits(request_limit=6)
SEARCH_CANDIDATES: Final = 12  # passages the search ranks
SEARCH_PASSAGES: Final = 4  # passages the agent reads, in its language when there are some
# What one search returns at most: each later request of the turn carries it again, and a request
# must stay within the reserve's bytes (one passed 32,000 in the run of 2026-10-12).
SEARCH_CHARS: Final = 2500


class ReplyKind(StrEnum):
    GREETING = "greeting"  # only a greeting or small talk
    ANSWER = "answer"
    CLARIFY = "clarify"  # it cannot answer before the patient says what they mean
    HANDOFF = "handoff"
    NOTICE = "notice"  # an answer to the practical part of a message that describes symptoms
    EMERGENCY = "emergency"
    CLINICAL = "clinical"


class ReceptionReply(BaseModel):
    text: str = Field(description="The message to send the patient, in the conversation's language")
    kind: ReplyKind = Field(
        description="greeting (only a greeting or small talk), answer (you answered, showed "
        "times or asked for the next booking step), clarify (you cannot answer before they say what "
        "they mean), handoff (you send them to reception), notice (you answered the practical part of "
        "a message that describes symptoms), clinical (they ask for medical advice) or emergency"
    )


@dataclass
class Desk:
    """What the agent works with during one turn. The tools update `booking`, `seen`, `sources`
    and `booked`; the turn reads them afterwards."""

    pack: BusinessPack
    catalog: Catalog
    session_factory: sessionmaker[Session]
    scheduler: Scheduler | None
    embedder: Embedder
    min_similarity: float
    conversation_id: str
    language: Language
    now: datetime
    message: str
    first_turn: bool
    turn: int  # the turns before this one: a summary shown now may be confirmed at turn + 1
    facts: str
    conversation: str  # the conversation so far, as text: a number in it may be repeated
    booking: BookingStep | None
    seen: list[str] = field(default_factory=list)  # what the tools returned this turn
    sources: list[str] = field(default_factory=list)
    booked: bool = False
    summary: str | None = None  # the summary prepare_booking wrote this turn, shown as written
    booked_text: str | None = None  # what confirm_booking said when it booked

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.pack.timezone)

    @property
    def today(self) -> date:
        return self.now.astimezone(self.zone).date()

    def context(self) -> BookingContext:
        if self.scheduler is None:
            raise ValueError("a booking needs the runtime's calendar")
        return BookingContext(
            pack=self.pack,
            catalog=self.catalog,
            scheduler=self.scheduler,
            session_factory=self.session_factory,
            conversation_id=self.conversation_id,
            language=self.language,
            now=self.now,
        )

    def said(self, text: str) -> str:
        self.seen.append(text)
        return text


def history_messages(history: tuple[HistoryEntry, ...]) -> list[ModelMessage]:
    """The conversation so far as the model's own messages: what the patient wrote, what the
    assistant answered (the tools of earlier turns are not kept)."""
    return [
        ModelRequest(parts=[UserPromptPart(entry.text)])
        if entry.role == "patient"
        else ModelResponse(parts=[TextPart(entry.text)])
        for entry in history
    ]


def render_situation(desk: Desk) -> str:
    local = desk.now.astimezone(desk.zone)
    lines = [
        f"Now: {local:%A} {local:%Y-%m-%d %H:%M} ({desk.pack.timezone}). "
        f"Conversation language: {desk.language}.",
    ]
    if desk.first_turn:
        lines.append(
            "This is the patient's first message: the system sends its own introduction first, "
            "which greets and asks how it can help. Do not greet, introduce yourself or ask that "
            "again; if the patient only greeted, kind greeting with no text."
        )
    if desk.booking is not None:
        lines.append(f"Booking under way: {_booking_state(desk, desk.booking)}")
    lines.append(f"Clinic facts:\n{desk.facts}")
    return "\n\n".join(lines)


def _booking_state(desk: Desk, step: BookingStep) -> str:
    if step.stage is BookingStage.CHOOSE:
        times = "; ".join(
            f"{number}) {slot_time(start.astimezone(desk.zone), desk.language)}"
            for number, start in enumerate(step.offered, start=1)
        )
        return f"times shown to the patient: {times}."
    if step.stage is BookingStage.CONFIRM:
        return "the summary was shown; a yes from the patient lets you call confirm_booking."
    return "a time is held; ask for the full name and phone number, then call prepare_booking."


def build_reception(
    model: Model,
    profile: Profile,
    *,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
) -> Agent[Desk, ReceptionReply]:
    """`instructions` is the prompt text a caller has already read (`read_prompt`)."""
    if not isinstance(model, Model):
        raise TypeError(
            f"the reception agent needs a Model object, not {type(model).__name__}: build live "
            "models in the model gateway, because a model-id string would skip its EU pin"
        )
    agent = Agent(
        model,
        name=f"{profile.value}-reception",
        deps_type=Desk,
        output_type=ReceptionReply,
        instructions=instructions_for(profile, "reception", instructions),
        retries=1,
        model_settings=model_settings,
    )

    @agent.instructions
    def situation(ctx: RunContext[Desk]) -> str:
        return render_situation(ctx.deps)

    @agent.tool
    def search_information(ctx: RunContext[Desk], question: str) -> str:
        """Search the clinic's approved documents: which days a doctor is in, how a visit works,
        what to bring, payment and other practical questions. Use only what the passages say.

        Args:
            question: A short practical question, without names, numbers, contact or health details.
        """
        return _search(ctx.deps, question)

    @agent.tool
    def free_times(
        ctx: RunContext[Desk],
        day: str | None = None,
        from_day: str | None = None,
        part: SlotPart | None = None,
        more: bool = False,
        doctor_id: str | None = None,
        service_id: str | None = None,
    ) -> str:
        """The free times to book, read from the calendar. Show them to the patient as the numbered
        list written here. Without arguments: the first free times.

        Args:
            day: One day the patient asks about, as YYYY-MM-DD.
            from_day: The first day the patient can come, as YYYY-MM-DD ("next week": its Monday).
            part: morning or afternoon, when the patient says so.
            more: True when the patient asks for times other than those already shown.
            doctor_id: The doctor's id from the facts, when the patient names one.
            service_id: The service's id from the facts, when the patient names one.
        """
        return _free_times(ctx.deps, day, from_day, part, more, doctor_id, service_id)

    @agent.tool
    def hold_time(ctx: RunContext[Desk], number: int) -> str:
        """Hold the time the patient picked, by its number in the last list shown.

        Args:
            number: The time's number in that list.
        """
        return _hold_time(ctx.deps, number)

    @agent.tool
    def prepare_booking(ctx: RunContext[Desk], name: str, phone: str) -> str:
        """Put the patient's full name and phone number on the held time. Returns the summary to
        show them, word for word, before they confirm.

        Args:
            name: First and last name, as the patient wrote them.
            phone: The phone number, as the patient wrote it.
        """
        return _prepare_booking(ctx.deps, name, phone)

    @agent.tool
    def confirm_booking(ctx: RunContext[Desk]) -> str:
        """Book the held time. Only after the patient said yes to the summary, in their last
        message."""
        return _confirm_booking(ctx.deps)

    @agent.tool
    def cancel_booking(ctx: RunContext[Desk]) -> str:
        """Let go of the held time, when the patient changes their mind. Nothing is booked."""
        return _cancel_booking(ctx.deps)

    @agent.output_validator
    def checked(ctx: RunContext[Desk], output: ReceptionReply) -> ReceptionReply:
        desk = ctx.deps
        if output.kind in {ReplyKind.EMERGENCY, ReplyKind.CLINICAL}:
            return output  # the pack's fixed text answers
        if not output.text.strip() and (
            output.kind is ReplyKind.HANDOFF
            or (output.kind is ReplyKind.GREETING and desk.first_turn)
        ):
            return output  # the handoff text, or the introduction, says it all
        known = "\n".join([desk.facts, *desk.seen, desk.conversation, desk.message])
        problems = list(reply_problems(output.text, known, booked=desk.booked))
        if output.kind is ReplyKind.NOTICE and contains_clinical_terms(output.text):
            # Next to the medical notice, the answer is only the practical part (ADR-0006): a word
            # on the health problem there is the model talking health (security review, 2026-10-12).
            problems.append("it talks about the health problem; answer only the practical part")
        if problems:
            raise ModelRetry(
                "Rewrite the reply: " + "; ".join(problems) + ". Use only the clinic facts, the "
                "tool results and the conversation."
            )
        return output

    return agent


def _search(desk: Desk, question: str) -> str:
    reason = check_free_text(question)
    if reason is not None:
        return desk.said(f"Not searched ({reason}): ask without names, numbers or health details.")
    vector = desk.embedder.embed([question])[0]
    with business_session(desk.session_factory, desk.pack.id) as session:
        chunks = hybrid_search(
            session,
            business_id=desk.pack.id,
            query=question,
            query_vector=vector,
            today=desk.today,
            limit=SEARCH_CANDIDATES,
            max_chars=SEARCH_CHARS * 4,
            min_similarity=desk.min_similarity,
        )
    # The conversation's language first: the same document in three languages is three times the
    # reading, and made the run of 2026-10-12 slow (live probe).
    own = [chunk for chunk in chunks if chunk.language == desk.language]
    chosen: list[RetrievedChunk] = []
    size = 0
    for chunk in (own or chunks)[:SEARCH_PASSAGES]:
        if chosen and size + len(chunk.content) > SEARCH_CHARS:
            break
        chosen.append(chunk)
        size += len(chunk.content)
    passages = [neutralized(chunk_evidence(chunk)) for chunk in chosen]
    desk.sources += [item.source_ref.partition("#")[0] for item in passages]
    found = "\n\n".join(f"[{item.title}]\n{item.content}" for item in passages)
    return desk.said(found or "Nothing in the documents answers this.")


def _free_times(
    desk: Desk,
    day: str | None,
    from_day: str | None,
    part: SlotPart | None,
    more: bool,
    doctor_id: str | None,
    service_id: str | None,
) -> str:
    calendar = desk.pack.integrations.calendar
    if desk.scheduler is None or calendar is None or not calendar.chat_booking:
        return desk.said("Bookings are made by reception: give its phone number.")
    target = booking_target(
        desk.pack.bookable_doctors(),
        [doctor_id] if doctor_id else [],
        [service_id] if service_id else [],
    )
    if isinstance(target, ClarifyField):
        return desk.said(f"Ask the patient which {target.value} they mean.")
    if target is None:
        return desk.said("That cannot be booked in the chat: give reception's phone number.")
    wish = SlotWish(day=_day(desk, day), start_from=_day(desk, from_day), part=part)
    earlier = desk.booking
    same = earlier is not None and (earlier.doctor_id, earlier.service_id) == target
    reply = booking_offer(
        desk.context(),
        *target,
        wish=wish,
        shown=earlier.shown if same and earlier is not None else (),
        others=more and same,
    )
    desk.sources.append(f"calendar:{target[0]}")
    desk.booking = reply.next
    return desk.said(reply.text)


def _day(desk: Desk, text: str | None) -> date | None:
    """A day within the calendar's reach, or None."""
    calendar = desk.pack.integrations.calendar
    try:
        day = date.fromisoformat(text.strip()) if text else None
    except ValueError:
        return None
    last = desk.today + timedelta(days=calendar.horizon_days if calendar else 0)
    return day if day is not None and desk.today <= day <= last else None


def _hold_time(desk: Desk, number: int) -> str:
    step = desk.booking
    _calendar_source(desk, step)
    if step is None or not step.offered:
        return desk.said("No times were shown yet: call free_times first.")
    if not 1 <= number <= len(step.offered):
        return desk.said(f"There is no time {number}: the list has {len(step.offered)}.")
    choosing = step.model_copy(update={"stage": BookingStage.CHOOSE, "summary_turn": None})
    reply = take(desk.context(), choosing, step.offered[number - 1])
    desk.booking = reply.next
    if reply.next is None or reply.next.stage is not BookingStage.NAME:
        return desk.said(reply.text)  # taken meanwhile: fresh times
    when = slot_time(step.offered[number - 1].astimezone(desk.zone), desk.language)
    return desk.said(
        f"Held for 15 minutes: {when}. Now ask the patient for their full name and phone number."
    )


def _prepare_booking(desk: Desk, name: str, phone: str) -> str:
    step = desk.booking
    _calendar_source(desk, step)
    if step is None or step.hold_id is None:
        return desk.said("No time is held: call hold_time first.")
    code = country_code(desk.pack)
    full_name, number = valid_name(name), valid_phone(phone, code)
    if full_name is None:
        return desk.said("That is not a first and last name: ask for it again.")
    if number is None:
        return desk.said(
            f"That phone number is not valid: ask for it with the international prefix ({code}…)."
        )
    with business_session(desk.session_factory, desk.pack.id) as session:
        current = held(session, step.hold_id, desk.now)
        if current is not None:
            attach(session, current.id, name=full_name, phone=number)
    if current is None:
        desk.booking = None
        return desk.said("The time is no longer held: call free_times again.")
    desk.booking = step.model_copy(
        update={"stage": BookingStage.CONFIRM, "summary_turn": desk.turn}
    )
    summary = phrase(
        "summary",
        desk.language,
        when=slot_time(current.start.astimezone(desk.zone), desk.language),
        service=desk.catalog.services.get(step.service_id, {}).get(desk.language, step.service_id),
        doctor=desk.catalog.doctors.get(step.doctor_id, step.doctor_id),
        name=full_name,
        phone=number,
    )
    desk.summary = summary
    return desk.said(summary)


def _confirm_booking(desk: Desk) -> str:
    step = desk.booking
    _calendar_source(desk, step)
    if step is None or step.stage is not BookingStage.CONFIRM:
        return desk.said("There is no summary to confirm: call prepare_booking first.")
    if step.summary_turn != desk.turn - 1:
        # A yes counts only as the answer to the summary in the reply just before it.
        return desk.said("The summary was not your last message: call prepare_booking again.")
    if not affirms(desk.message):
        return desk.said("The patient has not said yes in their last message: ask them to confirm.")
    reply: StepReply = confirm(desk.context(), step)
    desk.booking = reply.next
    desk.booked = reply.booked
    desk.booked_text = reply.text if reply.booked else None
    return desk.said(reply.text)


def _cancel_booking(desk: Desk) -> str:
    _calendar_source(desk, desk.booking)
    if desk.booking is not None:
        cancel(desk.context(), desk.booking)
        desk.booking = None
    return desk.said("Nothing is booked; the time is free again.")


def _calendar_source(desk: Desk, step: BookingStep | None) -> None:
    if step is not None:
        desk.sources.append(f"calendar:{step.doctor_id}")
