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

from collections.abc import Sequence
from datetime import timedelta
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from frontdesk.agent.answer_check import numbers_in
from frontdesk.agent.classification import check_free_text
from frontdesk.agent.prompts import instructions_for
from frontdesk.agent.reply_check import Problem, reply_problems, says_booked
from frontdesk.agent.vocab import SlotPart
from frontdesk.conversation import reception_booking as booking
from frontdesk.conversation import reception_hours as hours
from frontdesk.conversation.desk import Desk
from frontdesk.conversation.state import BookingStage, BookingStep, HistoryEntry
from frontdesk.db.engine import business_session
from frontdesk.knowledge.evidence import neutralized
from frontdesk.knowledge.retrieval import RetrievedChunk, chunk_evidence, hybrid_search
from frontdesk.safety.rules import contains_clinical_terms
from frontdesk.tenancy.pack import Language, Profile

RECEPTION_LIMITS: Final = UsageLimits(request_limit=6)
# How a reply the check refused is named in the turn's check reasons, for the run's report.
SENT_BACK: Final = "sent_back:"
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
    text: str = Field(
        description="The message to send the patient, in the language of their last message"
    )
    kind: ReplyKind = Field(
        description="greeting (only a greeting or small talk), answer (you answered, showed "
        "times or asked for the next booking step), clarify (you cannot answer before they say what "
        "they mean), handoff (you send them to reception), notice (you answered the practical part of "
        "a message that describes symptoms), clinical (they ask for medical advice) or emergency"
    )
    language: Language | None = Field(
        default=None,
        description="The language of the patient's last message, which the text is written in: "
        "sq, it or en",
    )


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
    tomorrow = local + timedelta(days=1)
    lines = [
        f"Now: {local:%A} {local:%Y-%m-%d %H:%M} ({desk.pack.timezone}). "
        f"Tomorrow: {tomorrow:%A} {tomorrow:%Y-%m-%d}. "
        f"Language of the conversation so far: {desk.language}.",
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
    if desk.documents:
        lines.append(f"Clinic documents:\n{desk.documents}")
    return "\n\n".join(lines)


def _booking_state(desk: Desk, step: BookingStep) -> str:
    if step.stage is BookingStage.CHOOSE:
        days: dict[str, list[str]] = {}
        for start in step.offered:
            local = start.astimezone(desk.zone)
            days.setdefault(f"{local.date().isoformat()}", []).append(f"{local:%H:%M}")
        listed = "; ".join(f"{day}: {', '.join(times)}" for day, times in days.items())
        return f"the times last shown to the patient, by day: {listed}."
    if step.stage is BookingStage.CONFIRM:
        return "the summary was shown; a yes from the patient lets you call confirm_booking."
    return "a time is held; ask for the full name and phone number, then call prepare_booking."


def build_reception(
    model: Model,
    profile: Profile,
    *,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
    search: bool = True,
) -> Agent[Desk, ReceptionReply]:
    """`instructions` is the prompt text a caller has already read (`read_prompt`). `search`: the
    agent searches the documents; without it they ride in its prompt (Desk.documents)."""
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

    if search:

        @agent.tool
        def search_information(ctx: RunContext[Desk], question: str) -> str:
            """Search the clinic's approved documents: which days a doctor is in, how a visit
            works, what to bring, payment and other practical questions. Use only what the
            passages say.

            Args:
                question: A short practical question, without names, numbers, contact or health
                    details.
            """
            return _search(ctx.deps, question)

    @agent.tool
    def clinic_hours(ctx: RunContext[Desk], day: str | None = None) -> str:
        """When the clinic receives patients, read from the calendar: it has no fixed hours and is
        open when a doctor is in, weekends included. Without a day: the next days.

        Args:
            day: The day asked about, as YYYY-MM-DD ("domani" is Tomorrow in the situation).
        """
        return hours.clinic_hours(ctx.deps, day)

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
        """The free times to book, read from the calendar: every free time of each day it lists,
        and whether other days have free times. Without arguments: the first days with free times.

        Args:
            day: One day the patient asks about, as YYYY-MM-DD.
            from_day: The first day the patient can come, as YYYY-MM-DD ("next week": its Monday).
            part: morning or afternoon, when the patient says so.
            more: True when the patient asks for times other than those already shown.
            doctor_id: The doctor's id from the facts, when the patient names one.
            service_id: The service's id from the facts, when the patient names one.
        """
        return booking.free_times(ctx.deps, day, from_day, part, more, doctor_id, service_id)

    @agent.tool
    def hold_time(ctx: RunContext[Desk], day: str, time: str) -> str:
        """Hold the time the patient picked, among the times free_times gave.

        Args:
            day: Its day, as YYYY-MM-DD.
            time: Its time, as HH:MM.
        """
        return booking.hold_time(ctx.deps, day, time)

    @agent.tool
    def prepare_booking(
        ctx: RunContext[Desk], name: str, phone: str, country: str | None = None
    ) -> str:
        """Put the patient's full name and phone number on the held time. Returns the summary to
        show them, word for word, before they confirm.

        Args:
            name: First and last name, as the patient wrote them.
            phone: The phone number, as the patient wrote it, with or without its prefix.
            country: The two-letter code of the phone's country (IT, AL, GR…) when the patient
                says where they or their phone are from; leave it out otherwise.
        """
        return booking.prepare_booking(ctx.deps, name, phone, country)

    @agent.tool
    def confirm_booking(ctx: RunContext[Desk]) -> str:
        """Book the held time. Only after the patient said yes to the summary, in their last
        message."""
        return booking.confirm_booking(ctx.deps)

    @agent.tool
    def cancel_booking(ctx: RunContext[Desk]) -> str:
        """Let go of the held time, when the patient changes their mind. Nothing is booked."""
        return booking.cancel_booking(ctx.deps)

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
        known = "\n".join([desk.facts, desk.documents, *desk.seen, desk.conversation, desk.message])
        times = booking.shown_times(desk)
        # A booking this conversation made in an earlier turn may be told again; the calendar is
        # read only for a reply that says one is done.
        told = not desk.booked and says_booked(output.text)
        earlier = booking.earlier_slots(desk) if told else ()
        problems = list(
            reply_problems(
                output.text,
                known,
                booked=desk.booked,
                times=times,
                earlier=earlier,
                today=desk.today,
                previous=desk.previous,
                released=desk.released,
            )
        )
        if output.kind is ReplyKind.NOTICE and contains_clinical_terms(output.text):
            # Next to the medical notice, the answer is only the practical part (ADR-0006): a word
            # on the health problem there is the model talking health (security review, 2026-10-12).
            detail = "it talks about the health problem; answer only the practical part"
            problems.append(Problem("health", detail))
        if problems:
            # The codes only: the details hold values of the reply, for the model alone.
            desk.rejected.append(",".join(problem.code for problem in problems))
            raise ModelRetry(
                "Rewrite the reply: "
                + "; ".join(problem.detail for problem in problems)
                + ". Use only the clinic facts, the "
                "tool results and the conversation."
            )
        return output

    return agent


def reply_sources(
    reply: str,
    tool_sources: Sequence[str],
    seen: Sequence[str],
    documents: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """The sources of a reply: what its tools read (a search's documents, the calendar), and, when
    the documents ride in the prompt (key and text each), every one the reply quotes a number of
    that no tool gave (`seen`). A reply no tool explains has every document in the prompt (run
    agent-16: a reply with the doctor's stay and the free times named the calendar alone)."""
    tools = tuple(dict.fromkeys(tool_sources))
    if not tools:
        return tuple(key for key, _ in documents)
    quoted = numbers_in(reply) - numbers_in("\n".join(seen))
    return tools + tuple(key for key, text in documents if quoted & numbers_in(text))


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
