"""One conversation turn (ADR-0006, ADR-0009).

precheck → deterministic rules → classifier → validation → facts written by
code, or document evidence → responder → claim checks. Model failures, other
errors and timeouts end in fixed texts and a handoff; only the type of an
error is logged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Final

from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.answer_check import check_answer
from frontdesk.agent.classification import (
    ORDER_REFERENCE,
    POSTAL_CODE,
    ClassifierOutput,
    Rejection,
    ValidatedFields,
    validate_classification,
)
from frontdesk.agent.classifier import CLASSIFIER_LIMITS, ClassifierDeps, classifier_deps
from frontdesk.agent.deps import ResponderDeps
from frontdesk.agent.responder import (
    RESPONDER_LIMITS,
    AnswerAction,
    ResponderOutput,
    collect_evidence,
    render_turn_prompt,
)
from frontdesk.agent.vocab import ADMINISTRATIVE_INTENTS, Category, ClarifyField, Intent
from frontdesk.conversation.compose import (
    FACT_INTENTS,
    clarify_question,
    compose_facts,
    unanswered,
)
from frontdesk.conversation.preload import document_evidence, fact_evidence
from frontdesk.conversation.state import ConversationState
from frontdesk.db.engine import business_session
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.knowledge.facts import Catalog, load_catalog
from frontdesk.observability.logs import log_event
from frontdesk.observability.tracing import record_content, record_metadata, turn_span
from frontdesk.safety.language import detect_language
from frontdesk.safety.rules import Verdict, emergency_rule, evaluate, rule_text_length
from frontdesk.shop.attempts import consume_order_lookup
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, Language, Profile

logger = logging.getLogger(__name__)
MAX_MESSAGE_CHARS: Final = 2000
TURN_TIMEOUT_SECONDS: Final = 45.0
_NOTICES: Final = {
    Category.CLINICAL: FixedTextKey.CLINICAL,
    Category.UNCERTAIN: FixedTextKey.HANDOFF,
}
_SMALL_TALK: Final = frozenset({Intent.GREETING, Intent.OTHER})
# The clarifying questions each profile's responder prompt offers (agents/<profile>/prompts/
# responder.md); the shop's prompt has no clarify action. A test pins this table to the prompts.
RESPONDER_CLARIFY_FIELDS: Final[dict[Profile, frozenset[ClarifyField]]] = {
    Profile.RECEPTIONIST: frozenset({ClarifyField.SERVICE, ClarifyField.DOCTOR}),
    Profile.SUPPORT_SALES: frozenset(),
}


class TurnOutcome(StrEnum):
    REPLIED = "replied"
    ASKED = "asked"
    FIXED = "fixed"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class TurnTrace:
    """Metadata about a turn, never message text or evidence content, with one exception.

    `evidence_ids` lists documents and records. Live evidence (an order's status) is only counted
    in `live_evidence`: its ID is derived from the order reference, so it is never listed.
    `sources` can hold `order:<reference>`, and the reference is the customer's own data: a caller
    never logs, stores or exports it. The terminal chat hides it, and offline reports keep it
    because they hold synthetic data only.
    """

    rule: str
    category: Category | None = None
    intent: Intent | None = None
    service_ids: tuple[str, ...] = ()
    doctor_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    live_evidence: int = 0
    sources: tuple[str, ...] = ()
    check_reasons: tuple[str, ...] = ()
    invalid_evidence_ids: int = 0
    rejection: str | None = None
    clarify: ClarifyField | None = None


@dataclass(frozen=True)
class DraftClaim:
    text: str
    evidence_id: str
    quote: str


@dataclass(frozen=True)
class RejectedDraft:
    """The responder's answer that a check refused, as the model wrote it.

    Kept only when the runtime asks for it (`keep_drafts`): the live evaluation shows it to the
    person who judges whether the refusal was right. It is model text derived from a message, so
    nothing logs, traces or stores it otherwise.
    """

    action: str
    claims: tuple[DraftClaim, ...]
    clarify: str | None


@dataclass(frozen=True)
class TurnResult:
    text: str
    outcome: TurnOutcome
    language: Language
    trace: TurnTrace
    draft: RejectedDraft | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TurnRuntime:
    """What a turn runs on. The prompt versions are those of the texts the agents were built
    with (`read_prompt`): a turn reads no prompt file, so editing one changes nothing here."""

    pack: BusinessPack
    classifier: Agent[ClassifierDeps, ClassifierOutput]
    responder: Agent[ResponderDeps, ResponderOutput]
    classifier_prompt_version: str
    responder_prompt_version: str
    session_factory: sessionmaker[Session]
    embedder: Embedder
    today: Callable[[], date]
    min_support_similarity: float = 0.5
    min_retrieval_similarity: float = 0.0
    capture_content: bool = False
    timeout_seconds: float = TURN_TIMEOUT_SECONDS
    orders: OrderStore | None = None
    now: Callable[[], datetime] = _utc_now
    keep_drafts: bool = False  # the live evaluation keeps refused answers for its reviewer


async def handle_message(
    text: str, state: ConversationState, runtime: TurnRuntime
) -> tuple[TurnResult, ConversationState]:
    with turn_span(runtime.pack.id) as span:
        result = await _run_turn(text, state, runtime)
        trace = result.trace
        record_metadata(
            span,
            rule=trace.rule,
            category=trace.category,
            intent=trace.intent,
            outcome=result.outcome,
            language=result.language,
            evidence_ids=list(trace.evidence_ids),
            live_evidence=trace.live_evidence,
            invalid_evidence_ids=trace.invalid_evidence_ids,
            check_reasons=list(trace.check_reasons),
            rejection=trace.rejection,
            classifier_prompt_version=runtime.classifier_prompt_version,
            responder_prompt_version=runtime.responder_prompt_version,
        )
        record_content(span, enabled=runtime.capture_content, message=text, reply=result.text)
    _log_turn(result, runtime.pack.id)
    return result, _next_state(state, result, runtime.pack)


async def _run_turn(text: str, state: ConversationState, runtime: TurnRuntime) -> TurnResult:
    pack = runtime.pack
    language = detect_language(
        text, current=state.language, supported=pack.languages, default=pack.default_language
    )
    try:
        async with asyncio.timeout(runtime.timeout_seconds):
            result = await _decide(text, language, state, runtime)
    except TimeoutError:
        result = _handoff(pack, language, FixedTextKey.HANDOFF, TurnTrace(rule="timeout"))
    except Exception as error:  # any failure becomes a handoff; only its type is logged
        log_event(logger, "turn.failed", business_id=pack.id, reason=type(error).__name__)
        result = _handoff(pack, language, FixedTextKey.HANDOFF, TurnTrace(rule="error"))
    if state.turns == 0:
        result = _with_disclosure(pack, language, result)
    return result


async def _decide(
    text: str, language: Language, state: ConversationState, runtime: TurnRuntime
) -> TurnResult:
    pack = runtime.pack
    stripped = text.strip()
    if not any(char.isalnum() for char in stripped):
        return _fixed(pack, language, FixedTextKey.CLARIFICATION, TurnTrace(rule="empty"))
    # The length as written first, so a very long message is not folded only to be measured.
    if len(stripped) > MAX_MESSAGE_CHARS or rule_text_length(stripped) > MAX_MESSAGE_CHARS:
        # Too long for the other rules (not linear in the length), but not for the emergency ones.
        found = emergency_rule(stripped)
        if found is None:
            return _handoff(pack, language, FixedTextKey.HANDOFF, TurnTrace(rule="too_long"))
        return _emergency(pack, language, found)
    rule = evaluate(stripped)
    if rule.verdict is Verdict.EMERGENCY:
        return _emergency(pack, language, rule.rule)
    if rule.verdict is Verdict.CLINICAL_ADVICE:
        return _handoff(pack, language, FixedTextKey.CLINICAL, TurnTrace(rule=rule.rule))
    if rule.verdict is Verdict.PERSONAL_DATA:
        trace = TurnTrace(rule=f"personal_data:{rule.rule}")
        return _fixed(pack, language, FixedTextKey.PERSONAL_DATA, trace)
    catalog = await asyncio.to_thread(_load_catalog, runtime)
    if catalog is None:
        trace = TurnTrace(rule="pass", rejection="no_active_knowledge")
        return _handoff(pack, language, FixedTextKey.HANDOFF, trace)
    classified = await _classify(stripped, language, state, catalog, runtime)
    if isinstance(classified, Rejection):
        trace = TurnTrace(rule="pass", rejection=classified.reason)
        return _handoff(pack, language, FixedTextKey.HANDOFF, trace)
    return await _route(classified, state, catalog, runtime)


async def _classify(
    text: str, language: Language, state: ConversationState, catalog: Catalog, runtime: TurnRuntime
) -> ValidatedFields | Rejection:
    deps = classifier_deps(
        catalog,
        business_name=runtime.pack.name,
        language=language,
        previous_intents=state.previous_intents,
        topic_service_ids=state.topic_service_ids,
        topic_doctor_ids=state.topic_doctor_ids,
        pending_clarification=state.pending_clarification,
    )
    try:
        run = await runtime.classifier.run(text, deps=deps, usage_limits=CLASSIFIER_LIMITS)
    except AgentRunError as error:
        log_event(
            logger, "classifier.failed", business_id=runtime.pack.id, reason=type(error).__name__
        )
        return Rejection(reason="classifier_error")
    return validate_classification(
        run.output,
        message=text,
        catalog=catalog,
        profile=runtime.pack.profile,
        features=runtime.pack.features,
        language=language,
    )


async def _route(
    fields: ValidatedFields, state: ConversationState, catalog: Catalog, runtime: TurnRuntime
) -> TurnResult:
    pack = runtime.pack
    language = fields.language
    trace = TurnTrace(
        rule="pass",
        category=fields.category,
        intent=fields.intent,
        service_ids=fields.service_ids,
        doctor_ids=fields.doctor_ids,
    )
    if fields.category is Category.EMERGENCY:
        return _handoff(pack, language, FixedTextKey.EMERGENCY, trace)
    notice = _NOTICES.get(fields.category)
    if fields.intent in {Intent.HUMAN, Intent.BOOKING}:
        # Bookings arrive with M6; until then reception takes them. A clinical notice stays.
        return _handoff(pack, language, notice or FixedTextKey.HANDOFF, trace)
    if notice is None:
        if fields.category is Category.ADMINISTRATIVE and fields.intent in _SMALL_TALK:
            return _fixed(pack, language, FixedTextKey.GREETING, trace)
        return await _answer(fields, state, catalog, runtime, trace)
    if fields.intent not in ADMINISTRATIVE_INTENTS:
        return _handoff(pack, language, notice, trace)
    # A clinical message with an administrative question: answer only that question, then hand off.
    reply = await _answer(fields, state, catalog, runtime, trace, with_closing=False)
    prefix = reply.text if reply.outcome is TurnOutcome.REPLIED else None
    # A question that was not shown to the person is not a pending one.
    trace = replace(reply.trace, clarify=None)
    return _handoff(pack, language, notice, trace, prefix=prefix, draft=reply.draft)


async def _answer(
    fields: ValidatedFields,
    state: ConversationState,
    catalog: Catalog,
    runtime: TurnRuntime,
    trace: TurnTrace,
    *,
    with_closing: bool = True,
) -> TurnResult:
    pack = runtime.pack
    language = fields.language
    next_step = _next_step(pack, fields, state) if with_closing else None
    if fields.intent is Intent.ORDER_STATUS:
        return await _order_reply(
            fields, state, catalog, runtime, trace, with_closing=with_closing, next_step=next_step
        )
    if fields.category is Category.ORIENTATION or fields.intent in FACT_INTENTS:
        missing = _missing_field(fields)
        if missing is not None:
            return _clarify(catalog, language, missing, trace)
        facts = await asyncio.to_thread(_facts, fields, runtime)
        return _facts_reply(
            pack, fields, facts, catalog, trace, with_closing=with_closing, next_step=next_step
        )
    return await _document_answer(fields, catalog, runtime, trace, next_step=next_step)


def _next_step(pack: BusinessPack, fields: ValidatedFields, state: ConversationState) -> str | None:
    """The pack's next step (how to book) for this answer: once per conversation, and never after
    the contact itself, which the step would only repeat."""
    if state.next_step_shown or fields.intent is Intent.CONTACT:
        return None
    return pack.fixed_texts[fields.language].get(FixedTextKey.NEXT_STEP)


async def _order_reply(
    fields: ValidatedFields,
    state: ConversationState,
    catalog: Catalog,
    runtime: TurnRuntime,
    trace: TurnTrace,
    *,
    with_closing: bool,
    next_step: str | None,
) -> TurnResult:
    pack = runtime.pack
    language = fields.language
    if runtime.orders is None:
        return _handoff(pack, language, FixedTextKey.HANDOFF, trace)
    details = _order_details(fields)
    if details is None:
        return _clarify(catalog, language, ClarifyField.ORDER_DETAILS, trace)
    reference, postal_code = details
    now = runtime.now()
    # Committed in its own transaction before the lookup: a later failure keeps it used.
    if not await asyncio.to_thread(_consume_order_lookup, runtime, state.conversation_id, now):
        exhausted = replace(trace, rejection="order_lookups_exhausted")
        return _handoff(pack, language, FixedTextKey.HANDOFF, exhausted)
    found = runtime.orders.lookup(reference, postal_code, language=language, now=now)
    facts = [found] if found is not None else []
    return _facts_reply(
        pack, fields, facts, catalog, trace, with_closing=with_closing, next_step=next_step
    )


def _order_details(fields: ValidatedFields) -> tuple[str, str] | None:
    """The order number and postal code, if both are present and exactly in their form.

    `fullmatch` on the raw strings: the patterns end in `$`, which `match` lets a trailing newline
    pass. Anything else counts as missing, so it is asked for again and uses no attempt.
    """
    reference, postal_code = fields.order_reference, fields.postal_code
    if reference is None or postal_code is None:
        return None
    if ORDER_REFERENCE.fullmatch(reference) is None or POSTAL_CODE.fullmatch(postal_code) is None:
        return None
    return reference, postal_code


def _missing_field(fields: ValidatedFields) -> ClarifyField | None:
    if fields.category is Category.ORIENTATION:
        return None
    if fields.intent in {Intent.PRICE, Intent.SERVICE_INFO} and not fields.service_ids:
        return ClarifyField.SERVICE
    if fields.intent is Intent.DOCTOR_INFO and not fields.doctor_ids:
        return ClarifyField.DOCTOR
    return None


def _facts_reply(
    pack: BusinessPack,
    fields: ValidatedFields,
    facts: Sequence[Evidence],
    catalog: Catalog,
    trace: TurnTrace,
    *,
    with_closing: bool,
    next_step: str | None,
) -> TurnResult:
    language = fields.language
    listed = [item for item in facts if item.kind is not EvidenceKind.LIVE]
    checked = replace(
        trace,
        evidence_ids=tuple(item.evidence_id for item in listed),
        live_evidence=len(facts) - len(listed),
        sources=_sources(facts),
    )
    no_information = _fixed_text(pack, language, FixedTextKey.NO_INFORMATION)
    absent = [
        f"{_asked_name(catalog, fields, item)}: {no_information}"
        for item in unanswered(fields, facts)
    ]
    text = compose_facts(
        fields, facts, with_closing=with_closing and not absent, next_step=next_step, missing=absent
    )
    if text is None:
        return _handoff(pack, language, FixedTextKey.NO_INFORMATION, checked)
    if absent:
        # Part of the question has no record: say which part, then how to reach a person.
        return _handoff(pack, language, FixedTextKey.HANDOFF, checked, prefix=text)
    return TurnResult(text, TurnOutcome.REPLIED, language, checked)


def _asked_name(catalog: Catalog, fields: ValidatedFields, item: str) -> str:
    if fields.intent is Intent.DOCTOR_INFO:
        return catalog.doctors.get(item, item)
    return catalog.services.get(item, {}).get(fields.language, item)


async def _document_answer(
    fields: ValidatedFields,
    catalog: Catalog,
    runtime: TurnRuntime,
    trace: TurnTrace,
    *,
    next_step: str | None,
) -> TurnResult:
    pack = runtime.pack
    language = fields.language
    preload = await asyncio.to_thread(_documents, fields, catalog, runtime)
    deps = ResponderDeps(
        business_id=pack.id, business_name=pack.name, language=language, fields=fields
    )
    try:
        run = await runtime.responder.run(
            render_turn_prompt(fields, preload, today=runtime.today()),
            deps=deps,
            usage_limits=RESPONDER_LIMITS,
        )
    except AgentRunError as error:
        log_event(logger, "responder.failed", business_id=pack.id, reason=type(error).__name__)
        return _handoff(
            pack, language, FixedTextKey.HANDOFF, replace(trace, rejection="responder_error")
        )
    output = run.output
    check = await asyncio.to_thread(
        check_answer,
        output,
        collect_evidence(run.all_messages(), preload),
        embedder=runtime.embedder,
        min_similarity=runtime.min_support_similarity,
    )
    checked = replace(
        trace,
        evidence_ids=tuple(item.evidence_id for item in check.cited),
        sources=_sources(check.cited),
        check_reasons=check.reasons,
        invalid_evidence_ids=check.invalid_ids,
    )
    draft = _draft(output) if runtime.keep_drafts else None
    if not check.ok or output.action is AnswerAction.HANDOFF:
        refused = draft if not check.ok else None
        return _handoff(pack, language, FixedTextKey.NO_INFORMATION, checked, draft=refused)
    if output.action is AnswerAction.CLARIFY:
        return _responder_clarify(pack, catalog, language, output.clarify, checked, draft=draft)
    text = check.text
    if _names_a_contact(check.cited, pack):
        # The passage sends the person to the business: the system writes the contact from the
        # pack, as it writes every contact; the model was told never to (live review, 2026-10-11).
        text = f"{text}\n\n{_fixed_text(pack, language, FixedTextKey.HANDOFF)}"
    elif next_step:
        text = f"{text}\n\n{next_step}"
    return TurnResult(text, TurnOutcome.REPLIED, language, checked)


def _names_a_contact(cited: Sequence[Evidence], pack: BusinessPack) -> bool:
    """Some cited passage holds the business's phone or email."""
    phone = "".join(char for char in pack.location.phone if char.isdigit())
    email = pack.location.email.strip().casefold()
    for item in cited:
        digits = "".join(char for char in item.content if char.isdigit())
        if (email and email in item.content.casefold()) or (len(phone) >= 6 and phone in digits):
            return True
    return False


def _responder_clarify(
    pack: BusinessPack,
    catalog: Catalog,
    language: Language,
    field: ClarifyField | None,
    trace: TurnTrace,
    *,
    draft: RejectedDraft | None,
) -> TurnResult:
    """Ask the question the responder chose, if its profile's prompt offers it."""
    if field is None or field not in RESPONDER_CLARIFY_FIELDS[pack.profile]:
        # Any other question is a rejected answer: it fails like a bad claim.
        reasons = (*trace.check_reasons, "clarify_not_offered")
        rejected = replace(trace, check_reasons=reasons)
        return _handoff(pack, language, FixedTextKey.NO_INFORMATION, rejected, draft=draft)
    return _clarify(catalog, language, field, trace)


def _draft(output: ResponderOutput) -> RejectedDraft:
    claims = tuple(DraftClaim(c.text, c.evidence_id, c.quote) for c in output.claims)
    clarify = output.clarify.value if output.clarify is not None else None
    return RejectedDraft(action=output.action.value, claims=claims, clarify=clarify)


def _clarify(
    catalog: Catalog, language: Language, field: ClarifyField, trace: TurnTrace
) -> TurnResult:
    text = clarify_question(field, language, catalog)
    return TurnResult(text, TurnOutcome.ASKED, language, replace(trace, clarify=field))


def _sources(items: Sequence[Evidence]) -> tuple[str, ...]:
    """Readable provenance: record keys and document keys, without chunk IDs."""
    return tuple(dict.fromkeys(item.source_ref.partition("#")[0] for item in items))


def _load_catalog(runtime: TurnRuntime) -> Catalog | None:
    with business_session(runtime.session_factory, runtime.pack.id) as session:
        return load_catalog(session, runtime.pack.id)


def _consume_order_lookup(runtime: TurnRuntime, conversation_id: str, now: datetime) -> bool:
    with business_session(runtime.session_factory, runtime.pack.id) as session:
        return consume_order_lookup(
            session, business_id=runtime.pack.id, conversation_id=conversation_id, now=now
        )


def _facts(fields: ValidatedFields, runtime: TurnRuntime) -> list[Evidence]:
    with business_session(runtime.session_factory, runtime.pack.id) as session:
        return fact_evidence(session, fields, business_id=runtime.pack.id, today=runtime.today())


def _documents(fields: ValidatedFields, catalog: Catalog, runtime: TurnRuntime) -> list[Evidence]:
    with business_session(runtime.session_factory, runtime.pack.id) as session:
        return document_evidence(
            session,
            fields,
            business_id=runtime.pack.id,
            catalog=catalog,
            embedder=runtime.embedder,
            today=runtime.today(),
            min_similarity=runtime.min_retrieval_similarity,
        )


def _log_turn(result: TurnResult, business_id: str) -> None:
    trace = result.trace
    log_event(
        logger,
        "turn.completed",
        business_id=business_id,
        rule=trace.rule,
        category=trace.category,
        intent=trace.intent,
        outcome=result.outcome,
        evidence_ids=list(trace.evidence_ids),
        live_evidence=trace.live_evidence,
        invalid_evidence_ids=trace.invalid_evidence_ids,
        check_reasons=list(trace.check_reasons),
        reason=trace.rejection,
        language=result.language,
    )


def _next_state(
    state: ConversationState, result: TurnResult, pack: BusinessPack
) -> ConversationState:
    trace = result.trace
    step = pack.fixed_texts[result.language].get(FixedTextKey.NEXT_STEP)
    return state.advanced(
        language=result.language,
        intent=trace.intent,
        category=trace.category,
        service_ids=trace.service_ids,
        doctor_ids=trace.doctor_ids,
        clarify=trace.clarify,
        next_step_shown=step is not None and step in result.text,
    )


def _fixed_text(pack: BusinessPack, language: Language, key: FixedTextKey) -> str:
    return pack.fixed_texts[language][key]


def _fixed(
    pack: BusinessPack, language: Language, key: FixedTextKey, trace: TurnTrace
) -> TurnResult:
    return TurnResult(_fixed_text(pack, language, key), TurnOutcome.FIXED, language, trace)


def _handoff(
    pack: BusinessPack,
    language: Language,
    reason: FixedTextKey,
    trace: TurnTrace,
    *,
    prefix: str | None = None,
    draft: RejectedDraft | None = None,
) -> TurnResult:
    """The reason's fixed text, then how to reach a person. Nothing is promised (ADR-0006)."""
    parts = [prefix] if prefix else []
    if reason is not FixedTextKey.HANDOFF:
        parts.append(_fixed_text(pack, language, reason))
    parts.append(_fixed_text(pack, language, FixedTextKey.HANDOFF))
    return TurnResult("\n\n".join(parts), TurnOutcome.HANDOFF, language, trace, draft)


def _emergency(pack: BusinessPack, language: Language, rule: str) -> TurnResult:
    # The category lets the next state drop the topic, as for an emergency the classifier found.
    trace = TurnTrace(rule=rule, category=Category.EMERGENCY)
    return _handoff(pack, language, FixedTextKey.EMERGENCY, trace)


def _with_disclosure(pack: BusinessPack, language: Language, result: TurnResult) -> TurnResult:
    """The first reply says the assistant is automatic (ADR-0006). An emergency still reads first."""
    disclosure = _fixed_text(pack, language, FixedTextKey.DISCLOSURE)
    emergency = _fixed_text(pack, language, FixedTextKey.EMERGENCY)
    if result.trace.category is Category.EMERGENCY and result.text.startswith(emergency):
        contact = result.text.removeprefix(emergency)  # "\n\n" and the contact text
        return replace(result, text=f"{emergency}\n\n{disclosure}{contact}")
    return replace(result, text=f"{disclosure}\n\n{result.text}")
