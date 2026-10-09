import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.prompts import prompt_version
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import MAX_MESSAGE_CHARS, TurnOutcome, handle_message
from frontdesk.observability.logs import MetadataOnlyFormatter
from frontdesk.observability.tracing import ALLOWED_ATTRIBUTES, record_metadata, redact, turn_span
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack, Profile
from tests.conversation.helpers import classifier_model, document_responder, make_runtime

Activate = Callable[..., sessionmaker[Session]]
MESSAGE_SENTINEL = "SENTINELMSG"
DOCUMENT_SENTINEL = "SENTINELDOC"
ERROR_SENTINEL = "SENTINELERR"
QUESTION = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "Is there parking?",
}
PRICE = {"category": "administrative", "intent": "price", "service_ids": ["physiotherapy"]}
PRICES = {  # osteopathy has no price record
    "category": "administrative",
    "intent": "price",
    "service_ids": ["physiotherapy", "osteopathy"],
}
RETURNS = {
    "category": "administrative",
    "intent": "general_question",
    "general_question": "How do I return an item?",
}
# A session factory bound to no database: a long message must not reach the database either.
NO_DATABASE: sessionmaker[Session] = sessionmaker(expire_on_commit=False)


def _pack_with_sentinel(businesses_dir: Path, tmp_path: Path) -> BusinessPack:
    target = tmp_path / "demo-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", target)
    parking = target / "knowledge" / "parking-en.md"
    parking.write_text(
        parking.read_text(encoding="utf-8").replace(
            "behind the building.", f"behind the building {DOCUMENT_SENTINEL}."
        ),
        encoding="utf-8",
    )
    return load_pack(target)


def _exported(exporter: InMemorySpanExporter) -> str:
    return json.dumps(
        [
            {
                "name": span.name,
                "attributes": dict(span.attributes or {}),
                "events": [
                    {"name": e.name, "attributes": dict(e.attributes or {})} for e in span.events
                ],
                "status": span.status.description,
            }
            for span in exporter.get_finished_spans()
        ],
        default=str,
    )


def _logs(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(MetadataOnlyFormatter().format(record) for record in caplog.records)


def _turn_attributes(exporter: InMemorySpanExporter) -> dict[str, Any]:
    """The attributes of the one span a turn records, all of them from the allowlist."""
    (span,) = exporter.get_finished_spans()
    assert span.name == "frontdesk.turn"
    attributes = dict(span.attributes or {})
    assert set(attributes) <= {f"frontdesk.{key}" for key in ALLOWED_ATTRIBUTES}
    return attributes


def test_redaction_masks_phones_and_emails() -> None:
    assert redact("call +355691234567 or a1234567@b.example") == "call [phone] or [email]"


def test_failed_spans_record_only_the_exception_type(span_exporter: InMemorySpanExporter) -> None:
    span_exporter.clear()
    with pytest.raises(RuntimeError), turn_span("demo-clinic"):
        raise RuntimeError(ERROR_SENTINEL)
    assert ERROR_SENTINEL not in _exported(span_exporter)
    assert span_exporter.get_finished_spans()[-1].status.description == "RuntimeError"


@pytest.mark.db
async def test_default_telemetry_carries_no_content(
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    activate_pack: Activate,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    span_exporter.clear()
    pack = _pack_with_sentinel(businesses_dir, tmp_path)
    runtime = make_runtime(
        pack, activate_pack(pack), classifier_model(QUESTION), document_responder()
    )
    with caplog.at_level(logging.INFO):
        result, _ = await handle_message(
            f"Is there parking? {MESSAGE_SENTINEL}", ConversationState(turns=1), runtime
        )
    assert DOCUMENT_SENTINEL in result.text  # the reply itself does carry content
    exported = _exported(span_exporter)
    logs = _logs(caplog)
    assert '"frontdesk.outcome": "replied"' in exported
    assert '"event": "turn.completed"' in logs
    for sentinel in (MESSAGE_SENTINEL, DOCUMENT_SENTINEL):
        assert sentinel not in exported
        assert sentinel not in logs


@pytest.mark.db
async def test_provoked_errors_leave_no_content(
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    activate_pack: Activate,
    clinic_pack: BusinessPack,
) -> None:
    def failing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name="test", body=ERROR_SENTINEL)

    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(QUESTION), FunctionModel(failing)
    )
    with caplog.at_level(logging.INFO):
        result, _ = await handle_message(
            f"Is there parking? {MESSAGE_SENTINEL}", ConversationState(turns=1), runtime
        )
    assert result.outcome is TurnOutcome.HANDOFF
    exported = _exported(span_exporter)
    logs = _logs(caplog)
    assert '"reason": "ModelHTTPError"' in logs
    for sentinel in (MESSAGE_SENTINEL, ERROR_SENTINEL):
        assert sentinel not in exported
        assert sentinel not in logs


@pytest.mark.db
async def test_unexpected_errors_leave_no_content(
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"failed on {MESSAGE_SENTINEL}")

    monkeypatch.setattr("frontdesk.conversation.turn.fact_evidence", broken)
    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(PRICE), document_responder()
    )
    with caplog.at_level(logging.INFO):
        result, _ = await handle_message(
            "Quanto costa la fisioterapia?", ConversationState(turns=1), runtime
        )
    assert result.outcome is TurnOutcome.HANDOFF
    logs = _logs(caplog)
    assert '"reason": "RuntimeError"' in logs
    assert MESSAGE_SENTINEL not in logs
    assert MESSAGE_SENTINEL not in _exported(span_exporter)


@pytest.mark.db
@pytest.mark.parametrize("evidence_id", ["D-5e471e1d", "SENTINELID"])
async def test_ids_the_model_invents_never_reach_telemetry(
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    evidence_id: str,
) -> None:
    def invented(passage_id: str, content: str) -> dict[str, str]:
        sentence = content.split(". ")[0]
        return {"text": sentence, "evidence_id": evidence_id, "quote": sentence}

    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(QUESTION),
        document_responder(claim=invented),
    )
    with caplog.at_level(logging.INFO):
        result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    exported = _exported(span_exporter)
    logs = _logs(caplog)
    assert evidence_id not in exported
    assert evidence_id not in logs


@pytest.mark.db
async def test_content_capture_is_explicit_and_redacted(
    span_exporter: InMemorySpanExporter, activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack,
        activate_pack(clinic_pack),
        classifier_model(None),
        document_responder(),
        capture_content=True,
    )
    await handle_message("my number is +355691234567", ConversationState(turns=1), runtime)
    exported = _exported(span_exporter)
    assert "frontdesk.content" in exported
    assert "[phone]" in exported
    assert "691234567" not in exported


# What a turn span does carry, and what holds around it. The tests above show what it never carries.


@pytest.mark.db
async def test_the_span_names_the_prompt_versions_but_carries_no_content(
    span_exporter: InMemorySpanExporter,
    activate_pack: Activate,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    span_exporter.clear()
    pack = _pack_with_sentinel(businesses_dir, tmp_path)
    runtime = make_runtime(
        pack, activate_pack(pack), classifier_model(QUESTION), document_responder()
    )
    result, _ = await handle_message(
        f"Is there parking? {MESSAGE_SENTINEL}", ConversationState(turns=1), runtime
    )
    assert DOCUMENT_SENTINEL in result.text
    attributes = _turn_attributes(span_exporter)
    assert attributes["frontdesk.classifier_prompt_version"] == prompt_version(
        Profile.RECEPTIONIST, "classifier"
    )
    assert attributes["frontdesk.responder_prompt_version"] == prompt_version(
        Profile.RECEPTIONIST, "responder"
    )
    exported = _exported(span_exporter)
    for sentinel in (MESSAGE_SENTINEL, DOCUMENT_SENTINEL):
        assert sentinel not in exported


@pytest.mark.db
async def test_the_prompt_versions_are_those_of_the_packs_profile(
    span_exporter: InMemorySpanExporter, activate_pack: Activate, shop_pack: BusinessPack
) -> None:
    span_exporter.clear()
    runtime = make_runtime(
        shop_pack, activate_pack(shop_pack), classifier_model(RETURNS), document_responder()
    )
    await handle_message("How do I return an item?", ConversationState(turns=1), runtime)
    attributes = _turn_attributes(span_exporter)
    classifier = attributes["frontdesk.classifier_prompt_version"]
    responder = attributes["frontdesk.responder_prompt_version"]
    assert classifier == prompt_version(Profile.SUPPORT_SALES, "classifier")
    assert responder == prompt_version(Profile.SUPPORT_SALES, "responder")
    assert classifier != prompt_version(Profile.RECEPTIONIST, "classifier")
    assert responder != prompt_version(Profile.RECEPTIONIST, "responder")


@pytest.mark.db
@pytest.mark.parametrize(
    ("classified", "message"),
    [
        pytest.param(PRICE, "Quanto costa la fisioterapia?", id="record"),
        pytest.param(PRICES, "Quanto costano fisioterapia e osteopatia?", id="unanswered-item"),
        pytest.param(QUESTION, "Is there parking?", id="document"),
    ],
)
async def test_verified_evidence_ids_are_recorded_and_source_references_are_not(
    span_exporter: InMemorySpanExporter,
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    classified: dict[str, Any],
    message: str,
) -> None:
    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(classified), document_responder()
    )
    result, _ = await handle_message(message, ConversationState(turns=1), runtime)
    trace = result.trace
    assert trace.evidence_ids
    assert trace.sources  # the turn's trace names them; the span must not
    attributes = _turn_attributes(span_exporter)
    assert attributes["frontdesk.outcome"] == result.outcome
    assert attributes["frontdesk.evidence_ids"] == trace.evidence_ids
    assert attributes["frontdesk.invalid_evidence_ids"] == 0
    exported = _exported(span_exporter)
    assert result.text.splitlines()[0] not in exported  # the reply is not telemetry
    for source in trace.sources:
        assert source not in exported


def _citing(evidence_id: str) -> Callable[[str, str], dict[str, str]]:
    """A claim builder that cites `evidence_id` whatever passage it was given."""

    def claim(passage_id: str, content: str) -> dict[str, str]:
        sentence = content.split(". ")[0]
        return {"text": sentence, "evidence_id": evidence_id, "quote": sentence}

    return claim


def _clarifying(field: str) -> FunctionModel:
    """A responder that always asks the person for `field`."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        args = {"action": "clarify", "clarify": field}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(model)


@pytest.mark.db
@pytest.mark.parametrize(
    ("responder", "invalid", "reasons", "rejection"),
    [
        # Well formed, but no passage has it: check_answer counts it and gives its reasons.
        pytest.param(
            document_responder(claim=_citing("D-5e471e1d")),
            1,
            ("unknown_evidence", "no_claims"),
            None,
            id="unknown-id",
        ),
        # Not even well formed: the agent refuses the output, so there is nothing to count.
        pytest.param(
            document_responder(claim=_citing("SENTINELID")),
            0,
            (),
            "responder_error",
            id="malformed-id",
        ),
        pytest.param(
            _clarifying("order_details"),
            0,
            ("clarify_not_offered",),
            None,
            id="clarify-not-offered",
        ),
    ],
)
async def test_a_rejected_answer_records_its_codes_and_the_count_of_invalid_ids(
    span_exporter: InMemorySpanExporter,
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    responder: FunctionModel,
    invalid: int,
    reasons: tuple[str, ...],
    rejection: str | None,
) -> None:
    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack, activate_pack(clinic_pack), classifier_model(QUESTION), responder
    )
    result, _ = await handle_message("Is there parking?", ConversationState(turns=1), runtime)
    assert result.outcome is TurnOutcome.HANDOFF
    attributes = _turn_attributes(span_exporter)
    assert attributes["frontdesk.outcome"] == "handoff"
    assert attributes["frontdesk.invalid_evidence_ids"] == invalid
    assert attributes["frontdesk.evidence_ids"] == ()  # nothing cited was verified
    assert attributes["frontdesk.check_reasons"] == reasons
    assert attributes.get("frontdesk.rejection") == rejection
    exported = _exported(span_exporter)
    assert "D-5e471e1d" not in exported
    assert "SENTINELID" not in exported


async def test_a_long_emergency_message_is_traced_by_its_rule_without_its_text(
    span_exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture, clinic_pack: BusinessPack
) -> None:
    span_exporter.clear()
    runtime = make_runtime(clinic_pack, NO_DATABASE, classifier_model(None), document_responder())
    message = f"{'Vorrei sapere gli orari. ' * 100}{MESSAGE_SENTINEL}. Aiuto, mi fa male il petto"
    assert len(message) > MAX_MESSAGE_CHARS
    with caplog.at_level(logging.INFO):
        result, _ = await handle_message(message, ConversationState(), runtime)  # the first turn
    assert result.outcome is TurnOutcome.HANDOFF
    attributes = _turn_attributes(span_exporter)
    assert attributes["frontdesk.rule"] == "emergency"
    assert attributes["frontdesk.category"] == "emergency"
    assert attributes["frontdesk.outcome"] == "handoff"
    assert attributes["frontdesk.language"] == "it"
    assert MESSAGE_SENTINEL not in _exported(span_exporter)
    assert MESSAGE_SENTINEL not in _logs(caplog)


def test_record_metadata_drops_what_is_not_allowlisted(span_exporter: InMemorySpanExporter) -> None:
    span_exporter.clear()
    with turn_span("demo-clinic") as span:
        record_metadata(
            span,
            intent="price",
            evidence_ids=["R-0a1b2c3d"],
            invalid_evidence_ids=0,
            rejection=None,
            message=MESSAGE_SENTINEL,
            sources=["price:physiotherapy"],
        )
    (finished,) = span_exporter.get_finished_spans()
    assert dict(finished.attributes or {}) == {
        "frontdesk.business_id": "demo-clinic",
        "frontdesk.intent": "price",
        "frontdesk.evidence_ids": ("R-0a1b2c3d",),
        "frontdesk.invalid_evidence_ids": 0,
    }
    assert MESSAGE_SENTINEL not in _exported(span_exporter)


# Redaction runs on the whole message, which can be 100,000 characters long.
PLAIN_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
ADDRESS_PIECES = ["a", "_", ".", "+", "-", "@", " ", "é", "x@y.z", "+x@y.z"]


@given(st.lists(st.sampled_from(ADDRESS_PIECES), max_size=24).map("".join))
@example("x@y.z+x@y.z")
def test_redaction_masks_the_addresses_a_plain_pattern_finds(text: str) -> None:
    """The pieces hold no digits, so no phone number is masked."""
    assert redact(text) == PLAIN_EMAIL.sub("[email]", text)


def test_digits_inside_an_address_do_not_make_it_a_phone_number() -> None:
    assert redact("write to 12345678@b.example") == "write to [email]"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a" * 100_000, id="one-long-word"),
        pytest.param("a@" + "a" * 100_000, id="one-long-domain"),
        pytest.param("a@" * 50_000, id="many-at-signs"),
        pytest.param("a@b.co+" * 14_000, id="addresses-joined-by-plus"),
        pytest.param("1 " * 50_000, id="digits"),
    ],
)
def test_redaction_takes_time_linear_in_the_length(text: str) -> None:
    started = time.perf_counter()
    redact(text)
    assert time.perf_counter() - started < 1.0


async def test_content_capture_of_a_very_long_message_takes_linear_time(
    span_exporter: InMemorySpanExporter, clinic_pack: BusinessPack
) -> None:
    span_exporter.clear()
    runtime = make_runtime(
        clinic_pack, NO_DATABASE, classifier_model(None), document_responder(), capture_content=True
    )
    started = time.perf_counter()
    await handle_message("a" * 100_000, ConversationState(turns=1), runtime)
    assert time.perf_counter() - started < 1.0
    assert "frontdesk.content" in _exported(span_exporter)
