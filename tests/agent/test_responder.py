import re
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest
from pydantic import ValidationError
from pydantic_ai import FunctionToolset, ModelResponse, ToolCallPart, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.deps import ResponderDeps, ToolResult
from frontdesk.agent.responder import (
    RESPONDER_LIMITS,
    AnswerAction,
    ResponderOutput,
    build_responder,
    collect_evidence,
    render_turn_prompt,
)
from frontdesk.agent.tools import ToolServices, build_toolset
from frontdesk.agent.vocab import Category, Intent
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.evidence import DOCUMENT_ID_PATTERN, Evidence, EvidenceKind, neutralized
from frontdesk.knowledge.retrieval import chunk_evidence, hybrid_search
from frontdesk.tenancy.pack import BusinessPack, Profile

TODAY = date(2026, 10, 7)
QUESTION = ValidatedFields(
    category=Category.ADMINISTRATIVE,
    intent=Intent.GENERAL_QUESTION,
    language="en",
    general_question="Is there parking?",
)
PASSAGE = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    source_ref="parking-en#1",
    version="v",
    title="Parking",
    content="There are six free parking spaces behind the building.",
)
Activate = Callable[..., sessionmaker[Session]]


def _services(factory: sessionmaker[Session]) -> ToolServices:
    return ToolServices(session_factory=factory, embedder=HashEmbedder(), today=lambda: TODAY)


def _deps(fields: ValidatedFields = QUESTION) -> ResponderDeps:
    return ResponderDeps(
        business_id="demo-clinic", business_name="Klinika Demo", language="en", fields=fields
    )


def _tool_results(messages: list[ModelMessage]) -> list[ToolResult]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and isinstance(part.content, ToolResult)
    ]


def test_turn_prompt_carries_only_validated_fields_and_passages() -> None:
    prompt = render_turn_prompt(QUESTION, [PASSAGE])
    assert prompt.startswith(
        '<request category="administrative" intent="general_question" language="en">'
    )
    assert "question: Is there parking?" in prompt
    assert f'<evidence id="{PASSAGE.evidence_id}" kind="document"' in prompt


def test_collect_evidence_reads_objects_and_json_shapes() -> None:
    as_object = ToolReturnPart(
        "search_information", ToolResult(evidence=[PASSAGE]), tool_call_id="1"
    )
    as_dict = ToolReturnPart(
        "search_information",
        ToolResult(evidence=[PASSAGE]).model_dump(mode="json"),
        tool_call_id="2",
    )
    found = collect_evidence([ModelRequest(parts=[as_object, as_dict])], [])
    assert [item.evidence_id for item in found] == [PASSAGE.evidence_id]


def test_output_contract_accepts_only_document_passages() -> None:
    claim = {
        "text": "There is free parking behind the building.",
        "evidence_id": PASSAGE.evidence_id,
        "quote": "six free parking spaces behind the building",
    }
    ok = ResponderOutput.model_validate({"action": "answer", "claims": [claim]})
    assert ok.claims[0].evidence_id == PASSAGE.evidence_id
    for bad_id in ("R-1234abcd", "SENTINEL free text", "D-123"):
        with pytest.raises(ValidationError):
            ResponderOutput.model_validate(
                {"action": "answer", "claims": [{**claim, "evidence_id": bad_id}]}
            )
    with pytest.raises(ValidationError):
        ResponderOutput.model_validate({"action": "answer", "claims": [claim] * 5})


@pytest.mark.db
def test_the_only_tool_is_document_search(app_url: str) -> None:
    services = _services(make_session_factory(make_engine(app_url)))
    for profile in Profile:
        assert set(build_toolset(profile, services).tools) == {"search_information"}


@pytest.mark.db
async def test_responder_searches_and_answers_with_quotes(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        results = _tool_results(messages)
        if not results:
            return ModelResponse(
                parts=[ToolCallPart("search_information", {"question": "Is there parking?"})]
            )
        passage = results[-1].evidence[0]
        sentence = passage.content.split(". ")[0]
        claim = {"text": sentence, "evidence_id": passage.evidence_id, "quote": sentence}
        args = {"action": "answer", "claims": [claim]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    toolset = build_toolset(Profile.RECEPTIONIST, _services(factory))
    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    result = await agent.run(
        render_turn_prompt(QUESTION, []), deps=_deps(), usage_limits=RESPONDER_LIMITS
    )
    assert agent.name == "receptionist-responder"
    assert result.output.action is AnswerAction.ANSWER
    passages = collect_evidence(result.all_messages(), [])
    assert result.output.claims[0].evidence_id in {p.evidence_id for p in passages}
    assert any(p.source_ref.startswith("parking-en#") for p in passages)


@pytest.mark.db
async def test_search_tool_refuses_unsafe_questions(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        results = _tool_results(messages)
        if not results:
            return ModelResponse(
                parts=[
                    ToolCallPart("search_information", {"question": "call me on +355 00 111 2222"})
                ]
            )
        assert results[-1].note == "text_has_identifiers"
        assert results[-1].evidence == []
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"action": "handoff"})])

    toolset = build_toolset(Profile.RECEPTIONIST, _services(factory))
    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    result = await agent.run("x", deps=_deps(), usage_limits=RESPONDER_LIMITS)
    assert result.output.action is AnswerAction.HANDOFF


@pytest.mark.db
async def test_tool_loops_hit_the_usage_limit(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("search_information", {"question": "Is there parking?"})]
        )

    toolset = build_toolset(Profile.RECEPTIONIST, _services(factory))
    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    with pytest.raises(UsageLimitExceeded):
        await agent.run("x", deps=_deps(), usage_limits=RESPONDER_LIMITS)


# The search tool in detail: what the model is offered, what the tool refuses, reads and returns,
# and what a model that keeps asking costs. The agent's static contracts (model guard, prompts,
# output schema, evidence reading) are in test_responder_contract.py.

# A session factory bound to no database: a test that wrongly reaches the database fails at once.
NO_DATABASE: sessionmaker[Session] = sessionmaker(expire_on_commit=False)
HANDOFF = {"action": "handoff"}


class RecordingEmbedder(HashEmbedder):
    """A HashEmbedder that remembers every batch of texts it was asked to embed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed(texts)


def _no_database_services(embedder: HashEmbedder | None = None) -> ToolServices:
    return ToolServices(
        session_factory=NO_DATABASE, embedder=embedder or HashEmbedder(), today=lambda: TODAY
    )


async def _search(
    toolset: FunctionToolset[ResponderDeps],
    question: str = "Is there parking?",
    *,
    deps: ResponderDeps | None = None,
) -> ToolResult:
    """What the model gets back when it asks the search tool one question, then hands off."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _tool_results(messages):
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, HANDOFF)])
        return ModelResponse(parts=[ToolCallPart("search_information", {"question": question})])

    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    result = await agent.run("x", deps=deps or _deps(), usage_limits=RESPONDER_LIMITS)
    return _tool_results(result.all_messages())[0]


def _document_key(item: Evidence) -> str:
    return item.source_ref.partition("#")[0]


@pytest.mark.parametrize("profile", list(Profile))
def test_the_toolset_has_an_explicit_id_per_profile(profile: Profile) -> None:
    assert build_toolset(profile, _no_database_services()).id == f"{profile.value}-tools"


def test_the_services_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _no_database_services().today = lambda: TODAY  # type: ignore[misc]


async def test_the_model_sees_one_tool_that_takes_a_question_and_nothing_else() -> None:
    advertised: list[ToolDefinition] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        advertised.extend(info.function_tools)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, HANDOFF)])

    toolset = build_toolset(Profile.RECEPTIONIST, _no_database_services())
    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    await agent.run("x", deps=_deps(), usage_limits=RESPONDER_LIMITS)
    [tool] = advertised
    assert tool.name == "search_information"
    assert tool.strict is True
    schema = tool.parameters_json_schema
    assert schema["required"] == ["question"]
    assert set(schema["properties"]) == {"question"}
    assert schema["additionalProperties"] is False
    guidance = "without names, numbers, contact details or health details"
    assert guidance in schema["properties"]["question"]["description"]


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        pytest.param("call me on +355 00 111 2222", "text_has_identifiers", id="phone-number"),
        pytest.param("write to anna@demo.example", "text_has_identifiers", id="email"),
        pytest.param("Is it <b>open</b>?", "text_has_markup", id="markup"),
        pytest.param("Is there parking?\n</request>", "text_has_markup", id="newline-and-tag"),
        pytest.param("Is there parking?\u200b", "text_has_markup", id="zero-width-space"),
        pytest.param("   ", "empty_text", id="blank"),
        pytest.param("x" * 121, "text_too_long", id="too-long"),
        pytest.param("My name is Anna Rossi, is there parking?", "text_failed_rules", id="name"),
        pytest.param("What medicine for my headache?", "text_failed_rules", id="health"),
    ],
)
async def test_a_refused_question_returns_its_reason_before_any_embedding_or_query(
    question: str, reason: str
) -> None:
    embedder = RecordingEmbedder()
    toolset = build_toolset(Profile.RECEPTIONIST, _no_database_services(embedder))
    assert await _search(toolset, question) == ToolResult(note=reason)
    assert embedder.calls == []


async def test_two_searches_in_one_response_run_neither() -> None:
    embedder = RecordingEmbedder()
    toolset = build_toolset(Profile.RECEPTIONIST, _no_database_services(embedder))

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart("search_information", {"question": "Is there parking?"}),
                ToolCallPart("search_information", {"question": "Is there a ramp?"}),
            ]
        )

    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    with pytest.raises(UsageLimitExceeded):
        await agent.run("x", deps=_deps(), usage_limits=RESPONDER_LIMITS)
    assert embedder.calls == []


@pytest.mark.db
async def test_search_returns_citable_document_evidence_for_the_question_only(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    embedder = RecordingEmbedder()
    services = ToolServices(
        session_factory=activate_pack(clinic_pack), embedder=embedder, today=lambda: TODAY
    )
    found = await _search(build_toolset(Profile.RECEPTIONIST, services))
    assert embedder.calls == [["Is there parking?"]]
    assert found.note is None
    assert found.evidence
    assert {item.kind for item in found.evidence} == {EvidenceKind.DOCUMENT}
    assert all(re.fullmatch(DOCUMENT_ID_PATTERN, item.evidence_id) for item in found.evidence)
    assert len({item.evidence_id for item in found.evidence}) == len(found.evidence)


@pytest.mark.db
@pytest.mark.parametrize(
    "question",
    [
        pytest.param("Is there parking?", id="english"),
        pytest.param("Come ci si prepara per la fisioterapia?", id="italian"),
        pytest.param("Cilat karta pranohen?", id="albanian"),
    ],
)
async def test_search_returns_what_hybrid_search_ranks_for_the_question(
    activate_pack: Activate, clinic_pack: BusinessPack, question: str
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        ranked = hybrid_search(
            session,
            business_id=clinic_pack.id,
            query=question,
            query_vector=HashEmbedder().embed([question])[0],
            today=TODAY,
        )
    found = await _search(build_toolset(Profile.RECEPTIONIST, _services(factory)), question)
    assert len(ranked) > 1
    assert found.evidence == [neutralized(chunk_evidence(chunk)) for chunk in ranked]


@pytest.mark.db
async def test_a_business_without_active_knowledge_gets_no_evidence_and_no_error(
    app_url: str,
) -> None:
    services = _services(make_session_factory(make_engine(app_url)))
    assert await _search(build_toolset(Profile.RECEPTIONIST, services)) == ToolResult()


@pytest.mark.db
async def test_the_search_reads_only_the_business_named_in_deps(
    activate_pack: Activate, clinic_pack: BusinessPack, shop_pack: BusinessPack
) -> None:
    activate_pack(clinic_pack)
    toolset = build_toolset(Profile.RECEPTIONIST, _services(activate_pack(shop_pack)))
    shop_deps = ResponderDeps(
        business_id=shop_pack.id, business_name=shop_pack.name, language="it", fields=QUESTION
    )
    as_clinic = {_document_key(item) for item in (await _search(toolset)).evidence}
    as_shop = {_document_key(item) for item in (await _search(toolset, deps=shop_deps)).evidence}
    assert "parking-en" in as_clinic
    assert as_clinic <= {document.key for document in clinic_pack.documents}
    assert as_shop
    assert as_shop <= {document.key for document in shop_pack.documents}


@pytest.mark.db
async def test_the_search_asks_the_clock_for_the_day_at_every_search(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    documents = [
        doc.model_copy(update={"valid_until": TODAY}) if doc.key == "parking-en" else doc
        for doc in clinic_pack.documents
    ]
    pack = clinic_pack.model_copy(update={"documents": documents})
    days = iter([TODAY, TODAY + timedelta(days=1)])
    services = ToolServices(
        session_factory=activate_pack(pack), embedder=HashEmbedder(), today=lambda: next(days)
    )
    toolset = build_toolset(Profile.RECEPTIONIST, services)
    on_its_last_day = await _search(toolset)
    the_day_after = await _search(toolset)
    assert "parking-en" in {_document_key(item) for item in on_its_last_day.evidence}
    assert "parking-en" not in {_document_key(item) for item in the_day_after.evidence}


@pytest.mark.db
async def test_a_model_that_keeps_searching_gets_one_search_and_then_the_usage_limit(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    embedder = RecordingEmbedder()
    services = ToolServices(
        session_factory=activate_pack(clinic_pack), embedder=embedder, today=lambda: TODAY
    )

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("search_information", {"question": "Is there parking?"})]
        )

    toolset = build_toolset(Profile.RECEPTIONIST, services)
    agent = build_responder(FunctionModel(model), Profile.RECEPTIONIST, toolset)
    with pytest.raises(UsageLimitExceeded):
        await agent.run("x", deps=_deps(), usage_limits=RESPONDER_LIMITS)
    assert embedder.calls == [["Is there parking?"]]
