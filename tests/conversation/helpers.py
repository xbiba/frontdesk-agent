"""Deterministic stand-ins for the two models, used by the pipeline tests."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from pydantic_ai import ModelResponse, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classifier import build_classifier
from frontdesk.agent.prompts import read_prompt
from frontdesk.agent.responder import build_responder
from frontdesk.agent.tools import ToolServices, build_toolset
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.knowledge.embedders import Embedder, HashEmbedder
from frontdesk.tenancy.pack import BusinessPack

TODAY = date(2026, 10, 7)
PASSAGE = re.compile(
    r'<evidence id="(D-[0-9a-f]{8})" kind="document"[^>]*>\n(.*?)\n</evidence>', re.DOTALL
)
ClaimBuilder = Callable[[str, str], dict[str, Any]]


def first_prompt(messages: Sequence[ModelMessage]) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ""


def _output(info: AgentInfo, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])


def classifier_model(
    args: dict[str, Any] | None, prompts: list[str] | None = None
) -> FunctionModel:
    """Answers with `args`; records each prompt it receives. `None` means it must not run."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if prompts is not None:
            prompts.append(first_prompt(messages))
        if args is None:
            raise AssertionError("the classifier must not be called for this message")
        return _output(info, args)

    return FunctionModel(model)


def classifier_turns(outputs: list[dict[str, Any]], instructions: list[str]) -> FunctionModel:
    """One output per turn, in order; records the instructions of each call."""
    remaining = list(outputs)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        seen = request.instructions if isinstance(request, ModelRequest) else None
        instructions.append(seen or "")
        return _output(info, remaining.pop(0))

    return FunctionModel(model)


def _verbatim(passage_id: str, content: str) -> dict[str, Any]:
    sentence = content.split(". ")[0].rstrip(".")
    return {"text": f"{sentence}.", "evidence_id": passage_id, "quote": sentence}


def document_responder(
    prompts: list[str] | None = None, *, claim: ClaimBuilder | None = None
) -> FunctionModel:
    """Answers from the first passage in the prompt; `claim` builds the claim from (id, content)."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = first_prompt(messages)
        if prompts is not None:
            prompts.append(prompt)
        match = PASSAGE.search(prompt)
        if match is None:
            return _output(info, {"action": "handoff"})
        build = claim or _verbatim
        return _output(
            info, {"action": "answer", "claims": [build(match.group(1), match.group(2))]}
        )

    return FunctionModel(model)


class SpyEmbedder(HashEmbedder):
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        return super().embed(texts)


def make_runtime(
    pack: BusinessPack,
    factory: sessionmaker[Session],
    classifier: FunctionModel,
    responder: FunctionModel,
    embedder: Embedder | None = None,
    *,
    capture_content: bool = False,
    timeout_seconds: float = 45.0,
) -> TurnRuntime:
    used = embedder or HashEmbedder()
    services = ToolServices(session_factory=factory, embedder=used, today=lambda: TODAY)
    classifier_prompt = read_prompt(pack.profile, "classifier")
    responder_prompt = read_prompt(pack.profile, "responder")
    return TurnRuntime(
        pack=pack,
        classifier=build_classifier(classifier, pack.profile, instructions=classifier_prompt.text),
        responder=build_responder(
            responder,
            pack.profile,
            build_toolset(pack.profile, services),
            instructions=responder_prompt.text,
        ),
        classifier_prompt_version=classifier_prompt.version,
        responder_prompt_version=responder_prompt.version,
        session_factory=factory,
        embedder=used,
        today=lambda: TODAY,
        capture_content=capture_content,
        timeout_seconds=timeout_seconds,
    )
