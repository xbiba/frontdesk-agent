"""Responder agent: explains approved documents and never sees the original message (ADR-0009).

Facts from records are written by code. The responder answers from document
passages only, as claims that each carry a verbatim quote; code checks every
claim before anything is sent (Task 19).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import Agent, FunctionToolset, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.deps import ResponderDeps, ToolResult
from frontdesk.agent.prompts import load_prompt
from frontdesk.agent.vocab import ClarifyField
from frontdesk.knowledge.evidence import DOCUMENT_ID_PATTERN, Evidence, render_evidence
from frontdesk.tenancy.pack import Profile

# With the classifier's two requests, at most four model calls per message.
RESPONDER_LIMITS: Final = UsageLimits(request_limit=2, tool_calls_limit=1)
MAX_CLAIMS: Final = 4


class AnswerAction(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    HANDOFF = "handoff"


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=300)
    evidence_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    quote: str = Field(min_length=12, max_length=400)


class ResponderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AnswerAction
    claims: list[Claim] = Field(default_factory=list, max_length=MAX_CLAIMS)
    clarify: ClarifyField | None = None


def build_responder(
    model: Model,
    profile: Profile,
    toolset: FunctionToolset[ResponderDeps],
    *,
    model_settings: ModelSettings | None = None,
) -> Agent[ResponderDeps, ResponderOutput]:
    # The annotation is not enough: an untyped caller could pass "anthropic:..." and skip the gateway.
    if not isinstance(model, Model):
        raise TypeError(
            f"the responder needs a Model object, not {type(model).__name__}: build live models "
            "in the model gateway, because a model-id string would skip its EU pin"
        )
    agent = Agent(
        model,
        name=f"{profile.value}-responder",
        deps_type=ResponderDeps,
        output_type=ResponderOutput,
        instructions=load_prompt(profile, "responder"),
        toolsets=[toolset],
        retries=1,
        model_settings=model_settings,
    )

    @agent.instructions
    def business_context(ctx: RunContext[ResponderDeps]) -> str:
        return f"You answer for {ctx.deps.business_name}. Reply in language '{ctx.deps.language}'."

    return agent


def render_turn_prompt(fields: ValidatedFields, preload: Sequence[Evidence]) -> str:
    lines = [
        f'<request category="{fields.category}" intent="{fields.intent}" language="{fields.language}">'
    ]
    if fields.service_ids:
        lines.append("service_ids: " + ", ".join(fields.service_ids))
    if fields.doctor_ids:
        lines.append("doctor_ids: " + ", ".join(fields.doctor_ids))
    if fields.general_question:
        lines.append(f"question: {fields.general_question}")
    lines.append("</request>")
    if preload:
        lines.append(render_evidence(preload))
    return "\n".join(lines)


def collect_evidence(
    messages: Sequence[ModelMessage], preload: Sequence[Evidence]
) -> list[Evidence]:
    found: dict[str, Evidence] = {item.evidence_id: item for item in preload}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                result = read_tool_result(part.content)
                if result is not None:
                    found.update({item.evidence_id: item for item in result.evidence})
    return list(found.values())


def read_tool_result(content: object) -> ToolResult | None:
    """A tool return as ToolResult: the object in-process, its JSON shape after serialization."""
    if isinstance(content, ToolResult):
        return content
    try:
        return ToolResult.model_validate(content)
    except ValidationError:
        return None
