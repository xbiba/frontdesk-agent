"""Serializable agent contracts: deps and tool results can cross activity boundaries.

Plain data, so that a durable runtime could run them as activities (ADR-0011).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from frontdesk.agent.classification import ValidatedFields
from frontdesk.knowledge.evidence import Evidence
from frontdesk.tenancy.pack import Language


class ResponderDeps(BaseModel):
    model_config = ConfigDict(frozen=True)

    business_id: str
    business_name: str
    language: Language
    fields: ValidatedFields


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence: list[Evidence] = Field(default_factory=list)
    note: str | None = None
