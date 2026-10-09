"""Execution context: who a unit of work belongs to, always built by code."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    business_id: str
    conversation_id: str
    correlation_id: str
    agent_version: str
