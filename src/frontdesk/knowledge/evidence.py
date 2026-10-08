"""Typed provenance (spec §7): document passage, approved record, live tool result."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

DOCUMENT_ID_PATTERN: Final = r"^D-[0-9a-f]{8}$"


class EvidenceKind(StrEnum):
    DOCUMENT = "document"
    RECORD = "record"
    LIVE = "live"


_PREFIX = {EvidenceKind.DOCUMENT: "D", EvidenceKind.RECORD: "R", EvidenceKind.LIVE: "L"}


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    kind: EvidenceKind
    source_ref: str
    version: str
    title: str
    content: str
    values: dict[str, str] = Field(default_factory=dict)
    public_url: str | None = None

    @classmethod
    def create(
        cls,
        *,
        kind: EvidenceKind,
        source_ref: str,
        version: str,
        title: str,
        content: str,
        values: dict[str, str] | None = None,
        public_url: str | None = None,
    ) -> Evidence:
        digest = hashlib.sha256(f"{kind}|{source_ref}|{version}".encode()).hexdigest()[:8]
        return cls(
            evidence_id=f"{_PREFIX[kind]}-{digest}",
            kind=kind,
            source_ref=source_ref,
            version=version,
            title=title,
            content=content,
            values=values or {},
            public_url=public_url,
        )


def render_evidence(items: Sequence[Evidence]) -> str:
    """Evidence blocks for prompts. Content is data, never instructions (blueprint SEC-002)."""
    return "\n".join(
        f'<evidence id="{item.evidence_id}" kind="{item.kind}" title="{_attribute(item.title)}">\n'
        f"{_escape(item.content)}\n</evidence>"
        for item in items
    )


def _attribute(value: str) -> str:
    return _escape(value).replace('"', "'")


def _escape(value: str) -> str:
    return value.replace("<", "\u2039").replace(">", "\u203a")  # single angle quotes
