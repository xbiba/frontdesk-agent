"""Versioned knowledge with approval.

draft -> approved -> active -> retired. Only an approved version can become
active, and embeddings are computed at activation, after approval. Every
transition is one guarded UPDATE, so a version that is not in the expected state is refused;
two concurrent activations are stopped by the unique index on active versions, and the one that
loses gets an `IntegrityError`.
Each version records which embedder computed its vectors: vectors from two
models are never compared (ADR-0005).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import groupby
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from frontdesk.db.models import EMBEDDING_DIM, Chunk, Document, KnowledgeVersion, Record
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import Embedder, NoEmbedder
from frontdesk.safety.normalize import normalize
from frontdesk.tenancy.pack import BusinessPack

_KNOWLEDGE_FIELDS = {
    "location",
    "services",
    "departments",
    "doctors",
    "prices",
    "orientation",
    "orientation_map",
    "documents",
}
_TEXT_INDENT = " " * 6  # deeper than any report line, so a line of text cannot pass for one


class VersionStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    ACTIVE = "active"
    RETIRED = "retired"


class KnowledgeStateError(ValueError):
    """A knowledge version is not in the state an operation requires."""


class EmbedderMismatchError(KnowledgeStateError):
    """The active knowledge was embedded by another model than the configured one."""


@dataclass(frozen=True)
class DraftResult:
    version_id: uuid.UUID
    unchanged: bool


def pack_digest(pack: BusinessPack) -> str:
    content = pack.model_dump(mode="json", include=_KNOWLEDGE_FIELDS)
    encoded = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def active_version_id(session: Session, business_id: str) -> uuid.UUID | None:
    return session.scalar(
        select(KnowledgeVersion.id).where(
            KnowledgeVersion.business_id == business_id,
            KnowledgeVersion.status == VersionStatus.ACTIVE,
        )
    )


def version_status(session: Session, version_id: uuid.UUID) -> VersionStatus | None:
    status = session.scalar(
        select(KnowledgeVersion.status).where(KnowledgeVersion.id == version_id)
    )
    return VersionStatus(status) if status is not None else None


def create_draft(session: Session, pack: BusinessPack) -> DraftResult:
    digest = pack_digest(pack)
    existing = session.scalar(
        select(KnowledgeVersion.id).where(
            KnowledgeVersion.business_id == pack.id,
            KnowledgeVersion.source_digest == digest,
            KnowledgeVersion.status.in_(
                [VersionStatus.DRAFT, VersionStatus.APPROVED, VersionStatus.ACTIVE]
            ),
        )
    )
    if existing is not None:
        return DraftResult(version_id=existing, unchanged=True)
    version = KnowledgeVersion(
        business_id=pack.id, status=VersionStatus.DRAFT, source_digest=digest
    )
    session.add(version)
    session.flush()
    _add_documents(session, pack, version.id)
    _add_records(session, pack, version.id)
    session.flush()
    return DraftResult(version_id=version.id, unchanged=False)


def approve(
    session: Session, version_id: uuid.UUID, *, approved_by: str, evidence: str, now: datetime
) -> None:
    if not approved_by.strip() or not evidence.strip():
        raise KnowledgeStateError("approval needs who approved and the evidence of approval")
    _transition(
        session,
        version_id,
        from_status=VersionStatus.DRAFT,
        values={
            "status": VersionStatus.APPROVED,
            "approved_by": approved_by.strip(),
            "approval_evidence": evidence.strip(),
            "approved_at": now,
        },
    )


def activate(session: Session, version_id: uuid.UUID, embedder: Embedder, *, now: datetime) -> None:
    business_id = session.scalar(
        select(KnowledgeVersion.business_id).where(
            KnowledgeVersion.id == version_id, KnowledgeVersion.status == VersionStatus.APPROVED
        )
    )
    if business_id is None:
        raise KnowledgeStateError(f"knowledge version {version_id} is not approved")
    _embed_chunks(session, version_id, embedder)
    _retire_active(session, business_id, now)
    _transition(
        session,
        version_id,
        from_status=VersionStatus.APPROVED,
        values={
            "status": VersionStatus.ACTIVE,
            "embedder": embedder.identity,
            "activated_at": now,
            "retired_at": None,
        },
    )


def rollback(session: Session, business_id: str, embedder: Embedder, *, now: datetime) -> uuid.UUID:
    previous = session.execute(
        select(KnowledgeVersion.id, KnowledgeVersion.embedder)
        .where(
            KnowledgeVersion.business_id == business_id,
            KnowledgeVersion.status == VersionStatus.RETIRED,
            KnowledgeVersion.approved_by.is_not(None),
        )
        .order_by(KnowledgeVersion.retired_at.desc())
        .limit(1)
    ).one_or_none()
    if previous is None:
        raise KnowledgeStateError("no previous approved version to roll back to")
    previous_id: uuid.UUID = previous.id
    if previous.embedder != embedder.identity:
        _clear_vectors(session, previous_id)
        _embed_chunks(session, previous_id, embedder)
    _retire_active(session, business_id, now)
    _transition(
        session,
        previous_id,
        from_status=VersionStatus.RETIRED,
        values={
            "status": VersionStatus.ACTIVE,
            "embedder": embedder.identity,
            "activated_at": now,
            "retired_at": None,
        },
    )
    return previous_id


def check_embedder(session: Session, business_id: str, embedder: Embedder) -> None:
    active = session.execute(
        select(KnowledgeVersion.id, KnowledgeVersion.embedder).where(
            KnowledgeVersion.business_id == business_id,
            KnowledgeVersion.status == VersionStatus.ACTIVE,
        )
    ).one_or_none()
    if active is not None and active.embedder != embedder.identity:
        raise EmbedderMismatchError(
            f"active knowledge {active.id} was embedded with '{active.embedder}', "
            f"this deployment uses '{embedder.identity}': run `frontdesk knowledge reembed`"
        )


def reembed_active(session: Session, business_id: str, embedder: Embedder) -> int:
    version_id = active_version_id(session, business_id)
    if version_id is None:
        raise KnowledgeStateError("no active knowledge version to re-embed")
    _clear_vectors(session, version_id)
    count = _embed_chunks(session, version_id, embedder)
    session.execute(
        update(KnowledgeVersion)
        .where(KnowledgeVersion.id == version_id)
        .values(embedder=embedder.identity)
    )
    return count


def review_report(session: Session, version_id: uuid.UUID) -> str:
    """What the owner reads before approving: the version against the active one.

    An added or removed document or record is listed with all its values, a changed one with each
    changed value before and after. A document's text is shown whole when the document is added or
    removed and as a unified diff when it changed; its metadata stays ordinary fields.
    """
    version = session.get(KnowledgeVersion, version_id)
    if version is None:
        raise KnowledgeStateError(f"knowledge version {version_id} does not exist")
    active = active_version_id(session, version.business_id)
    old_docs = _documents(session, active) if active else {}
    old_records = _records(session, active) if active else {}
    return "\n".join(
        [
            f"# Knowledge version {version_id}",
            "",
            f"- Business: {version.business_id}",
            f"- Status: {version.status}",
            f"- Compared with active version: {active or 'none'}",
            "",
            "## Documents",
            *_diff_lines(old_docs, _documents(session, version_id)),
            "",
            "## Records",
            *_diff_lines(old_records, _records(session, version_id)),
        ]
    )


def _transition(
    session: Session, version_id: uuid.UUID, *, from_status: VersionStatus, values: dict[str, Any]
) -> None:
    result = cast(
        CursorResult[Any],
        session.execute(
            update(KnowledgeVersion)
            .where(KnowledgeVersion.id == version_id, KnowledgeVersion.status == from_status)
            .values(**values)
        ),
    )
    if result.rowcount != 1:
        raise KnowledgeStateError(f"knowledge version {version_id} is not {from_status}")


def _retire_active(session: Session, business_id: str, now: datetime) -> None:
    session.execute(
        update(KnowledgeVersion)
        .where(
            KnowledgeVersion.business_id == business_id,
            KnowledgeVersion.status == VersionStatus.ACTIVE,
        )
        .values(status=VersionStatus.RETIRED, retired_at=now)
    )


def _add_documents(session: Session, pack: BusinessPack, version_id: uuid.UUID) -> None:
    for item in pack.documents:
        document = Document(
            business_id=pack.id,
            knowledge_version_id=version_id,
            doc_key=item.key,
            title=item.title,
            language=item.language,
            public_url=item.public_url,
            valid_from=item.valid_from,
            valid_until=item.valid_until,
            content_hash=hashlib.sha256(item.body.encode("utf-8")).hexdigest(),
        )
        session.add(document)
        session.flush()
        for piece in chunk_markdown(item.title, item.body):
            session.add(
                Chunk(
                    business_id=pack.id,
                    knowledge_version_id=version_id,
                    document_id=document.id,
                    ordinal=piece.ordinal,
                    heading=piece.heading,
                    content=piece.content,
                    search_text=normalize(f"{piece.heading} {piece.content}"),
                )
            )


def _add_records(session: Session, pack: BusinessPack, version_id: uuid.UUID) -> None:
    rows: list[tuple[str, str, dict[str, Any]]] = [
        ("location", "main", pack.location.model_dump(mode="json"))
    ]
    rows += [("service", s.id, s.model_dump(mode="json")) for s in pack.services]
    rows += [("department", d.id, d.model_dump(mode="json")) for d in pack.departments]
    rows += [("doctor", d.id, d.model_dump(mode="json")) for d in pack.doctors]
    rows += [("price", p.service_id, p.model_dump(mode="json")) for p in pack.prices]
    rows += [("orientation", o.id, o.model_dump(mode="json")) for o in pack.orientation]
    if pack.orientation_map is not None:
        rows.append(("orientation_map", "approval", pack.orientation_map.model_dump(mode="json")))
    for kind, key, payload in rows:
        session.add(
            Record(
                business_id=pack.id,
                knowledge_version_id=version_id,
                kind=kind,
                record_key=key,
                payload=payload,
            )
        )


def _clear_vectors(session: Session, version_id: uuid.UUID) -> None:
    session.execute(
        update(Chunk).where(Chunk.knowledge_version_id == version_id).values(embedding=None)
    )


def _embed_chunks(session: Session, version_id: uuid.UUID, embedder: Embedder) -> int:
    if embedder.identity == NoEmbedder.identity:
        return 0  # the knowledge rides in the prompt: no vectors
    rows = session.execute(
        select(Chunk.id, Chunk.heading, Chunk.content).where(
            Chunk.knowledge_version_id == version_id, Chunk.embedding.is_(None)
        )
    ).all()
    if not rows:
        return 0
    vectors = embedder.embed([f"{row.heading}\n{row.content}" for row in rows])
    if any(len(vector) != EMBEDDING_DIM for vector in vectors):
        raise KnowledgeStateError("embedding dimension does not match the schema")
    for row, vector in zip(rows, vectors, strict=True):
        session.execute(update(Chunk).where(Chunk.id == row.id).values(embedding=vector))
    return len(rows)


@dataclass(frozen=True)
class _Entry:
    """A document or a record as stored, ready to be compared with its other version."""

    fields: dict[str, str]
    wording: str | None = None  # a document's text as its chunks hold it
    content_hash: str | None = None  # a document's content hash


def _documents(session: Session, version_id: uuid.UUID) -> dict[str, _Entry]:
    chunks = _chunks_by_document(session, version_id)
    rows = session.scalars(select(Document).where(Document.knowledge_version_id == version_id))
    return {
        row.doc_key: _Entry(
            fields=_flatten(
                {
                    "title": row.title,
                    "language": row.language,
                    "public_url": row.public_url,
                    "valid_from": row.valid_from,
                    "valid_until": row.valid_until,
                }
            ),
            wording=_wording(row.title, chunks.get(row.id, [])),
            content_hash=row.content_hash,
        )
        for row in rows
    }


def _chunks_by_document(
    session: Session, version_id: uuid.UUID
) -> dict[uuid.UUID, list[tuple[str, str]]]:
    rows = session.execute(
        select(Chunk.document_id, Chunk.heading, Chunk.content)
        .where(Chunk.knowledge_version_id == version_id)
        .order_by(Chunk.document_id, Chunk.ordinal)
    ).all()
    return {
        document_id: [(row.heading, row.content) for row in group]
        for document_id, group in groupby(rows, key=lambda row: row.document_id)
    }


def _wording(title: str, chunks: list[tuple[str, str]]) -> str:
    """A document's text as its chunks hold it: a '## section' line per section, then paragraphs.

    The chunker keeps no heading levels and no extra blank lines, so this is the text the bot can
    quote, not a byte-for-byte copy of the file; a paragraph that was cut in the middle shows the
    cut as a paragraph break.
    """
    blocks: list[str] = []
    previous = ""
    for heading, content in chunks:
        section = "" if heading == title else heading.removeprefix(f"{title} > ")
        starts_section = section != "" and section != previous
        blocks.append(f"## {section}\n{content}" if starts_section else content)
        previous = section
    return "\n\n".join(blocks)


def _records(session: Session, version_id: uuid.UUID) -> dict[str, _Entry]:
    rows = session.execute(
        select(Record.kind, Record.record_key, Record.payload).where(
            Record.knowledge_version_id == version_id
        )
    ).all()
    return {f"{row.kind}:{row.record_key}": _Entry(fields=_flatten(row.payload)) for row in rows}


def _flatten(value: Any, path: str = "") -> dict[str, str]:
    """Nested values as dotted fields: {"conditions": {"it": "x"}} -> {"conditions.it": "x"}."""
    if value is None:
        return {}
    if isinstance(value, dict):
        pairs: list[tuple[Any, Any]] = list(value.items())
    elif isinstance(value, list):
        pairs = list(enumerate(value))
    else:
        return {path: str(value)}
    flat: dict[str, str] = {}
    for key, item in pairs:
        flat |= _flatten(item, f"{path}.{key}" if path else str(key))
    return flat


def _diff_lines(old: dict[str, _Entry], new: dict[str, _Entry]) -> list[str]:
    """What the owner approves: what is added or removed, with its values, and what changed."""
    lines: list[str] = []
    for key in sorted(old.keys() | new.keys()):
        if key not in old:
            lines += _listing("added", key, new[key])
        elif key not in new:
            lines += _listing("removed", key, old[key])
        elif old[key] != new[key]:
            lines += _changes(key, old[key], new[key])
    return lines or ["- no changes"]


def _listing(verb: str, key: str, entry: _Entry) -> list[str]:
    """An added or removed item: every value and, for a document, its text."""
    lines = [f"- {verb}: {key}"]
    lines += [f"  - {field}: {_shown(value)}" for field, value in sorted(entry.fields.items())]
    if entry.wording is not None:
        lines += _text_lines(entry.wording)
    return lines


def _text_lines(wording: str) -> list[str]:
    """A document's text under its values; a document without text says so."""
    if not wording:
        return ["  - text: —"]
    return ["  - text:", *_indented(wording.splitlines())]


def _changes(key: str, before: _Entry, after: _Entry) -> list[str]:
    """A changed item: each changed value before and after, then what happened to its text."""
    lines = [f"- changed: {key}"]
    for field in sorted(before.fields.keys() | after.fields.keys()):
        if before.fields.get(field) != after.fields.get(field):
            lines.append(
                f"  - {field}: {_shown(before.fields.get(field))} → {_shown(after.fields.get(field))}"
            )
    return lines + _text_changes(before, after)


def _text_changes(before: _Entry, after: _Entry) -> list[str]:
    if before.wording == after.wording and before.content_hash == after.content_hash:
        return []
    diff = list(
        difflib.unified_diff(
            (before.wording or "").splitlines(),
            (after.wording or "").splitlines(),
            fromfile="active version",
            tofile="this version",
            lineterm="",
        )
    )
    if not diff:
        return ["  - text changed (only formatting: the wording is the same)"]
    return ["  - text changed:", *_indented(diff)]


def _indented(lines: list[str]) -> list[str]:
    return [f"{_TEXT_INDENT}{line}".rstrip() for line in lines]


def _shown(value: str | None) -> str:
    return "—" if value is None else json.dumps(value, ensure_ascii=False)
