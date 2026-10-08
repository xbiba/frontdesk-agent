"""Versioned knowledge with approval (spec §7; blueprint §11).

draft -> approved -> active -> retired. Only an approved version can become
active, and embeddings are computed at activation, after approval. Every
transition is one guarded UPDATE, so a concurrent change cannot slip in between.
Each version records which embedder computed its vectors: vectors from two
models are never compared (ADR-0005).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from frontdesk.db.models import EMBEDDING_DIM, Chunk, Document, KnowledgeVersion, Record
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import Embedder
from frontdesk.safety.normalize import normalize
from frontdesk.tenancy.pack import BusinessPack

_KNOWLEDGE_FIELDS = {"location", "services", "doctors", "prices", "orientation", "documents"}
_TEXT = "text"  # the body hash of a document: reported as "text changed", never shown


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
    rows += [("doctor", d.id, d.model_dump(mode="json")) for d in pack.doctors]
    rows += [("price", p.service_id, p.model_dump(mode="json")) for p in pack.prices]
    rows += [("orientation", o.id, o.model_dump(mode="json")) for o in pack.orientation]
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


def _documents(session: Session, version_id: uuid.UUID) -> dict[str, dict[str, str]]:
    rows = session.scalars(select(Document).where(Document.knowledge_version_id == version_id))
    return {
        row.doc_key: _flatten(
            {
                "title": row.title,
                "language": row.language,
                "public_url": row.public_url,
                "valid_from": row.valid_from,
                "valid_until": row.valid_until,
                _TEXT: row.content_hash,
            }
        )
        for row in rows
    }


def _records(session: Session, version_id: uuid.UUID) -> dict[str, dict[str, str]]:
    rows = session.execute(
        select(Record.kind, Record.record_key, Record.payload).where(
            Record.knowledge_version_id == version_id
        )
    ).all()
    return {f"{row.kind}:{row.record_key}": _flatten(row.payload) for row in rows}


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


def _diff_lines(old: dict[str, dict[str, str]], new: dict[str, dict[str, str]]) -> list[str]:
    """What the owner approves: every changed field with its value before and after."""
    lines: list[str] = []
    for key in sorted(old.keys() | new.keys()):
        if key not in old:
            lines.append(f"- added: {key}")
            lines += [
                f"  - {field}: {_shown(value)}"
                for field, value in sorted(new[key].items())
                if field != _TEXT
            ]
        elif key not in new:
            lines.append(f"- removed: {key}")
        elif old[key] != new[key]:
            before, after = old[key], new[key]
            lines.append(f"- changed: {key}")
            for field in sorted(before.keys() | after.keys()):
                if before.get(field) == after.get(field):
                    continue
                if field == _TEXT:
                    lines.append("  - text changed")
                else:
                    lines.append(
                        f"  - {field}: {_shown(before.get(field))} → {_shown(after.get(field))}"
                    )
    return lines or ["- no changes"]


def _shown(value: str | None) -> str:
    return "—" if value is None else json.dumps(value, ensure_ascii=False)
