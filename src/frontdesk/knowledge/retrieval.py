"""Hybrid retrieval: full-text plus vector search fused with Reciprocal Rank Fusion (spec §7).

Filters come first — business, active version, validity dates — so nothing
unauthorized or expired is ever a candidate.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from frontdesk.db.models import Chunk, Document
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.knowledge.versions import active_version_id
from frontdesk.safety.normalize import tokens

RRF_K = 60
MIN_TOKEN_LENGTH = 3
STOPWORDS = frozenset(
    {
        # sq
        "per",
        "dhe",
        "nje",
        "eshte",
        "jane",
        "keni",
        "kam",
        "nga",
        "nuk",
        "ose",
        "por",
        "cfare",
        "mund",
        "une",
        "juve",
        "kjo",
        "ky",
        "qe",
        # it
        "che",
        "con",
        "non",
        "una",
        "uno",
        "del",
        "della",
        "dei",
        "delle",
        "gli",
        "alla",
        "sono",
        "come",
        "dove",
        "quando",
        "cosa",
        "posso",
        "vorrei",
        "avete",
        # en
        "the",
        "and",
        "for",
        "with",
        "are",
        "you",
        "your",
        "can",
        "what",
        "how",
        "where",
        "when",
        "does",
        "have",
        "there",
        "this",
        "that",
        "from",
        "any",
        "has",
        "is",
    }
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_key: str
    title: str
    heading: str
    content: str
    version_id: uuid.UUID
    public_url: str | None
    score: float


def rrf_fuse(
    rankings: Sequence[Sequence[uuid.UUID]], *, k: int = RRF_K
) -> list[tuple[uuid.UUID, float]]:
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))


def build_tsquery(query: str) -> str | None:
    terms = sorted(
        {term for term in tokens(query) if len(term) >= MIN_TOKEN_LENGTH and term not in STOPWORDS}
    )
    return " | ".join(terms) if terms else None


def hybrid_search(
    session: Session,
    *,
    business_id: str,
    query: str,
    query_vector: Sequence[float] | None,
    today: date,
    limit: int = 5,
    candidates: int = 20,
    max_chars: int = 4000,
) -> list[RetrievedChunk]:
    """Best chunks of the business's active knowledge for a query: full text and vector, fused.

    The first chunk is always returned; later ones only while they fit in `max_chars`. A query
    vector of all zeros (a message without tokens) counts as no vector: only the terms are searched.
    """
    version_id = active_version_id(session, business_id)
    if version_id is None:
        return []
    eligible = _eligible(business_id, version_id, today)
    rankings: list[list[uuid.UUID]] = []
    tsquery = build_tsquery(query)
    if tsquery is not None:
        ts = func.to_tsquery("simple", tsquery)
        rankings.append(
            list(
                session.scalars(
                    eligible.where(Chunk.tsv.op("@@")(ts))
                    .order_by(func.ts_rank_cd(Chunk.tsv, ts).desc(), Chunk.id)
                    .limit(candidates)
                )
            )
        )
    if query_vector is not None and _has_direction(query_vector):
        nearest = _nearest(eligible, list(query_vector), candidates)
        rankings.append(list(session.scalars(nearest)))
    return _load(session, rrf_fuse(rankings)[:limit], max_chars)


def chunk_evidence(chunk: RetrievedChunk) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.DOCUMENT,
        source_ref=f"{chunk.document_key}#{chunk.chunk_id}",
        version=str(chunk.version_id),
        title=chunk.heading,
        content=chunk.content,
        public_url=chunk.public_url,
    )


def _has_direction(vector: Sequence[float]) -> bool:
    """False for the zero vector, which HashEmbedder returns for text without tokens.

    pgvector measures the cosine distance from a zero vector to every chunk as NaN, so the vector
    side would rank the chunks by id: arbitrary evidence. Such a query is searched by terms alone.
    """
    return any(value != 0.0 for value in vector)


def _nearest(eligible: Select[Any], vector: list[float], candidates: int) -> Select[Any]:
    """The eligible chunks nearest to the vector by cosine distance, found exactly.

    The eligible chunks are materialized first, so the sort cannot run on the HNSW index, which
    filters after its scan and returns few or no rows once retired versions hold the same chunks.
    """
    pool = (
        eligible.add_columns(Chunk.embedding)
        .where(Chunk.embedding.is_not(None))
        .cte("eligible_vectors")
        .prefix_with("MATERIALIZED")
    )
    return (
        select(pool.c.id)
        .order_by(pool.c.embedding.cosine_distance(vector), pool.c.id)
        .limit(candidates)
    )


def _eligible(business_id: str, version_id: uuid.UUID, today: date) -> Select[Any]:
    return (
        select(Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.business_id == business_id,
            Chunk.knowledge_version_id == version_id,
            or_(Document.valid_from.is_(None), Document.valid_from <= today),
            or_(Document.valid_until.is_(None), Document.valid_until >= today),
        )
    )


def _load(
    session: Session, fused: list[tuple[uuid.UUID, float]], max_chars: int
) -> list[RetrievedChunk]:
    if not fused:
        return []
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.heading,
            Chunk.content,
            Chunk.knowledge_version_id,
            Document.doc_key,
            Document.title,
            Document.public_url,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_([chunk_id for chunk_id, _ in fused]))
    ).all()
    by_id: dict[uuid.UUID, Any] = {row.id: row for row in rows}
    results: list[RetrievedChunk] = []
    used = 0
    for chunk_id, score in fused:
        row = by_id.get(chunk_id)
        if row is None:
            continue
        if results and used + len(row.content) > max_chars:
            break
        used += len(row.content)
        results.append(
            RetrievedChunk(
                chunk_id=row.id,
                document_key=row.doc_key,
                title=row.title,
                heading=row.heading,
                content=row.content,
                version_id=row.knowledge_version_id,
                public_url=row.public_url,
                score=score,
            )
        )
    return results
