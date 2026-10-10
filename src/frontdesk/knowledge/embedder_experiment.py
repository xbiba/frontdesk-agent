"""The embedder experiment (M3 design §4.5, ADR-0005).

For the reference and each candidate: recall at 1, 3 and 5 and MRR per language, and the time and
cost of indexing and of search. Each embedder's vectors go into a table of their own dimension in
a scratch database, beside the pack's active knowledge; retrieval is the production hybrid search
with its vector side pointed at that table, so the embedder is the only thing that differs.

The decision rule, fixed before the run: a candidate replaces the reference only if it improves
the mean recall@3 over the languages by at least 10 points and is worse in no language; at parity
the reference stays. Among several that qualify, the highest mean recall@3 wins, then the highest
mean MRR, then the first in the order given.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from statistics import fmean
from typing import Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, MetaData, Table, delete, insert, select, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session

from frontdesk.db.models import Chunk
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.evaluation import RetrievalCase, recall_at_k, reciprocal_rank
from frontdesk.knowledge.retrieval import NearestQuery, PlacedSelect, hybrid_search
from frontdesk.knowledge.versions import active_version_id

KS: Final = (1, 3, 5)
RECALL_GAIN: Final = 0.10  # ten points of mean recall@3
# A recall is a mean of fractions, and means that are the same number in arithmetic can differ in
# the last bits of a float: a difference under this is none, for the gain and for "no language worse".
FLOAT_NOISE: Final = 1e-9
_TABLE: Final = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class Candidate:
    name: str  # "reference" or the route id
    dimension: int
    documents: Embedder  # embeds the passages
    queries: Embedder  # embeds the questions


@dataclass(frozen=True)
class LanguageScores:
    cases: int
    recall: Mapping[int, float]  # k -> mean recall@k
    mrr: float


@dataclass(frozen=True)
class CandidateResult:
    name: str
    identity: str
    dimension: int
    languages: Mapping[str, LanguageScores]
    index_seconds: float
    index_usd: float
    query_seconds: tuple[float, ...] = field(default_factory=tuple)
    query_usd: float = 0.0

    def mean_recall(self, k: int) -> float:
        """The mean over the languages, so that each language weighs the same."""
        return fmean(scores.recall[k] for scores in self.languages.values())

    @property
    def mean_mrr(self) -> float:
        return fmean(scores.mrr for scores in self.languages.values())


@dataclass(frozen=True)
class Decision:
    chosen: str
    reasons: tuple[str, ...]


def table_name(candidate: str) -> str:
    name = "embedding_experiment_" + re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_")
    if not _TABLE.fullmatch(name):
        raise ValueError(f"no table name for candidate '{candidate}'")
    return name


def vector_table(name: str, dimension: int) -> Table:
    return Table(
        name,
        MetaData(),
        Column("chunk_id", UUID(as_uuid=True), primary_key=True),
        Column("embedding", Vector(dimension), nullable=False),
    )


def side_table_nearest(table: Table) -> NearestQuery:
    """The vector side of the hybrid search, read from `table` instead of the chunks."""

    def nearest(eligible: PlacedSelect, vector: list[float], candidates: int) -> PlacedSelect:
        pool = (
            eligible.add_columns(table.c.embedding)
            .join(table, table.c.chunk_id == Chunk.id)
            .cte("experiment_vectors")
            .prefix_with("MATERIALIZED")
        )
        return (
            select(pool.c.id, pool.c.doc_key, pool.c.ordinal)
            .order_by(
                pool.c.embedding.cosine_distance(vector),
                pool.c.doc_key.collate("C"),
                pool.c.ordinal,
            )
            .limit(candidates)
        )

    return nearest


def run_candidate(
    session: Session,
    *,
    business_id: str,
    cases: Sequence[RetrievalCase],
    candidate: Candidate,
    today: date,
    spent_usd: Callable[[], float],
    clock: Callable[[], float] = time.perf_counter,
) -> CandidateResult:
    """Index the active knowledge with the candidate, then answer every case with it."""
    version_id = active_version_id(session, business_id)
    if version_id is None:
        raise ValueError(f"business '{business_id}' has no active knowledge version")
    table = _fresh_table(session, candidate)
    chunks = session.execute(
        select(Chunk.id, Chunk.heading, Chunk.content).where(
            Chunk.knowledge_version_id == version_id
        )
    ).all()
    before, started = spent_usd(), clock()
    vectors = candidate.documents.embed([f"{row.heading}\n{row.content}" for row in chunks])
    index_seconds, index_usd = clock() - started, spent_usd() - before
    session.execute(
        insert(table),
        [
            {"chunk_id": row.id, "embedding": vector}
            for row, vector in zip(chunks, vectors, strict=True)
        ],
    )
    nearest = side_table_nearest(table)
    before = spent_usd()
    timings: list[float] = []
    found: dict[str, list[tuple[RetrievalCase, tuple[str, ...]]]] = defaultdict(list)
    for case in cases:
        started = clock()
        vector = candidate.queries.embed([case.question])[0]
        timings.append(clock() - started)
        hits = hybrid_search(
            session,
            business_id=business_id,
            query=case.question,
            query_vector=vector,
            today=today,
            limit=max(KS) * 3,
            max_chars=1_000_000,
            nearest=nearest,
        )
        keys = tuple(dict.fromkeys(hit.document_key for hit in hits))[: max(KS)]
        found[case.language].append((case, keys))
    return CandidateResult(
        name=candidate.name,
        identity=candidate.queries.identity,
        dimension=candidate.dimension,
        languages={language: _scores(results) for language, results in sorted(found.items())},
        index_seconds=index_seconds,
        index_usd=index_usd,
        query_seconds=tuple(timings),
        query_usd=spent_usd() - before,
    )


def decide(reference: CandidateResult, candidates: Sequence[CandidateResult]) -> Decision:
    reasons: list[str] = []
    qualified: list[CandidateResult] = []
    base = reference.mean_recall(3)
    for result in candidates:
        gain = result.mean_recall(3) - base
        worse = [
            language
            for language, scores in reference.languages.items()
            if result.languages[language].recall[3] < scores.recall[3] - FLOAT_NOISE
        ]
        if worse:
            reasons.append(f"{result.name}: recall@3 is worse in {', '.join(worse)}")
        elif gain < RECALL_GAIN - FLOAT_NOISE:
            reasons.append(f"{result.name}: mean recall@3 gains {gain * 100:.1f} points, under 10")
        else:
            reasons.append(f"{result.name}: mean recall@3 gains {gain * 100:.1f} points")
            qualified.append(result)
    if not qualified:
        return Decision(reference.name, (*reasons, f"{reference.name} stays"))
    best = max(qualified, key=lambda result: (result.mean_recall(3), result.mean_mrr))
    return Decision(best.name, (*reasons, f"{best.name} replaces {reference.name}"))


def _fresh_table(session: Session, candidate: Candidate) -> Table:
    name = table_name(candidate.name)
    session.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {name} (chunk_id uuid PRIMARY KEY REFERENCES chunks(id) "
            f"ON DELETE CASCADE, embedding vector({int(candidate.dimension)}) NOT NULL)"
        )
    )
    table = vector_table(name, candidate.dimension)
    session.execute(delete(table))
    return table


def _scores(results: Sequence[tuple[RetrievalCase, tuple[str, ...]]]) -> LanguageScores:
    return LanguageScores(
        cases=len(results),
        recall={
            k: fmean(recall_at_k(keys, case.relevant, k) for case, keys in results) for k in KS
        },
        mrr=fmean(reciprocal_rank(keys, case.relevant) for case, keys in results),
    )


def result_json(result: CandidateResult) -> str:
    return json.dumps(
        {
            "name": result.name,
            "identity": result.identity,
            "dimension": result.dimension,
            "languages": {
                language: {
                    "cases": scores.cases,
                    "recall": {str(k): value for k, value in scores.recall.items()},
                    "mrr": scores.mrr,
                }
                for language, scores in result.languages.items()
            },
            "index_seconds": result.index_seconds,
            "index_usd": result.index_usd,
            "query_seconds": list(result.query_seconds),
            "query_usd": result.query_usd,
        },
        indent=2,
    )


def result_from_json(raw: str) -> CandidateResult:
    data = json.loads(raw)
    return CandidateResult(
        name=data["name"],
        identity=data["identity"],
        dimension=data["dimension"],
        languages={
            language: LanguageScores(
                cases=scores["cases"],
                recall={int(k): value for k, value in scores["recall"].items()},
                mrr=scores["mrr"],
            )
            for language, scores in data["languages"].items()
        },
        index_seconds=data["index_seconds"],
        index_usd=data["index_usd"],
        query_seconds=tuple(data["query_seconds"]),
        query_usd=data["query_usd"],
    )


def render_experiment(
    results: Sequence[CandidateResult], decision: Decision, *, usd_per_eur: float
) -> str:
    languages = sorted({language for result in results for language in result.languages})
    header = " | ".join(
        f"{language} R@1 | {language} R@3 | {language} R@5 | {language} MRR"
        for language in languages
    )
    rows = []
    for result in results:
        cells = " | ".join(
            f"{s.recall[1]:.2f} | {s.recall[3]:.2f} | {s.recall[5]:.2f} | {s.mrr:.2f}"
            for s in (result.languages[language] for language in languages)
        )
        query = fmean(result.query_seconds) if result.query_seconds else 0.0
        rows.append(
            f"| {result.name} | {result.dimension} | {cells} | {result.mean_recall(3):.2f} | "
            f"{result.index_seconds:.1f} | {result.index_usd / usd_per_eur:.4f} | "
            f"{query:.2f} | {result.query_usd / usd_per_eur:.4f} |"
        )
    columns = 4 * len(languages) + 7
    return "\n".join(
        [
            "# Embedder experiment (M3 design §4.5)",
            "",
            f"| embedder | dimension | {header} | mean R@3 | index s | index EUR | "
            "query s (mean) | queries EUR |",
            "|" + "---|" * columns,
            *rows,
            "",
            f"**Decision: {decision.chosen}**",
            "",
            *[f"- {reason}" for reason in decision.reasons],
            "",
        ]
    )
