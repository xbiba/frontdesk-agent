"""Retrieval evaluation (spec §14): recall@k and MRR per case, language and tag.

Results with HashEmbedder check contracts only; quality numbers come from runs
with a real embedder, recorded separately.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import fmean

import yaml
from sqlalchemy.orm import Session

from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.retrieval import hybrid_search


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    question: str
    language: str
    relevant: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseResult:
    case: RetrievalCase
    retrieved: tuple[str, ...]
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalReport:
    business_id: str
    k: int
    embedder: str
    results: tuple[CaseResult, ...]

    @property
    def mean_recall(self) -> float:
        return fmean(r.recall for r in self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        return fmean(r.reciprocal_rank for r in self.results) if self.results else 0.0

    def by_language(self) -> dict[str, tuple[float, float]]:
        return _grouped(self.results, lambda result: [result.case.language])

    def by_tag(self) -> dict[str, tuple[float, float]]:
        return _grouped(self.results, lambda result: list(result.case.tags))


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    wanted = set(relevant)
    if not wanted:
        raise ValueError("a case needs at least one relevant document")
    return len(set(retrieved[:k]) & wanted) / len(wanted)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    for rank, key in enumerate(retrieved, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def load_cases(path: Path) -> list[RetrievalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            id=item["id"],
            question=item["question"],
            language=item["language"],
            relevant=tuple(item["relevant"]),
            tags=tuple(item.get("tags", [])),
        )
        for item in raw
    ]


def run_retrieval_eval(
    session: Session,
    *,
    business_id: str,
    cases: Sequence[RetrievalCase],
    embedder: Embedder,
    k: int,
    today: date,
) -> RetrievalReport:
    results: list[CaseResult] = []
    for case in cases:
        vector = embedder.embed([case.question])[0]
        hits = hybrid_search(
            session,
            business_id=business_id,
            query=case.question,
            query_vector=vector,
            today=today,
            limit=k * 3,
            max_chars=1_000_000,
        )
        keys = tuple(dict.fromkeys(hit.document_key for hit in hits))[:k]
        results.append(
            CaseResult(
                case=case,
                retrieved=keys,
                recall=recall_at_k(keys, case.relevant, k),
                reciprocal_rank=reciprocal_rank(keys, case.relevant),
            )
        )
    return RetrievalReport(business_id, k, type(embedder).__name__, tuple(results))


def render_markdown(report: RetrievalReport) -> str:
    lines = [
        f"# Retrieval evaluation — {report.business_id}",
        "",
        f"Embedder: {report.embedder} · k = {report.k} · cases: {len(report.results)}",
        f"Mean recall@{report.k}: {report.mean_recall:.2f} · MRR: {report.mrr:.2f}",
        "",
        f"| case | language | recall@{report.k} | RR |",
        "|---|---|---|---|",
        *[
            f"| {r.case.id} | {r.case.language} | {r.recall:.2f} | {r.reciprocal_rank:.2f} |"
            for r in report.results
        ],
        "",
        "| group | recall | MRR |",
        "|---|---|---|",
        *[
            f"| lang:{name} | {rec:.2f} | {mrr:.2f} |"
            for name, (rec, mrr) in report.by_language().items()
        ],
        *[
            f"| tag:{name} | {rec:.2f} | {mrr:.2f} |"
            for name, (rec, mrr) in report.by_tag().items()
        ],
    ]
    return "\n".join(lines)


def to_json(report: RetrievalReport) -> str:
    return json.dumps(
        {
            "business_id": report.business_id,
            "k": report.k,
            "embedder": report.embedder,
            "mean_recall": report.mean_recall,
            "mrr": report.mrr,
            "results": [asdict(result) for result in report.results],
        },
        indent=2,
        ensure_ascii=False,
    )


def _grouped(
    results: Sequence[CaseResult], keys: Callable[[CaseResult], list[str]]
) -> dict[str, tuple[float, float]]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        for key in keys(result):
            groups[key].append(result)
    return {
        key: (fmean(r.recall for r in items), fmean(r.reciprocal_rank for r in items))
        for key, items in sorted(groups.items())
    }
