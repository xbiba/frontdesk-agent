"""Retrieval evaluation (ADR-0005): recall@k and MRR per case, language and tag.

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
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails
from sqlalchemy import select
from sqlalchemy.orm import Session

from frontdesk.db.models import Document
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.retrieval import hybrid_search
from frontdesk.knowledge.versions import active_version_id
from frontdesk.tenancy.pack import Language, NonBlankStr


class DatasetError(ValueError):
    """A retrieval dataset cannot be read, or some of its cases are invalid."""


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
    embedder: str  # the identity of the embedder, e.g. "hash-v1" or "fastembed:<model>"
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


class _CaseSpec(BaseModel):
    """A case as the dataset file writes it. A field that is not listed here is a typo: an error."""

    model_config = ConfigDict(extra="forbid")

    id: NonBlankStr
    question: NonBlankStr
    language: Language
    relevant: list[NonBlankStr] = Field(min_length=1)
    tags: list[NonBlankStr] = Field(default_factory=list)


def load_cases(path: Path) -> list[RetrievalCase]:
    """The cases of a dataset file; every problem in it is reported at once, naming file and case."""
    cases: list[RetrievalCase] = []
    problems: list[str] = []
    positions: dict[str, int] = {}
    for position, item in enumerate(_read_items(path), start=1):
        if not isinstance(item, dict):
            kind = type(item).__name__
            problems.append(f"{path}: case #{position}: expected a mapping, not a {kind}")
            continue
        try:
            spec = _CaseSpec.model_validate(item)
        except ValidationError as error:
            problems.append(f"{path}: {_label(position, item)}: {_describe(error)}")
            continue
        if spec.id in positions:
            problems.append(
                f"{path}: case '{spec.id}': duplicate id, case #{positions[spec.id]} has it too"
            )
            continue
        positions[spec.id] = position
        cases.append(
            RetrievalCase(
                id=spec.id,
                question=spec.question,
                language=spec.language,
                relevant=tuple(spec.relevant),
                tags=tuple(spec.tags),
            )
        )
    if problems:
        raise DatasetError("\n".join(problems))
    return cases


def unknown_relevant_documents(
    session: Session, *, business_id: str, cases: Sequence[RetrievalCase]
) -> list[tuple[str, str]]:
    """The (case id, key) pairs whose `relevant` key is no document of the active version.

    A case that names a document the knowledge does not hold can never be found, so it would score
    0 whatever the retrieval does: a mistyped key is not a retrieval failure. A document that
    exists but is not valid today is known; it scores 0 because it is not searched.
    """
    version_id = active_version_id(session, business_id)
    known = (
        set(
            session.scalars(
                select(Document.doc_key).where(Document.knowledge_version_id == version_id)
            )
        )
        if version_id is not None
        else set()
    )
    return [(case.id, key) for case in cases for key in case.relevant if key not in known]


def run_retrieval_eval(
    session: Session,
    *,
    business_id: str,
    cases: Sequence[RetrievalCase],
    embedder: Embedder,
    k: int,
    today: date,
    min_similarity: float = 0.0,
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
            min_similarity=min_similarity,
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
    return RetrievalReport(business_id, k, embedder.identity, tuple(results))


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


def _read_items(path: Path) -> list[Any]:
    """The entries of a dataset file; a file that is not a non-empty list of them is an error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetError(f"{path}: cannot be read ({error.strerror or error})") from error
    except UnicodeDecodeError as error:
        raise DatasetError(f"{path}: not valid UTF-8 text (save the file as UTF-8)") from error
    except yaml.YAMLError as error:
        raise DatasetError(f"{path}: invalid YAML{_where(error)}") from error
    if not raw:
        raise DatasetError(f"{path}: no cases")
    if not isinstance(raw, list):
        raise DatasetError(f"{path}: expected a list of cases, one '- id: ...' entry for each")
    return raw


def _where(error: yaml.YAMLError) -> str:
    """` at line L, column C: reason` for an error PyYAML placed in the text; else `: reason`."""
    if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
        mark = error.problem_mark
        return f" at line {mark.line + 1}, column {mark.column + 1}: {error.problem}"
    return ": " + " ".join(str(error).split())


def _label(position: int, item: object) -> str:
    """How an error names a case: by its id when it has a usable one, else by its position."""
    case_id = item.get("id") if isinstance(item, dict) else None
    if isinstance(case_id, str) and case_id.strip():
        return f"case '{case_id}'"
    return f"case #{position}"


def _describe(error: ValidationError) -> str:
    """The problems of one case, each with its field: `relevant: Field required`."""
    return "; ".join(
        _problem(problem) for problem in error.errors(include_url=False, include_input=False)
    )


def _problem(problem: ErrorDetails) -> str:
    field = ".".join(str(part) for part in problem["loc"])
    return f"{field}: {problem['msg']}" if field else problem["msg"]
