"""The embedder experiment: each candidate in a table of its own dimension, and the decision rule."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text as sql
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.embedder_experiment import (
    Candidate,
    CandidateResult,
    LanguageScores,
    decide,
    render_experiment,
    result_from_json,
    result_json,
    run_candidate,
    table_name,
)
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.evaluation import load_cases
from frontdesk.tenancy.pack import BusinessPack

TODAY = date(2026, 10, 7)
Activate = Callable[..., sessionmaker[Session]]


class SmallHashEmbedder:
    """Bag-of-tokens vectors of any size: a candidate whose dimension is not the schema's."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.identity = f"small-hash-{dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def _scores(sq: float, it: float, en: float) -> dict[str, LanguageScores]:
    return {
        language: LanguageScores(cases=10, recall={1: value, 3: value, 5: value}, mrr=value)
        for language, value in (("sq", sq), ("it", it), ("en", en))
    }


def _result(name: str, sq: float, it: float, en: float) -> CandidateResult:
    return CandidateResult(
        name=name,
        identity=name,
        dimension=8,
        languages=_scores(sq, it, en),
        index_seconds=0.0,
        index_usd=0.0,
    )


REFERENCE = _result("reference", 0.60, 0.70, 0.80)


@pytest.mark.parametrize(
    ("candidate", "chosen"),
    [
        pytest.param(_result("a", 0.70, 0.80, 0.90), "a", id="ten-points-everywhere"),
        pytest.param(_result("a", 0.69, 0.80, 0.90), "reference", id="just-under-ten"),
        pytest.param(_result("a", 0.95, 0.95, 0.75), "reference", id="worse-in-one-language"),
        pytest.param(_result("a", 0.60, 0.70, 0.80), "reference", id="parity"),
    ],
)
def test_a_candidate_must_gain_ten_points_and_lose_no_language(
    candidate: CandidateResult, chosen: str
) -> None:
    assert decide(REFERENCE, [candidate]).chosen == chosen


def test_among_candidates_that_qualify_the_best_mean_recall_at_3_wins() -> None:
    better = _result("better", 0.80, 0.90, 0.95)
    good = _result("good", 0.75, 0.85, 0.90)
    decision = decide(REFERENCE, [good, better])
    assert decision.chosen == "better"
    assert decision.reasons[-1] == "better replaces reference"


def test_equal_mean_recall_at_3_is_broken_by_the_mean_mrr() -> None:
    def with_mrr(name: str, mrr: float) -> CandidateResult:
        scores = {
            language: LanguageScores(cases=10, recall={1: 0.8, 3: 0.8, 5: 0.9}, mrr=mrr)
            for language in ("sq", "it", "en")
        }
        return CandidateResult(name, name, 8, scores, index_seconds=0.0, index_usd=0.0)

    assert decide(REFERENCE, [with_mrr("slow", 0.70), with_mrr("quick", 0.90)]).chosen == "quick"


def test_the_reference_stays_and_the_reasons_say_why() -> None:
    decision = decide(REFERENCE, [_result("a", 0.95, 0.95, 0.75), _result("b", 0.65, 0.75, 0.85)])
    assert decision.chosen == "reference"
    assert decision.reasons == (
        "a: recall@3 is worse in en",
        "b: mean recall@3 gains 5.0 points, under 10",
        "reference stays",
    )


def test_each_candidate_gets_a_table_name_it_can_use() -> None:
    assert table_name("vertex-ew1-gemini-embedding-001") == (
        "embedding_experiment_vertex_ew1_gemini_embedding_001"
    )


def test_a_name_too_long_for_a_table_is_refused() -> None:
    with pytest.raises(ValueError, match="no table name for candidate"):
        table_name("x" * 60)


def test_a_result_survives_the_file_it_is_saved_in() -> None:
    saved = CandidateResult(
        name="vertex-ew1-gemini-embedding-001",
        identity="vertex:gemini-embedding-001",
        dimension=3072,
        languages=_scores(0.5, 0.6, 0.7),
        index_seconds=1.5,
        index_usd=0.01,
        query_seconds=(0.1, 0.2, 0.3),
        query_usd=0.02,
    )
    assert result_from_json(result_json(saved)) == saved


def test_the_report_lists_every_embedder_with_its_figures_and_the_decision() -> None:
    candidate = _result("candidate", 0.70, 0.80, 0.90)
    report = render_experiment(
        [REFERENCE, candidate], decide(REFERENCE, [candidate]), usd_per_eur=1.25
    )
    lines = report.splitlines()
    header, separator = (line for line in lines if line.startswith(("| embedder", "|---")))
    assert header.count("|") == separator.count("|")  # one separator cell for each column
    assert header.index("en R@1") < header.index("it R@1") < header.index("sq R@1")
    assert "| reference | 8 | 0.80 | 0.80 | 0.80 | 0.80 | 0.70 |" in report  # English first
    assert "| candidate | 8 | 0.90 | 0.90 | 0.90 | 0.90 | 0.80 |" in report
    assert "**Decision: candidate**" in lines
    assert "- candidate: mean recall@3 gains 10.0 points" in lines


@pytest.mark.db
def test_each_candidate_is_indexed_and_searched_with_its_own_dimension(
    activate_pack: Activate,
    database: tuple[str, str],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
) -> None:
    activate_pack(clinic_pack)
    # The experiment creates tables: it runs as the owner of the scratch database.
    factory = make_session_factory(make_engine(database[0]))
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    spent = iter([0.0, 0.25, 0.25, 0.5])
    small = SmallHashEmbedder(16)
    with business_session(factory, clinic_pack.id) as session:
        result = run_candidate(
            session,
            business_id=clinic_pack.id,
            cases=cases,
            candidate=Candidate("small", 16, small, small),
            today=TODAY,
            spent_usd=lambda: next(spent),
        )
        reference = run_candidate(
            session,
            business_id=clinic_pack.id,
            cases=cases,
            candidate=Candidate("reference", 384, HashEmbedder(), HashEmbedder()),
            today=TODAY,
            spent_usd=lambda: 0.0,
        )
    assert result.dimension == 16
    assert set(result.languages) == {"sq", "it", "en"}
    assert sum(scores.cases for scores in result.languages.values()) == len(cases)
    assert all(0.0 <= scores.recall[3] <= 1.0 for scores in result.languages.values())
    assert (result.index_usd, result.query_usd) == (0.25, 0.25)
    assert len(result.query_seconds) == len(cases)
    assert reference.mean_recall(5) > 0


@pytest.mark.db
def test_a_candidate_run_again_replaces_its_vectors_and_never_touches_the_chunks(
    activate_pack: Activate,
    database: tuple[str, str],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
) -> None:
    activate_pack(clinic_pack)
    factory = make_session_factory(make_engine(database[0]))
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    small = SmallHashEmbedder(16)
    table = table_name("small")

    def counts(session: Session) -> tuple[int, int, int]:
        return (
            session.scalar(sql(f"SELECT count(*) FROM {table}")) or 0,
            session.scalar(sql("SELECT count(*) FROM chunks")) or 0,
            session.scalar(sql("SELECT count(*) FROM chunks WHERE embedding IS NULL")) or 0,
        )

    with business_session(factory, clinic_pack.id) as session:
        for _ in range(2):  # a run that stopped and started again
            run_candidate(
                session,
                business_id=clinic_pack.id,
                cases=cases,
                candidate=Candidate("small", 16, small, small),
                today=TODAY,
                spent_usd=lambda: 0.0,
            )
        vectors, chunks, unembedded = counts(session)
    assert vectors == chunks > 0  # one vector for each chunk of the active version, not two
    assert unembedded == 0  # the production vectors are as the pack's activation left them


@pytest.mark.db
def test_a_business_without_active_knowledge_has_nothing_to_index(
    database: tuple[str, str], businesses_dir: Path, clinic_pack: BusinessPack
) -> None:
    factory = make_session_factory(make_engine(database[0]))
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    small = SmallHashEmbedder(16)
    with (
        business_session(factory, clinic_pack.id) as session,
        pytest.raises(ValueError, match="has no active knowledge version"),
    ):
        run_candidate(
            session,
            business_id=clinic_pack.id,
            cases=cases,
            candidate=Candidate("small", 16, small, small),
            today=TODAY,
            spent_usd=lambda: 0.0,
        )
