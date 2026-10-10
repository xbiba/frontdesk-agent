"""The embedder experiment: each candidate in a table of its own dimension, and the decision rule."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from statistics import fmean

import pytest
from sqlalchemy import text as sql
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.embedder_experiment import (
    Candidate,
    CandidateResult,
    Decision,
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


LANGUAGES = ("sq", "it", "en")
Figures = tuple[float, float, float]  # one figure for each of sq, it and en


def _scores(
    at3: Figures,
    *,
    at1: Figures | None = None,
    at5: Figures | None = None,
    mrr: Figures | None = None,
) -> dict[str, LanguageScores]:
    """Recall@3 as given. Recall@1, recall@5 and the MRR are given too, or derived from it so that
    the four figures of a language all differ: a verdict read from the wrong one then changes."""
    first = at1 or tuple(round(value / 2, 4) for value in at3)
    fifth = at5 or tuple(min(1.0, round(value + 0.2, 4)) for value in at3)
    reciprocal = mrr or tuple(round(value * 0.9, 4) for value in at3)
    return {
        language: LanguageScores(
            cases=10, recall={1: first[i], 3: at3[i], 5: fifth[i]}, mrr=reciprocal[i]
        )
        for i, language in enumerate(LANGUAGES)
    }


def _result(
    name: str,
    sq: float,
    it: float,
    en: float,
    *,
    first: Figures | None = None,
    fifth: Figures | None = None,
    mrr: Figures | None = None,
) -> CandidateResult:
    return CandidateResult(
        name=name,
        identity=name,
        dimension=8,
        languages=_scores((sq, it, en), at1=first, at5=fifth, mrr=mrr),
        index_seconds=0.0,
        index_usd=0.0,
    )


REFERENCE = _result("reference", 0.60, 0.70, 0.80)
ZEROS: Figures = (0.0, 0.0, 0.0)
ONES: Figures = (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("candidate", "chosen"),
    [
        pytest.param(_result("a", 0.70, 0.80, 0.90), "a", id="ten-points-everywhere"),
        pytest.param(_result("a", 0.69, 0.80, 0.90), "reference", id="just-under-ten"),
        pytest.param(_result("a", 0.95, 0.95, 0.75), "reference", id="worse-in-one-language"),
        pytest.param(_result("a", 0.60, 0.70, 0.80), "reference", id="parity"),
        # Only recall@3 counts: a candidate is not held back by a loss at any other figure ...
        pytest.param(
            _result("a", 0.70, 0.80, 0.90, first=ZEROS), "a", id="lost-first-places-but-gains-at-3"
        ),
        pytest.param(
            _result("a", 0.70, 0.80, 0.90, fifth=ZEROS), "a", id="lost-fifth-places-but-gains-at-3"
        ),
        pytest.param(_result("a", 0.70, 0.80, 0.90, mrr=ZEROS), "a", id="lost-mrr-but-gains-at-3"),
        # ... nor carried by a gain at any other figure.
        pytest.param(
            _result("a", 0.65, 0.75, 0.85, first=ONES, fifth=ONES, mrr=ONES),
            "reference",
            id="gains-everywhere-but-at-3",
        ),
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


def test_the_best_of_several_is_read_at_recall_at_3_and_not_at_another_figure() -> None:
    best_at_3 = _result("best-at-3", 0.80, 0.90, 0.95, fifth=(0.5, 0.5, 0.5), mrr=(0.2, 0.2, 0.2))
    best_elsewhere = _result("best-elsewhere", 0.75, 0.85, 0.90, fifth=ONES, mrr=(0.99,) * 3)
    for order in ([best_at_3, best_elsewhere], [best_elsewhere, best_at_3]):
        assert decide(REFERENCE, order).chosen == "best-at-3"


def test_equal_mean_recall_at_3_is_broken_by_the_mean_mrr() -> None:
    def with_mrr(name: str, mrr: float) -> CandidateResult:
        scores = {
            language: LanguageScores(cases=10, recall={1: 0.8, 3: 0.8, 5: 0.9}, mrr=mrr)
            for language in ("sq", "it", "en")
        }
        return CandidateResult(name, name, 8, scores, index_seconds=0.0, index_usd=0.0)

    assert decide(REFERENCE, [with_mrr("slow", 0.70), with_mrr("quick", 0.90)]).chosen == "quick"


def test_equal_in_every_figure_the_first_in_the_order_given_wins() -> None:
    first, second = _result("first", 0.80, 0.90, 0.95), _result("second", 0.80, 0.90, 0.95)
    assert decide(REFERENCE, [first, second]).chosen == "first"
    assert decide(REFERENCE, [second, first]).chosen == "second"


def test_float_noise_does_not_make_a_language_worse() -> None:
    # Two means that are the same number in arithmetic (two fifths) and differ by a bit in floats.
    before = fmean([3 / 5, 1 / 4, 1 / 2, 1 / 4])
    after = fmean([3 / 5, 1 / 3, 0.0, 2 / 3])
    assert after < before  # as the machine adds them ...
    assert before - after < 1e-15  # ... and yet the same number

    def result(name: str, sq: float, other: float) -> CandidateResult:
        return CandidateResult(
            name, name, 8, _scores((sq, other, other)), index_seconds=0.0, index_usd=0.0
        )

    decision = decide(result("reference", before, 0.2), [result("candidate", after, 0.8)])
    assert decision.chosen == "candidate"
    assert decision.reasons == (
        "candidate: mean recall@3 gains 40.0 points",
        "candidate replaces reference",
    )


def test_a_language_that_is_worse_by_more_than_noise_still_holds_a_candidate_back() -> None:
    reference = CandidateResult("reference", "r", 8, _scores((0.4, 0.2, 0.2)), 0.0, 0.0)
    candidate = CandidateResult("candidate", "c", 8, _scores((0.4 - 1e-6, 0.8, 0.8)), 0.0, 0.0)
    decision = decide(reference, [candidate])
    assert decision.chosen == "reference"
    assert decision.reasons[0] == "candidate: recall@3 is worse in sq"


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
        languages=_scores((0.5, 0.6, 0.7)),
        index_seconds=1.5,
        index_usd=0.01,
        query_seconds=(0.1, 0.2, 0.3),
        query_usd=0.02,
    )
    assert result_from_json(result_json(saved)) == saved


def test_the_report_lists_every_embedder_and_the_decision() -> None:
    candidate = _result("candidate", 0.70, 0.80, 0.90)
    report = render_experiment(
        [REFERENCE, candidate], decide(REFERENCE, [candidate]), usd_per_eur=1.25
    )
    lines = report.splitlines()
    header, separator = (line for line in lines if line.startswith(("| embedder", "|---")))
    assert header.count("|") == separator.count("|")  # one separator cell for each column
    assert header.index("en R@1") < header.index("it R@1") < header.index("sq R@1")
    assert header.endswith("| mean R@3 | index s | index EUR | query s (mean) | queries EUR |")
    assert sum(line.startswith(("| reference |", "| candidate |")) for line in lines) == 2
    assert "**Decision: candidate**" in lines
    assert "- candidate: mean recall@3 gains 10.0 points" in lines


def test_a_row_of_the_report_reads_each_figure_from_its_own_column() -> None:
    languages = {
        "sq": LanguageScores(cases=10, recall={1: 0.11, 3: 0.33, 5: 0.55}, mrr=0.77),
        "it": LanguageScores(cases=10, recall={1: 0.12, 3: 0.34, 5: 0.56}, mrr=0.78),
        "en": LanguageScores(cases=10, recall={1: 0.13, 3: 0.35, 5: 0.57}, mrr=0.79),
    }
    result = CandidateResult(
        "vertex-x",
        "vertex:x",
        768,
        languages,
        index_seconds=12.34,
        index_usd=0.5,
        query_seconds=(0.2, 0.4),
        query_usd=0.125,
    )
    report = render_experiment([result], Decision("vertex-x", ()), usd_per_eur=1.25)
    row = (
        "| vertex-x | 768 "
        "| 0.13 | 0.35 | 0.57 | 0.79 "  # en: R@1, R@3, R@5, MRR (languages in alphabetical order)
        "| 0.12 | 0.34 | 0.56 | 0.78 "  # it
        "| 0.11 | 0.33 | 0.55 | 0.77 "  # sq
        "| 0.34 "  # mean R@3 over the languages
        "| 12.3 | 0.4000 "  # indexing: seconds, euros (dollars over the rate)
        "| 0.30 | 0.1000 |"  # questions: mean seconds, euros
    )
    assert row in report.splitlines()


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
