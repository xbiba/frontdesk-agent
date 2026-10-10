"""The figures the embedder decision rests on, against comparisons whose answers are known by hand
(M3 Task 9, fix round 1).

Nine questions, three for each language, over the five documents of the demo clinic. An embedder
here is a script: its vector side puts the documents in the order the script says, and the
questions are made of words that no document holds, so that full text finds nothing and that order
is all that ranks. The figures below are worked out from those orders, not read from a run, so a
change in how recall, the reciprocal rank, the list of documents or the costs are computed shows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import Chunk, Document
from frontdesk.knowledge.embedder_experiment import (
    Candidate,
    CandidateResult,
    decide,
    render_experiment,
    result_from_json,
    result_json,
    run_candidate,
)
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.evaluation import RetrievalCase, load_cases
from frontdesk.knowledge.versions import active_version_id
from frontdesk.tenancy.pack import BusinessPack
from tests.knowledge.test_embedder_experiment import SmallHashEmbedder

pytestmark = pytest.mark.db

TODAY = date(2026, 10, 10)
DOCUMENTS = ("first-visit-it", "parking-en", "payments-sq", "physiotherapy-it", "physiotherapy-sq")
Ranks = Mapping[str, int]  # document -> its place in the order of the embedder, 1 being nearest
Plan = Mapping[str, Ranks]  # case id -> the places of the case's relevant documents
Activate = Callable[..., sessionmaker[Session]]

CASES = (
    RetrievalCase("sq1", "zzqa1 zzqb1", "sq", ("payments-sq",)),
    RetrievalCase("sq2", "zzqa2 zzqb2", "sq", ("physiotherapy-sq",)),
    RetrievalCase("sq3", "zzqa3 zzqb3", "sq", ("payments-sq",)),
    RetrievalCase("it1", "zzqa4 zzqb4", "it", ("first-visit-it",)),
    RetrievalCase("it2", "zzqa5 zzqb5", "it", ("physiotherapy-it",)),
    RetrievalCase("it3", "zzqa6 zzqb6", "it", ("first-visit-it",)),
    RetrievalCase("en1", "zzqa7 zzqb7", "en", ("parking-en",)),
    RetrievalCase("en2", "zzqa8 zzqb8", "en", ("parking-en",)),
    # Two relevant documents: recall is a half when one of them is found.
    RetrievalCase("en3", "zzqa9 zzqb9", "en", ("physiotherapy-it", "physiotherapy-sq")),
)

REFERENCE: Plan = {
    "sq1": {"payments-sq": 1},
    "sq2": {"physiotherapy-sq": 2},
    "sq3": {"payments-sq": 4},
    "it1": {"first-visit-it": 1},
    "it2": {"physiotherapy-it": 3},
    "it3": {"first-visit-it": 5},
    "en1": {"parking-en": 1},
    "en2": {"parking-en": 2},
    "en3": {"physiotherapy-it": 1, "physiotherapy-sq": 4},
}


def _moved(**changes: Ranks) -> Plan:
    """The reference, with some of its cases ranked otherwise."""
    return {**REFERENCE, **changes}


PLANS: dict[str, Plan] = {
    "reference": REFERENCE,
    # Every relevant document within the first three places: recall@3 is 1 in each language.
    "better": _moved(
        sq2={"physiotherapy-sq": 1},
        sq3={"payments-sq": 2},
        it2={"physiotherapy-it": 2},
        it3={"first-visit-it": 3},
        en2={"parking-en": 1},
        en3={"physiotherapy-it": 1, "physiotherapy-sq": 2},
    ),
    # Perfect in Albanian and Italian, but English loses `en2` (place 2 -> 4): +11.1 points on
    # average and a language that is worse.
    "english-lost-a-question": _moved(
        sq2={"physiotherapy-sq": 1},
        sq3={"payments-sq": 1},
        it2={"physiotherapy-it": 1},
        it3={"first-visit-it": 1},
        en2={"parking-en": 4},
    ),
    # Only `en3` improves (its second document 4 -> 2): English +16.7 points, +5.6 on average.
    "english-gains-a-little": _moved(en3={"physiotherapy-it": 1, "physiotherapy-sq": 2}),
    # Only `sq3` improves (4 -> 3): Albanian +33.3 points, +11.1 on average, nobody worse.
    "albanian-gains-a-lot": _moved(sq3={"payments-sq": 3}),
    "parity": REFERENCE,
    # Everything is found within the first three places, but none at the first: recall@3 is 1 in
    # each language, recall@1 and the reciprocal rank are below the reference's everywhere.
    "first-places-lost": _moved(
        sq1={"payments-sq": 3},
        sq3={"payments-sq": 3},
        it1={"first-visit-it": 3},
        it2={"physiotherapy-it": 3},
        it3={"first-visit-it": 3},
        en1={"parking-en": 3},
        en2={"parking-en": 3},
        en3={"physiotherapy-it": 2, "physiotherapy-sq": 3},
    ),
}

# (recall@1, recall@3, recall@5, MRR) of each language, worked out from the places above:
#   sq  places 1, 2, 4   recall@1 1/0/0, @3 1/1/0, @5 1/1/1   reciprocal ranks 1, 1/2, 1/4
#   it  places 1, 3, 5   recall@1 1/0/0, @3 1/1/0, @5 1/1/1   reciprocal ranks 1, 1/3, 1/5
#   en  places 1, 2, and 1 and 4 for two documents:
#       recall@1 1/0/(1/2), @3 1/1/(1/2), @5 1/1/1           reciprocal ranks 1, 1/2, 1
REFERENCE_FIGURES = {
    "sq": (1 / 3, 2 / 3, 1.0, (1 + 1 / 2 + 1 / 4) / 3),
    "it": (1 / 3, 2 / 3, 1.0, (1 + 1 / 3 + 1 / 5) / 3),
    "en": (1 / 2, 5 / 6, 1.0, (1 + 1 / 2 + 1) / 3),
}
# The same for "first-places-lost" (places 3, 2, 3 / 3, 3, 3 / 3, 3, and 2 and 3 for two documents).
FIRST_PLACES_LOST_FIGURES = {
    "sq": (0.0, 1.0, 1.0, (1 / 3 + 1 / 2 + 1 / 3) / 3),
    "it": (0.0, 1.0, 1.0, 1 / 3),
    "en": (0.0, 1.0, 1.0, (1 / 3 + 1 / 3 + 1 / 2) / 3),
}
# recall@3 of each language, for every embedder.
RECALL_AT_3 = {
    "better": {"sq": 1.0, "it": 1.0, "en": 1.0},
    "english-lost-a-question": {"sq": 1.0, "it": 1.0, "en": 1 / 2},
    "english-gains-a-little": {"sq": 2 / 3, "it": 2 / 3, "en": 1.0},
    "albanian-gains-a-lot": {"sq": 1.0, "it": 2 / 3, "en": 5 / 6},
    "parity": {"sq": 2 / 3, "it": 2 / 3, "en": 5 / 6},
}


# Which of the candidates the rule chooses over the reference, given in this order, and why.
VERDICTS = [
    (["better"], "better", "+27.8 points and no language worse"),
    (["english-lost-a-question"], "reference", "+11.1 points, but English is worse"),
    (["english-gains-a-little"], "reference", "+5.6 points"),
    (["albanian-gains-a-lot"], "albanian-gains-a-lot", "+11.1 points, no language worse"),
    (["parity"], "reference", "parity"),
    (["albanian-gains-a-lot", "better"], "better", "the better of two"),
    (["better", "albanian-gains-a-lot"], "better", "the better of two, given the other way round"),
    (["first-places-lost"], "first-places-lost", "recall@3 decides, not recall@1 or the MRR"),
    # Equal at recall@3 (1 in every language): the better reciprocal ranks win, in either order.
    (["first-places-lost", "better"], "better", "a tie at recall@3 broken by the MRR"),
    (["better", "first-places-lost"], "better", "a tie at recall@3 broken by the MRR, reversed"),
]


def _order(ranks: Ranks) -> list[str]:
    """All the documents, nearest first: those with a place at their place, the others in the gaps."""
    placed = {rank: document for document, rank in ranks.items()}
    gaps = iter(document for document in DOCUMENTS if document not in ranks)
    return [placed.get(rank) or next(gaps) for rank in range(1, len(DOCUMENTS) + 1)]


class Script:
    """An embedder that ranks as its plan says.

    A passage is the direction of its document. A question mixes all the directions, with weights
    that halve from the document the plan puts first, so that the passages nearest to it are that
    document's, then the next one's, and so on: cosine distance orders them as the plan does.
    """

    dimension = len(DOCUMENTS)

    def __init__(self, name: str, plan: Plan, passages: Mapping[str, str]) -> None:
        self.identity = f"script:{name}"
        self._passages = passages
        self._orders = {case.question: _order(plan[case.id]) for case in CASES}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._passage(t) if t in self._passages else self._question(t) for t in texts]

    def _passage(self, text: str) -> list[float]:
        vector = [0.0] * len(DOCUMENTS)
        vector[DOCUMENTS.index(self._passages[text])] = 1.0
        return vector

    def _question(self, text: str) -> list[float]:
        weights = [0.0] * len(DOCUMENTS)
        for rank, document in enumerate(self._orders[text]):
            weights[DOCUMENTS.index(document)] = 0.5**rank
        norm = math.sqrt(sum(weight * weight for weight in weights))
        return [weight / norm for weight in weights]


def _passages(session: Session, business_id: str) -> dict[str, str]:
    """The text the experiment embeds for each chunk of the active knowledge, and its document."""
    rows = session.execute(
        select(Chunk.heading, Chunk.content, Document.doc_key)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.knowledge_version_id == active_version_id(session, business_id))
    ).all()
    passages = {f"{row.heading}\n{row.content}": row.doc_key for row in rows}
    assert set(passages.values()) == set(DOCUMENTS)
    return passages


@pytest.fixture
def clinic(
    activate_pack: Activate, database: tuple[str, str], clinic_pack: BusinessPack
) -> sessionmaker[Session]:
    """The demo clinic's knowledge, active in a database whose owner the experiment is."""
    activate_pack(clinic_pack)
    return make_session_factory(make_engine(database[0]))


@pytest.fixture
def measured(
    clinic: sessionmaker[Session], clinic_pack: BusinessPack
) -> dict[str, CandidateResult]:
    results: dict[str, CandidateResult] = {}
    with business_session(clinic, clinic_pack.id) as session:
        passages = _passages(session, clinic_pack.id)
        for name, plan in PLANS.items():
            script = Script(name, plan, passages)
            results[name] = run_candidate(
                session,
                business_id=clinic_pack.id,
                cases=CASES,
                candidate=Candidate(name, len(DOCUMENTS), script, script),
                today=TODAY,
                spent_usd=lambda: 0.0,
            )
    return results


def _figures(result: CandidateResult) -> dict[str, tuple[float, float, float, float]]:
    return {
        language: (scores.recall[1], scores.recall[3], scores.recall[5], scores.mrr)
        for language, scores in result.languages.items()
    }


def test_the_figures_of_the_reference_are_those_its_places_give(
    measured: dict[str, CandidateResult],
) -> None:
    reference = measured["reference"]
    assert {language: scores.cases for language, scores in reference.languages.items()} == {
        "en": 3,
        "it": 3,
        "sq": 3,
    }
    assert _figures(reference) == {
        language: pytest.approx(figures) for language, figures in REFERENCE_FIGURES.items()
    }
    assert reference.mean_recall(3) == pytest.approx(13 / 18)
    assert reference.mean_mrr == pytest.approx(
        sum(figures[3] for figures in REFERENCE_FIGURES.values()) / 3
    )


def test_the_figures_of_every_candidate_are_those_its_places_give(
    measured: dict[str, CandidateResult],
) -> None:
    # All four figures of the candidate that finds everything, but never first ...
    assert _figures(measured["first-places-lost"]) == {
        language: pytest.approx(figures) for language, figures in FIRST_PLACES_LOST_FIGURES.items()
    }
    # ... and the recall@3 of each language for the others, which is what the rule reads.
    for name, expected in RECALL_AT_3.items():
        got = {language: scores.recall[3] for language, scores in measured[name].languages.items()}
        assert got == pytest.approx(expected), name


def test_the_rule_gives_the_verdicts_worked_out_by_hand(
    measured: dict[str, CandidateResult],
) -> None:
    for names, chosen, why in VERDICTS:
        decision = decide(measured["reference"], [measured[name] for name in names])
        assert decision.chosen == chosen, why


def test_the_reasons_the_rule_gives_are_those_of_the_figures(
    measured: dict[str, CandidateResult],
) -> None:
    losers = ["english-lost-a-question", "english-gains-a-little", "parity"]
    decision = decide(measured["reference"], [measured[name] for name in losers])
    assert decision.chosen == "reference"
    assert decision.reasons == (
        "english-lost-a-question: recall@3 is worse in en",
        "english-gains-a-little: mean recall@3 gains 5.6 points, under 10",
        "parity: mean recall@3 gains 0.0 points, under 10",
        "reference stays",
    )


def test_the_saved_results_and_the_report_keep_the_figures(
    measured: dict[str, CandidateResult],
) -> None:
    for result in measured.values():
        assert result_from_json(result_json(result)) == result
    reference, better = measured["reference"], measured["better"]
    report = render_experiment([reference, better], decide(reference, [better]), usd_per_eur=1.0)
    lines = report.splitlines()
    # English, Italian and Albanian (alphabetical order): recall@1, @3, @5 and MRR of each, then the
    # mean recall@3 that the rule reads.
    reference_row = (
        "| reference | 5 | 0.50 | 0.83 | 1.00 | 0.83 | 0.33 | 0.67 | 1.00 | 0.51 "
        "| 0.33 | 0.67 | 1.00 | 0.58 | 0.72 | "
    )
    better_row = (
        "| better | 5 | 0.83 | 1.00 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 | 0.61 "
        "| 0.67 | 1.00 | 1.00 | 0.83 | 1.00 | "
    )
    assert any(line.startswith(reference_row) for line in lines)
    assert any(line.startswith(better_row) for line in lines)
    assert "**Decision: better**" in lines
    assert "- better: mean recall@3 gains 27.8 points" in lines


# --- What indexing and the questions cost, and how long they take ------------------------------


class Ledger:
    """What the embedders so far have spent: embedders add to it, the experiment reads it."""

    def __init__(self, spent: float) -> None:
        self.spent = spent

    def __call__(self) -> float:
        return self.spent


class Priced:
    """An embedder that adds what each of its texts costs to a ledger."""

    def __init__(self, inner: Embedder, ledger: Ledger, usd_per_text: float) -> None:
        self.dimension, self.identity = inner.dimension, inner.identity
        self._inner, self._ledger, self._usd_per_text = inner, ledger, usd_per_text
        self.texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += len(texts)
        self._ledger.spent += self._usd_per_text * len(texts)
        return self._inner.embed(texts)


def test_the_cost_and_the_time_of_indexing_are_kept_apart_from_those_of_the_questions(
    clinic: sessionmaker[Session], clinic_pack: BusinessPack, businesses_dir: Path
) -> None:
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    ledger = Ledger(spent=2.0)  # the embedders before this one have spent that
    small = SmallHashEmbedder(16)
    documents, queries = Priced(small, ledger, 0.05), Priced(small, ledger, 0.01)
    # Indexing takes 4 s; the n-th question takes n quarters of a second, whenever it is asked.
    ticks = [0.0, 4.0]
    for n in range(1, len(cases) + 1):
        ticks += [10.0 * n, 10.0 * n + 0.25 * n]
    with business_session(clinic, clinic_pack.id) as session:
        result = run_candidate(
            session,
            business_id=clinic_pack.id,
            cases=cases,
            candidate=Candidate("priced", 16, documents, queries),
            today=TODAY,
            spent_usd=ledger,
            clock=iter(ticks).__next__,
        )
    assert documents.texts > 0  # every chunk of the active knowledge, in one call
    assert queries.texts == len(cases)
    assert result.index_usd == pytest.approx(0.05 * documents.texts)
    assert result.query_usd == pytest.approx(0.01 * len(cases))
    assert ledger.spent == pytest.approx(2.0 + result.index_usd + result.query_usd)
    assert result.index_seconds == 4.0
    assert result.query_seconds == tuple(0.25 * n for n in range(1, len(cases) + 1))
