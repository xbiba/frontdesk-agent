import json
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session
from frontdesk.db.models import EMBEDDING_DIM
from frontdesk.knowledge.embedders import Embedder, HashEmbedder
from frontdesk.knowledge.evaluation import (
    CaseResult,
    DatasetError,
    RetrievalCase,
    RetrievalReport,
    load_cases,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
    run_retrieval_eval,
    to_json,
    unknown_relevant_documents,
)
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

TODAY = date(2026, 10, 7)
Activate = Callable[..., sessionmaker[Session]]
# The first parking chunk as it is embedded: its heading, a newline, its text.
PARKING_SECTION = (
    "Parking and access > Parking\n"
    "Klinika Demo has six free parking spaces behind the building. Street parking is paid on weekdays."
)


def test_recall_counts_relevant_documents_in_the_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert recall_at_k(["a"], ["a"], 5) == 1.0


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(["x", "y", "a"], ["a", "y"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_a_case_without_relevant_documents_is_invalid() -> None:
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], [], 5)


def test_datasets_load(businesses_dir: Path) -> None:
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    assert len(cases) == 8
    assert {"cross-lingual", "no-diacritics", "paraphrase"} <= {t for c in cases for t in c.tags}


@pytest.mark.db
def test_harness_reports_every_case(
    activate_pack: Callable[..., sessionmaker[Session]],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
) -> None:
    factory = activate_pack(clinic_pack)
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    with business_session(factory, clinic_pack.id) as session:
        report = run_retrieval_eval(
            session,
            business_id=clinic_pack.id,
            cases=cases,
            embedder=HashEmbedder(),
            k=5,
            today=TODAY,
        )
    assert len(report.results) == len(cases)
    lexical = {r.case.id: r.recall for r in report.results}
    assert lexical["parking-en"] == 1.0
    assert lexical["first-visit-it"] == 1.0
    assert set(report.by_language()) == {"en", "it", "sq"}
    markdown = render_markdown(report)
    assert "| case | language | recall@5 | RR |" in markdown
    assert "hash-v1" in markdown


def test_recall_only_counts_the_top_k_and_divides_by_the_relevant_documents() -> None:
    assert recall_at_k(["a", "b"], ["b"], 1) == 0.0  # b is ranked below k
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5  # one of two relevant, not of three
    assert recall_at_k([], ["a"], 5) == 0.0


def _result(
    case_id: str, language: str, recall: float, rank_score: float, *tags: str
) -> CaseResult:
    case = RetrievalCase(
        id=case_id,
        question=f"Domanda {case_id}: può?",
        language=language,
        relevant=("doc",),
        tags=tags,
    )
    return CaseResult(case=case, retrieved=("doc",), recall=recall, reciprocal_rank=rank_score)


def _report() -> RetrievalReport:
    """Three cases whose recall and reciprocal rank differ (a swapped column shows), listed in
    neither language nor tag order (a missing sort shows)."""
    return RetrievalReport(
        business_id="demo",
        k=5,
        embedder="hash-v1",
        results=(
            _result("a", "it", 0.5, 1.0, "paraphrase", "cross-lingual"),
            _result("b", "en", 1.0, 1.0),
            _result("c", "it", 0.0, 0.0, "cross-lingual"),
        ),
    )


def test_means_cover_every_case() -> None:
    report = _report()
    assert report.mean_recall == pytest.approx(0.5)
    assert report.mrr == pytest.approx(2 / 3)


def test_an_empty_report_scores_zero_and_still_renders() -> None:
    report = RetrievalReport(business_id="demo", k=5, embedder="hash-v1", results=())
    assert (report.mean_recall, report.mrr) == (0.0, 0.0)
    assert report.by_language() == {}
    assert report.by_tag() == {}
    assert "Mean recall@5: 0.00 · MRR: 0.00" in render_markdown(report)


def test_groups_average_recall_and_reciprocal_rank_apart_in_name_order() -> None:
    report = _report()
    assert report.by_language() == {"en": (1.0, 1.0), "it": (0.25, 0.5)}
    # A case with two tags counts in both groups, a case without tags in none.
    assert report.by_tag() == {"cross-lingual": (0.25, 0.5), "paraphrase": (0.5, 1.0)}
    assert list(report.by_language()) == ["en", "it"]
    assert list(report.by_tag()) == ["cross-lingual", "paraphrase"]


def test_markdown_tabulates_the_cases_then_the_groups() -> None:
    assert render_markdown(_report()) == "\n".join(
        [
            "# Retrieval evaluation — demo",
            "",
            "Embedder: hash-v1 · k = 5 · cases: 3",
            "Mean recall@5: 0.50 · MRR: 0.67",
            "",
            "| case | language | recall@5 | RR |",
            "|---|---|---|---|",
            "| a | it | 0.50 | 1.00 |",
            "| b | en | 1.00 | 1.00 |",
            "| c | it | 0.00 | 0.00 |",
            "",
            "| group | recall | MRR |",
            "|---|---|---|",
            "| lang:en | 1.00 | 1.00 |",
            "| lang:it | 0.25 | 0.50 |",
            "| tag:cross-lingual | 0.25 | 0.50 |",
            "| tag:paraphrase | 0.50 | 1.00 |",
        ]
    )


def test_json_carries_the_summary_and_every_case() -> None:
    raw = to_json(_report())
    data = json.loads(raw)
    assert (data["business_id"], data["k"], data["embedder"]) == ("demo", 5, "hash-v1")
    assert data["mean_recall"] == pytest.approx(0.5)
    assert data["mrr"] == pytest.approx(2 / 3)
    assert [result["case"]["id"] for result in data["results"]] == ["a", "b", "c"]
    assert data["results"][0] == {
        "case": {
            "id": "a",
            "question": "Domanda a: può?",
            "language": "it",
            "relevant": ["doc"],
            "tags": ["paraphrase", "cross-lingual"],
        },
        "retrieved": ["doc"],
        "recall": 0.5,
        "reciprocal_rank": 1.0,
    }
    assert "può" in raw  # readable text, not \u escapes


def test_cases_load_as_immutable_values(tmp_path: Path) -> None:
    dataset = tmp_path / "retrieval.yaml"
    dataset.write_text(
        '- id: pay-it\n  question: "Si può pagare con la carta?"\n  language: it\n'
        "  relevant: [payments-sq]\n"
        "- id: pay-sq\n  question: A pranoni karte?\n  language: sq\n"
        "  relevant: [payments-sq, payments-it]\n  tags: [no-diacritics]\n",
        encoding="utf-8",
    )
    assert load_cases(dataset) == [
        RetrievalCase(
            id="pay-it",
            question="Si può pagare con la carta?",
            language="it",
            relevant=("payments-sq",),
        ),
        RetrievalCase(
            id="pay-sq",
            question="A pranoni karte?",
            language="sq",
            relevant=("payments-sq", "payments-it"),
            tags=("no-diacritics",),
        ),
    ]


FINE_CASE = "- id: fine\n  question: Is there parking?\n  language: en\n  relevant: [parking-en]\n"


def _dataset(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "retrieval.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _refusal(path: Path) -> str:
    with pytest.raises(DatasetError) as refused:
        load_cases(path)
    return str(refused.value)


@pytest.mark.parametrize(
    ("bad_case", "named"),
    [
        pytest.param(
            "- id: no-relevant\n  question: Q?\n  language: en\n",
            ("case 'no-relevant'", "relevant"),
            id="relevant missing",
        ),
        pytest.param(
            "- id: no-documents\n  question: Q?\n  language: en\n  relevant: []\n",
            ("case 'no-documents'", "relevant"),
            id="relevant empty",
        ),
        pytest.param(
            "- id: one-name\n  question: Q?\n  language: en\n  relevant: parking-en\n",
            ("case 'one-name'", "relevant"),
            id="relevant is text, not a list",
        ),
        pytest.param(
            "- id: blank-name\n  question: Q?\n  language: en\n  relevant: ['  ']\n",
            ("case 'blank-name'", "relevant"),
            id="relevant names a blank document",
        ),
        pytest.param(
            "- id: blank-tag\n  question: Q?\n  language: en\n  relevant: [a]\n  tags: ['']\n",
            ("case 'blank-tag'", "tags"),
            id="tags has a blank tag",
        ),
        pytest.param(
            "- id: french\n  question: Q?\n  language: fr\n  relevant: [a]\n",
            ("case 'french'", "language"),
            id="unknown language",
        ),
        pytest.param(
            "- id: no-language\n  question: Q?\n  relevant: [a]\n",
            ("case 'no-language'", "language"),
            id="language missing",
        ),
        pytest.param(
            "- id: no-question\n  language: en\n  relevant: [a]\n",
            ("case 'no-question'", "question"),
            id="question missing",
        ),
        pytest.param(
            "- id: blank-question\n  question: ''\n  language: en\n  relevant: [a]\n",
            ("case 'blank-question'", "question"),
            id="question blank",
        ),
        pytest.param(
            "- id: typo\n  question: Q?\n  language: en\n  relevant: [a]\n  tag: [x]\n",
            ("case 'typo'", "tag"),
            id="unknown field",
        ),
        pytest.param(
            "- id: one-tag\n  question: Q?\n  language: en\n  relevant: [a]\n  tags: paraphrase\n",
            ("case 'one-tag'", "tags"),
            id="tags is text, not a list",
        ),
        pytest.param(
            "- question: Q?\n  language: en\n  relevant: [a]\n",
            ("case #2", "id"),
            id="id missing",
        ),
        pytest.param(
            "- id: ' '\n  question: Q?\n  language: en\n  relevant: [a]\n",
            ("case #2", "id"),
            id="id is blank",
        ),
        pytest.param(
            "- id: 7\n  question: Q?\n  language: en\n  relevant: [a]\n",
            ("case #2", "id"),
            id="id is a number",
        ),
        pytest.param("- just a sentence\n", ("case #2", "a mapping"), id="case is not a mapping"),
    ],
)
def test_a_malformed_case_is_refused_naming_the_file_and_the_case(
    tmp_path: Path, bad_case: str, named: tuple[str, ...]
) -> None:
    path = _dataset(tmp_path, FINE_CASE + bad_case)
    message = _refusal(path)
    assert str(path) in message
    for fragment in named:
        assert fragment in message
    assert "case 'fine'" not in message  # the good case is not blamed


def test_every_malformed_case_is_reported_at_once(tmp_path: Path) -> None:
    path = _dataset(
        tmp_path,
        "- id: french\n  question: Q?\n  language: fr\n  relevant: [a]\n"
        + FINE_CASE
        + "- id: no-documents\n  question: Q?\n  language: en\n  relevant: []\n",
    )
    message = _refusal(path)
    assert "case 'french'" in message
    assert "case 'no-documents'" in message
    assert "case 'fine'" not in message


def test_a_repeated_id_is_refused_naming_the_file_and_the_id(tmp_path: Path) -> None:
    path = _dataset(tmp_path, FINE_CASE + FINE_CASE)
    message = _refusal(path)
    assert str(path) in message
    assert "case 'fine'" in message
    assert "duplicate" in message


@pytest.mark.parametrize(
    ("content", "named"),
    [
        pytest.param("", "no cases", id="empty file"),
        pytest.param("[]\n", "no cases", id="empty list"),
        pytest.param("id: fine\nquestion: Q?\n", "list of cases", id="one mapping, not a list"),
        pytest.param("- id: [unclosed\n", "invalid YAML", id="broken YAML"),
    ],
)
def test_a_file_that_is_not_a_list_of_cases_is_refused_naming_the_file(
    tmp_path: Path, content: str, named: str
) -> None:
    path = _dataset(tmp_path, content)
    message = _refusal(path)
    assert str(path) in message
    assert named in message


def test_a_file_that_is_not_utf8_text_is_refused_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_bytes("- id: caffè\n".encode("latin-1"))
    message = _refusal(path)
    assert str(path) in message
    assert "UTF-8" in message


def test_a_missing_file_is_refused_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "evals" / "retrieval.yaml"
    message = _refusal(path)
    assert str(path) in message
    assert "cannot be read" in message


@pytest.mark.parametrize("business", ["demo-clinic", "demo-shop"])
def test_dataset_cases_point_at_documents_and_languages_of_their_business(
    businesses_dir: Path, business: str
) -> None:
    pack = load_pack(businesses_dir / business)
    cases = load_cases(businesses_dir / business / "evals" / "retrieval.yaml")
    documents = {document.key for document in pack.documents}
    assert cases
    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        assert case.relevant, case.id
        assert set(case.relevant) <= documents, case.id
        assert case.language in pack.languages, case.id


def _evaluate(
    factory: sessionmaker[Session],
    pack: BusinessPack,
    cases: Sequence[RetrievalCase],
    *,
    k: int,
    embedder: Embedder | None = None,
    today: date = TODAY,
) -> RetrievalReport:
    with business_session(factory, pack.id) as session:
        return run_retrieval_eval(
            session,
            business_id=pack.id,
            cases=cases,
            embedder=embedder or HashEmbedder(),
            k=k,
            today=today,
        )


@pytest.mark.db
def test_a_report_keeps_each_top_document_once_and_no_more_than_k(
    activate_pack: Activate, clinic_pack: BusinessPack, businesses_dir: Path
) -> None:
    factory = activate_pack(clinic_pack)
    cases = load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
    report = _evaluate(factory, clinic_pack, cases, k=2)
    assert (report.business_id, report.k, report.embedder) == (clinic_pack.id, 2, "hash-v1")
    assert [result.case for result in report.results] == cases
    for result in report.results:
        # Six chunks (k * 3) of five documents repeat a document, which must still count once.
        assert len(result.retrieved) == 2
        assert len(set(result.retrieved)) == 2


@pytest.mark.db
def test_the_right_document_ranks_first_when_only_it_shares_the_question_words(
    activate_pack: Activate, clinic_pack: BusinessPack, businesses_dir: Path
) -> None:
    """With k = 1 recall is not saturated. Only the right document contains a search term of these
    questions, so its best chunk is ranked by both searches and beats every chunk that only the
    vector search found, whatever way the equal scores among those are ordered."""
    unambiguous = {
        "parking-en",
        "physio-prepare-it",
        "physio-prepare-sq-plain",
        "payments-sq",
        "accessibility-en",
    }
    factory = activate_pack(clinic_pack)
    cases = [
        case
        for case in load_cases(businesses_dir / "demo-clinic" / "evals" / "retrieval.yaml")
        if case.id in unambiguous
    ]
    assert len(cases) == len(unambiguous)
    report = _evaluate(factory, clinic_pack, cases, k=1)
    assert [(r.recall, r.reciprocal_rank) for r in report.results] == [(1.0, 1.0)] * len(cases)


@pytest.mark.db
def test_recall_counts_the_relevant_documents_that_fit_in_k(
    activate_pack: Activate, shop_pack: BusinessPack, businesses_dir: Path
) -> None:
    """The returns questions have two relevant documents among the shop's three."""
    factory = activate_pack(shop_pack)
    cases = [
        case
        for case in load_cases(businesses_dir / "demo-shop" / "evals" / "retrieval.yaml")
        if case.id.startswith("returns-")
    ]
    assert len(cases) == 2
    one_slot = _evaluate(factory, shop_pack, cases, k=1)
    assert [(r.recall, r.reciprocal_rank) for r in one_slot.results] == [(0.5, 1.0)] * 2
    assert one_slot.mean_recall == 0.5
    every_document = _evaluate(factory, shop_pack, cases, k=3)
    assert [r.recall for r in every_document.results] == [1.0, 1.0]


@pytest.mark.db
def test_the_evaluation_searches_the_knowledge_valid_on_the_given_day(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    documents = [
        doc.model_copy(update={"valid_until": TODAY - timedelta(days=1)})
        if doc.key == "parking-en"
        else doc
        for doc in clinic_pack.documents
    ]
    pack = clinic_pack.model_copy(update={"documents": documents})
    factory = activate_pack(pack)
    case = RetrievalCase(
        id="parking",
        question="Is there parking near the clinic?",
        language="en",
        relevant=("parking-en",),
    )
    still_valid = _evaluate(factory, pack, [case], k=5, today=TODAY - timedelta(days=2))
    expired = _evaluate(factory, pack, [case], k=5, today=TODAY)
    assert still_valid.results[0].recall == 1.0
    assert expired.results[0].recall == 0.0
    assert "parking-en" not in expired.results[0].retrieved


class _ParkingEmbedder:
    """Gives every question the vector of the parking section, whatever the question says."""

    dimension = EMBEDDING_DIM
    identity = "parking-test"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.questions.extend(texts)
        return HashEmbedder().embed([PARKING_SECTION] * len(texts))


class _BlankEmbedder:
    """Has nothing to say about any question: the zero vector."""

    dimension = EMBEDDING_DIM
    identity = "blank-test"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIM for _ in texts]


@pytest.mark.db
def test_the_question_is_embedded_by_the_given_embedder(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    case = RetrievalCase(id="noise", question="qqq zzz", language="en", relevant=("parking-en",))
    embedder = _ParkingEmbedder()
    report = _evaluate(factory, clinic_pack, [case], k=1, embedder=embedder)
    assert embedder.questions == ["qqq zzz"]
    assert report.embedder == "parking-test"
    # No document contains these words, so only the vector can have found the parking section.
    assert report.results[0].retrieved == ("parking-en",)


@pytest.mark.db
def test_a_zero_vector_leaves_the_search_to_the_words_of_the_question(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    """The search takes a zero vector (what HashEmbedder gives a question without tokens) for no
    vector: a question with words is found by them, one without is a miss, never arbitrary text."""
    factory = activate_pack(clinic_pack)
    cases = [
        RetrievalCase(
            id="words",
            question="Is there parking near the clinic?",
            language="en",
            relevant=("parking-en",),
        ),
        RetrievalCase(id="silence", question="?? !!", language="en", relevant=("parking-en",)),
    ]
    report = _evaluate(factory, clinic_pack, cases, k=1, embedder=_BlankEmbedder())
    words, silence = report.results
    assert (words.retrieved, words.recall, words.reciprocal_rank) == (("parking-en",), 1.0, 1.0)
    assert (silence.retrieved, silence.recall, silence.reciprocal_rank) == ((), 0.0, 0.0)


@pytest.mark.db
def test_relevant_keys_the_active_version_lacks_are_unknown(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    cases = [
        RetrievalCase("a", "q", "en", ("parking-en", "parkng-en")),
        RetrievalCase("b", "q", "it", ("first-visit-it",)),
        RetrievalCase("c", "q", "sq", ("nope", "parking-en", "also-nope")),
    ]
    with business_session(factory, clinic_pack.id) as session:
        unknown = unknown_relevant_documents(session, business_id=clinic_pack.id, cases=cases)
    assert unknown == [("a", "parkng-en"), ("c", "nope"), ("c", "also-nope")]


@pytest.mark.db
def test_a_business_without_an_active_version_knows_no_document(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    cases = [RetrievalCase("a", "q", "en", ("parking-en",))]
    with business_session(factory, "another-business") as session:
        unknown = unknown_relevant_documents(session, business_id="another-business", cases=cases)
    assert unknown == [("a", "parking-en")]


@pytest.mark.db
def test_a_document_outside_its_validity_window_is_still_a_known_document(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = clinic_pack.model_copy(
        update={
            "documents": [
                document.model_copy(update={"valid_until": date(2020, 1, 1)})
                for document in clinic_pack.documents
            ]
        }
    )
    factory = activate_pack(pack)
    cases = [RetrievalCase("a", "q", "en", ("parking-en",))]
    with business_session(factory, pack.id) as session:
        assert unknown_relevant_documents(session, business_id=pack.id, cases=cases) == []
