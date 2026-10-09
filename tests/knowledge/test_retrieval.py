import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import EMBEDDING_DIM
from frontdesk.knowledge.chunking import Chunk as Section
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import HashEmbedder, cosine
from frontdesk.knowledge.evidence import EvidenceKind
from frontdesk.knowledge.retrieval import (
    RetrievedChunk,
    build_tsquery,
    chunk_evidence,
    hybrid_search,
    rrf_fuse,
)
from frontdesk.knowledge.versions import activate, active_version_id, approve, create_draft
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

TODAY = date(2026, 10, 7)
NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
Activate = Callable[..., sessionmaker[Session]]
NewDatabase = Callable[[], tuple[str, str]]
PARKING_TEXT = "Parking\nKlinika Demo has six free parking spaces behind the building."


def test_rrf_prefers_items_ranked_by_both_lists() -> None:
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    fused = rrf_fuse([[a, b], [b, c]], {a: ("a", 0), b: ("b", 0), c: ("c", 0)})
    assert fused[0][0] == b
    assert {item for item, _ in fused} == {a, b, c}


def test_rrf_ranks_equal_scores_by_document_key_then_position_never_by_id() -> None:
    # Each is first in one list, so all three score the same. Their ids sort the other way round.
    by_position = [uuid.UUID(int=3), uuid.UUID(int=2), uuid.UUID(int=1)]
    places = {
        by_position[0]: ("alpha", 0),
        by_position[1]: ("alpha", 1),
        by_position[2]: ("beta", 0),
    }
    fused = rrf_fuse([[by_position[2]], [by_position[1]], [by_position[0]]], places)
    assert [item for item, _ in fused] == by_position
    assert len({score for _, score in fused}) == 1


def test_tsquery_normalizes_and_drops_stopwords() -> None:
    assert build_tsquery("Si të përgatitem për fizioterapi?") == "fizioterapi | pergatitem"
    assert build_tsquery("?? !!") is None


def _search(
    factory: sessionmaker[Session], business_id: str, query: str, **kwargs: Any
) -> list[RetrievedChunk]:
    with business_session(factory, business_id) as session:
        return hybrid_search(session, business_id=business_id, query=query, today=TODAY, **kwargs)


@pytest.mark.db
def test_full_text_finds_the_document(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    hits = _search(factory, clinic_pack.id, "Is there parking?", query_vector=None)
    assert hits[0].document_key == "parking-en"


@pytest.mark.db
def test_albanian_without_diacritics_matches(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    hits = _search(factory, clinic_pack.id, "si te pergatiteni per fizioterapia", query_vector=None)
    assert hits[0].document_key == "physiotherapy-sq"


@pytest.mark.db
def test_vector_ranking_is_used(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    vector = HashEmbedder().embed(["free parking spaces behind the building"])[0]
    hits = _search(factory, clinic_pack.id, "", query_vector=vector)
    assert hits[0].document_key == "parking-en"


@pytest.mark.db
def test_a_zero_vector_ranks_nothing(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    """HashEmbedder gives a message without tokens the zero vector; its cosine distance is NaN."""
    factory = activate_pack(clinic_pack)
    message = "?? !!"
    vector = HashEmbedder().embed([message])[0]
    assert not any(vector)
    assert _search(factory, clinic_pack.id, message, query_vector=vector) == []


@pytest.mark.db
def test_a_zero_vector_leaves_the_full_text_ranking_alone(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    zero = [0.0] * EMBEDDING_DIM
    with_zero = _search(factory, clinic_pack.id, "parking", query_vector=zero)
    assert with_zero == _search(factory, clinic_pack.id, "parking", query_vector=None)
    assert {hit.document_key for hit in with_zero} == {"parking-en"}


@pytest.mark.db
def test_drafts_are_not_searchable(app_url: str, clinic_pack: BusinessPack) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, clinic_pack.id) as session:
        create_draft(session, clinic_pack)
    assert _search(factory, clinic_pack.id, "parking", query_vector=None) == []


def _with_parking(pack: BusinessPack, **changes: Any) -> BusinessPack:
    documents = [
        doc.model_copy(update=changes) if doc.key == "parking-en" else doc for doc in pack.documents
    ]
    return pack.model_copy(update={"documents": documents})


def _with_rooftop_garage(pack: BusinessPack) -> BusinessPack:
    """The same pack with another parking text: a second version of the same knowledge."""
    parking = next(doc for doc in pack.documents if doc.key == "parking-en")
    body = parking.body.replace("six free parking spaces", "a rooftop garage")
    return _with_parking(pack, body=body)


@pytest.mark.db
def test_only_the_active_version_is_searched(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    edited = _with_rooftop_garage(clinic_pack)
    with business_session(factory, edited.id) as session:
        create_draft(session, edited)

    def finds(word: str) -> bool:
        hits = _search(factory, clinic_pack.id, word, query_vector=None)
        return any(hit.document_key == "parking-en" for hit in hits)

    assert finds("spaces")
    assert not finds("rooftop")  # the draft waits for approval
    activate_pack(edited)
    assert finds("rooftop")
    assert not finds("spaces")  # the retired version is gone


@pytest.mark.db
def test_vector_search_sees_only_the_active_version(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    edited = _with_rooftop_garage(clinic_pack)
    vector = HashEmbedder().embed([PARKING_TEXT])[0]  # nearest to the first version's chunk
    with business_session(factory, clinic_pack.id) as session:
        first = active_version_id(session, clinic_pack.id)
        create_draft(session, edited)
    hits = _search(factory, clinic_pack.id, "", query_vector=vector)
    assert {hit.version_id for hit in hits} == {first}  # the draft waits for approval
    activate_pack(edited)
    with business_session(factory, clinic_pack.id) as session:
        second = active_version_id(session, clinic_pack.id)
    hits = _search(factory, clinic_pack.id, "", query_vector=vector)
    assert {hit.version_id for hit in hits} == {second}  # the retired version is gone


@pytest.mark.db
def test_vector_search_stays_exact_when_the_planner_prefers_the_hnsw_index(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    """An HNSW scan filters after it scans: a retired copy must not hide the active chunks."""
    factory = activate_pack(clinic_pack)
    activate_pack(_with_rooftop_garage(clinic_pack))  # the first version is retired, vectors kept
    vector = HashEmbedder().embed(["six free parking spaces"])[0]  # nearest to the retired chunk
    with business_session(factory, clinic_pack.id) as session:
        for statement in (
            "SET LOCAL enable_seqscan = off",  # prefer index scans...
            "SET LOCAL enable_sort = off",  # ...and an index-ordered scan to a sort
            "SET LOCAL hnsw.ef_search = 1",  # the index then returns one row only
        ):
            session.execute(text(statement))
        hits = hybrid_search(
            session,
            business_id=clinic_pack.id,
            query="",
            query_vector=vector,
            today=TODAY,
            limit=3,
        )
    assert len(hits) == 3


@pytest.mark.db
def test_other_businesses_are_invisible(
    activate_pack: Activate, clinic_pack: BusinessPack, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    activate_pack(shop_pack)
    clinic_hits = _search(factory, clinic_pack.id, "reso articolo etichetta", query_vector=None)
    assert all(hit.document_key not in {"returns-it", "returns-en"} for hit in clinic_hits)
    with business_session(factory, clinic_pack.id) as session:
        cross = hybrid_search(
            session, business_id=shop_pack.id, query="reso", query_vector=None, today=TODAY
        )
    assert cross == []


@pytest.mark.db
def test_expired_documents_are_excluded(
    activate_pack: Activate, businesses_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "demo-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", target)
    parking = target / "knowledge" / "parking-en.md"
    parking.write_text(
        parking.read_text(encoding="utf-8").replace(
            "language: en\n", "language: en\nvalid_until: 2020-01-01\n"
        ),
        encoding="utf-8",
    )
    pack = load_pack(target)
    factory = activate_pack(pack)
    hits = _search(factory, pack.id, "parking", query_vector=None)
    assert all(hit.document_key != "parking-en" for hit in hits)


def _parking_hits(
    factory: sessionmaker[Session], pack: BusinessPack, *, by_vector: bool
) -> list[RetrievedChunk]:
    """One search through one side only: the term "parking", or a vector near the parking chunk."""
    if by_vector:
        vector = HashEmbedder().embed([PARKING_TEXT])[0]
        return _search(factory, pack.id, "", query_vector=vector)
    return _search(factory, pack.id, "parking", query_vector=None)


@pytest.mark.db
@pytest.mark.parametrize("by_vector", [False, True], ids=["terms", "vector"])
@pytest.mark.parametrize(
    ("window", "searchable"),
    [
        ({"valid_from": TODAY + timedelta(days=1)}, False),
        ({"valid_from": TODAY}, True),
        ({"valid_until": TODAY - timedelta(days=1)}, False),
        ({"valid_until": TODAY}, True),
    ],
    ids=["starts tomorrow", "starts today", "ended yesterday", "ends today"],
)
def test_documents_are_searchable_from_their_first_to_their_last_day(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    window: dict[str, date],
    searchable: bool,
    by_vector: bool,
) -> None:
    pack = _with_parking(clinic_pack, **window)
    factory = activate_pack(pack)
    hits = _parking_hits(factory, pack, by_vector=by_vector)
    assert any(hit.document_key == "parking-en" for hit in hits) is searchable


@pytest.mark.db
def test_context_budget_limits_results(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    vector = HashEmbedder().embed(["fisioterapia visita parcheggio"])[0]
    hits = _search(factory, clinic_pack.id, "fisioterapia visita", query_vector=vector, max_chars=1)
    assert len(hits) == 1


@pytest.mark.db
def test_limit_caps_the_results(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    vector = HashEmbedder().embed(["fisioterapia visita parcheggio"])[0]
    hits = _search(factory, clinic_pack.id, "fisioterapia visita", query_vector=vector, limit=2)
    assert len(hits) == 2


@pytest.mark.db
def test_chunks_become_document_evidence(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    hit = _search(factory, clinic_pack.id, "parking", query_vector=None)[0]
    evidence = chunk_evidence(hit)
    assert evidence.kind is EvidenceKind.DOCUMENT
    assert evidence.source_ref.startswith("parking-en#")
    assert evidence.content == hit.content
    assert (hit.language, evidence.language) == ("en", "en")


@pytest.mark.db
def test_every_hit_carries_the_language_of_its_document(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    query = "parking fisioterapia fizioterapi visita pagesa"
    vector = HashEmbedder().embed([query])[0]
    hits = _search(
        factory, clinic_pack.id, query, query_vector=vector, limit=20, max_chars=1_000_000
    )
    languages = {document.key: document.language for document in clinic_pack.documents}
    assert len({hit.document_key for hit in hits}) >= 3
    assert {languages[hit.document_key] for hit in hits} >= {"en", "it", "sq"}
    assert all(hit.language == languages[hit.document_key] for hit in hits)
    assert all(chunk_evidence(hit).language == hit.language for hit in hits)


def _activated_in_new_database(
    new_database: NewDatabase, pack: BusinessPack
) -> sessionmaker[Session]:
    """The pack, active in a database of its own: same content, other random ids."""
    _, app_url = new_database()
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, pack.id) as session:
        draft = create_draft(session, pack)
        approve(session, draft.version_id, approved_by="test", evidence="synthetic", now=NOW)
        activate(session, draft.version_id, HashEmbedder(), now=NOW)
    return factory


def _ranking(
    factory: sessionmaker[Session], pack: BusinessPack, query: str, **kwargs: Any
) -> list[tuple[str, str, float]]:
    """What a ranking says without the random ids: document key, section, score."""
    hits = _search(factory, pack.id, query, **kwargs)
    return [(hit.document_key, hit.heading, hit.score) for hit in hits]


def _by_key_then_position(pack: BusinessPack) -> list[tuple[str, str]]:
    """Every section of the pack ordered by document key, then by position in the document."""
    return [
        (document.key, piece.heading)
        for document in sorted(pack.documents, key=lambda document: document.key)
        for piece in chunk_markdown(document.title, document.body)
    ]


@pytest.mark.db
def test_equal_vector_distances_rank_the_same_in_every_database(
    new_database: NewDatabase, clinic_pack: BusinessPack
) -> None:
    """The vector of "zzzz" is at right angles to every section's, so all nine are equally far."""
    vector = HashEmbedder().embed(["zzzz"])[0]
    sections = [
        f"{piece.heading}\n{piece.content}"
        for document in clinic_pack.documents
        for piece in chunk_markdown(document.title, document.body)
    ]
    assert [cosine(vector, other) for other in HashEmbedder().embed(sections)] == [0.0] * 9
    rankings = [
        _ranking(
            _activated_in_new_database(new_database, clinic_pack),
            clinic_pack,
            "zzzz",
            query_vector=vector,
            limit=20,
        )
        for _ in range(2)
    ]
    assert rankings[0] == rankings[1]
    assert [(key, heading) for key, heading, _ in rankings[0]] == _by_key_then_position(clinic_pack)


def _pack_of_copies(pack: BusinessPack, count: int, prefixes: dict[str, str]) -> BusinessPack:
    """A pack of only `count` copies of each document in `prefixes` (key -> key of the copies).

    Copies say the same, so their sections are exactly as relevant as each other.
    """
    documents = [
        document.model_copy(update={"key": f"{prefixes[document.key]}-{n}"})
        for document in pack.documents
        if document.key in prefixes
        for n in range(count)
    ]
    return pack.model_copy(update={"documents": documents})


@pytest.mark.db
def test_equal_full_text_scores_rank_by_document_key_then_position_in_every_database(
    new_database: NewDatabase, clinic_pack: BusinessPack
) -> None:
    """The word is once in the heading of every section of the copies: 16 chunks, one score."""
    pack = _pack_of_copies(clinic_pack, 8, {"first-visit-it": "visit"})
    expected = _by_key_then_position(pack)
    assert len(expected) == 16
    factories = [_activated_in_new_database(new_database, pack) for _ in range(2)]
    rankings = [
        _ranking(factory, pack, "visita", query_vector=None, limit=20) for factory in factories
    ]
    assert rankings[0] == rankings[1]
    assert [(key, heading) for key, heading, _ in rankings[0]] == expected
    # The cut at `candidates` falls among equal scores too: the first sections in place order stay.
    for factory in factories:
        cut = _search(factory, pack.id, "visita", query_vector=None, limit=3, candidates=3)
        assert [(hit.document_key, hit.heading) for hit in cut] == expected[:3]


@pytest.mark.db
def test_equal_fused_scores_rank_by_document_key_then_position_in_every_database(
    new_database: NewDatabase, clinic_pack: BusinessPack
) -> None:
    """Every chunk is in one search only, at the rank of a chunk the other search put there.

    The word "ramp" finds the second section of the eight parking copies, in key order. The vector
    points at the eight copies of one first section, also in key order, so the k-th chunk of one
    search ties with the k-th chunk of the other.
    """
    pack = _pack_of_copies(
        clinic_pack, 8, {"parking-en": "parking-copy", "physiotherapy-it": "physio-copy"}
    )
    parking = next(doc for doc in pack.documents if doc.key == "parking-copy-0")
    physio = next(doc for doc in pack.documents if doc.key == "physio-copy-0")
    parking_first, ramp = chunk_markdown(parking.title, parking.body)  # only `ramp` has the word
    physio_first = chunk_markdown(physio.title, physio.body)[0]

    def points_at(section: Section) -> list[float]:
        return HashEmbedder().embed([f"{section.heading}\n{section.content}"])[0]

    # The vector points at the physio copies: the tied chunks belong to two documents, and the key
    # decides ("parking-copy" before "physio-copy") although the parking section comes later in
    # its document (second, the other first).
    across_documents = [
        place
        for n in range(8)
        for place in (
            (f"parking-copy-{n}", ramp.heading),
            (f"physio-copy-{n}", physio_first.heading),
        )
    ]
    # The vector points at the parking copies: the tied chunks are two sections of one document, and
    # their position in it decides.
    within_a_document = [
        place
        for n in range(8)
        for place in (
            (f"parking-copy-{n}", parking_first.heading),
            (f"parking-copy-{n}", ramp.heading),
        )
    ]
    factories = [_activated_in_new_database(new_database, pack) for _ in range(2)]
    for section, expected in ((physio_first, across_documents), (parking_first, within_a_document)):
        rankings = [
            _ranking(
                factory,
                pack,
                "ramp",
                query_vector=points_at(section),
                candidates=8,
                limit=16,
            )
            for factory in factories
        ]
        assert rankings[0] == rankings[1]
        assert [(key, heading) for key, heading, _ in rankings[0]] == expected
        scores = [score for _, _, score in rankings[0]]
        assert scores[0::2] == scores[1::2]  # the pairs really are equal, not merely adjacent


@pytest.mark.db
@pytest.mark.parametrize("by_vector", [True, False], ids=["vector", "terms"])
def test_equal_scores_rank_by_position_in_the_document_not_by_where_the_row_is_stored(
    new_database: NewDatabase, clinic_pack: BusinessPack, by_vector: bool
) -> None:
    """With the positions of the sections reversed, the one stored first must come last."""
    if by_vector:
        pack, query = clinic_pack, "zzzz"
    else:
        pack, query = _pack_of_copies(clinic_pack, 8, {"first-visit-it": "visit"}), "visita"
    factory = _activated_in_new_database(new_database, pack)
    with business_session(factory, pack.id) as session:
        session.execute(text("UPDATE chunks SET ordinal = 1 - ordinal"))
    vector = HashEmbedder().embed([query])[0] if by_vector else None
    ranking = _ranking(factory, pack, query, query_vector=vector, limit=20)
    expected = [
        (document.key, section.heading)
        for document in sorted(pack.documents, key=lambda document: document.key)
        for section in reversed(chunk_markdown(document.title, document.body))
    ]
    assert [(key, heading) for key, heading, _ in ranking] == expected


@pytest.mark.db
def test_the_vector_search_materializes_the_eligible_chunks(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    """The planner-forcing test above also passes without this, so the SQL itself is checked.

    Unmaterialized, the planner may sort on the HNSW index, which filters after it scans.
    """
    factory = activate_pack(clinic_pack)
    statements: list[str] = []

    def record(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        statements.append(statement)

    with business_session(factory, clinic_pack.id) as session:
        connection = session.connection()
        event.listen(connection, "before_cursor_execute", record)
        try:
            hybrid_search(
                session,
                business_id=clinic_pack.id,
                query="",
                query_vector=HashEmbedder().embed(["parking"])[0],
                today=TODAY,
            )
        finally:
            event.remove(connection, "before_cursor_execute", record)
    assert any("MATERIALIZED" in statement for statement in statements)


@pytest.mark.db
@pytest.mark.parametrize("by_vector", [False, True], ids=["terms", "vector"])
def test_equal_scores_rank_document_keys_by_code_point_in_any_database_collation(
    activate_pack: Activate, clinic_pack: BusinessPack, by_vector: bool
) -> None:
    """`en_US.utf8`, the collation of the test database, compares keys as if their hyphens were
    absent and so puts "faqs" before "faq-zones". Code, and so the ranking, must not."""
    keys = ["faqs", "faq-zones"]  # stored in the opposite order to the one expected
    parking = next(doc for doc in clinic_pack.documents if doc.key == "parking-en")
    copies = [parking.model_copy(update={"key": key}) for key in keys]
    pack = clinic_pack.model_copy(update={"documents": copies})
    factory = activate_pack(pack)
    if by_vector:
        # "zzzz" is at right angles to both sections of both copies: four chunks, one distance.
        vector = HashEmbedder().embed(["zzzz"])[0]
        hits = _search(factory, pack.id, "zzzz", query_vector=vector, limit=20)
        expected = [key for key in sorted(keys) for _ in range(2)]
    else:
        # "ramp" is in the second section of each copy only: two chunks, one score.
        hits = _search(factory, pack.id, "ramp", query_vector=None, limit=20)
        expected = sorted(keys)
    assert [hit.document_key for hit in hits] == expected
