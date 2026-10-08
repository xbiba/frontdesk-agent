import shutil
import uuid
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import EMBEDDING_DIM
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.evidence import EvidenceKind
from frontdesk.knowledge.retrieval import (
    RetrievedChunk,
    build_tsquery,
    chunk_evidence,
    hybrid_search,
    rrf_fuse,
)
from frontdesk.knowledge.versions import create_draft
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

TODAY = date(2026, 10, 7)
Activate = Callable[..., sessionmaker[Session]]


def test_rrf_prefers_items_ranked_by_both_lists() -> None:
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    fused = rrf_fuse([[a, b], [b, c]])
    assert fused[0][0] == b
    assert {item for item, _ in fused} == {a, b, c}


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


@pytest.mark.db
def test_only_the_active_version_is_searched(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    parking = next(doc for doc in clinic_pack.documents if doc.key == "parking-en")
    body = parking.body.replace("six free parking spaces", "a rooftop garage")
    edited = _with_parking(clinic_pack, body=body)
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


@pytest.mark.db
@pytest.mark.parametrize(
    ("window", "searchable"),
    [
        ({"valid_from": TODAY + timedelta(days=1)}, False),
        ({"valid_from": TODAY}, True),
        ({"valid_until": TODAY}, True),
    ],
    ids=["starts tomorrow", "starts today", "ends today"],
)
def test_documents_are_searchable_from_their_first_to_their_last_day(
    activate_pack: Activate, clinic_pack: BusinessPack, window: dict[str, date], searchable: bool
) -> None:
    pack = _with_parking(clinic_pack, **window)
    factory = activate_pack(pack)
    hits = _search(factory, pack.id, "parking", query_vector=None)
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
