import shutil
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.db.models import Chunk, Document, KnowledgeVersion, Record
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import Embedder, HashEmbedder
from frontdesk.knowledge.versions import (
    EmbedderMismatchError,
    KnowledgeStateError,
    VersionStatus,
    activate,
    active_version_id,
    approve,
    check_embedder,
    create_draft,
    reembed_active,
    review_report,
    rollback,
    version_status,
)
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


class OtherEmbedder(HashEmbedder):
    identity = "other-v1"


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


def _status(session: Session, version_id: object) -> str:
    return session.scalar(select(KnowledgeVersion.status).where(KnowledgeVersion.id == version_id))


def _changed_price_pack(businesses_dir: Path, tmp_path: Path) -> BusinessPack:
    target = tmp_path / "demo-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", target)
    prices_path = target / "data" / "prices.yaml"
    prices = yaml.safe_load(prices_path.read_text(encoding="utf-8"))
    for price in prices:
        if price["service_id"] == "physiotherapy":
            price["amount"] = "32"
    prices_path.write_text(yaml.safe_dump(prices, allow_unicode=True), encoding="utf-8")
    return load_pack(target)


def _expiring_parking_pack(businesses_dir: Path, tmp_path: Path) -> BusinessPack:
    target = tmp_path / "demo-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", target)
    parking = target / "knowledge" / "parking-en.md"
    text = parking.read_text(encoding="utf-8")
    parking.write_text(
        text.replace("language: en\n", "language: en\nvalid_until: 2027-01-01\n"), encoding="utf-8"
    )
    return load_pack(target)


def test_a_draft_needs_approval_before_activation(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        assert draft.unchanged is False
        assert version_status(session, draft.version_id) is VersionStatus.DRAFT
        assert active_version_id(session, clinic_pack.id) is None
        with pytest.raises(KnowledgeStateError, match="not approved"):
            activate(session, draft.version_id, HashEmbedder(), now=NOW)


def test_approval_needs_who_and_evidence(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        with pytest.raises(KnowledgeStateError, match="who approved"):
            approve(session, draft.version_id, approved_by=" ", evidence="email", now=NOW)


def test_activation_embeds_every_chunk(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        approve(session, draft.version_id, approved_by="Owner", evidence="signed sheet", now=NOW)
        activate(session, draft.version_id, HashEmbedder(), now=NOW)
        assert active_version_id(session, clinic_pack.id) == draft.version_id
        missing = session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.knowledge_version_id == draft.version_id, Chunk.embedding.is_(None))
        )
        assert missing == 0


def test_unchanged_pack_reuses_the_existing_version(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = create_draft(session, clinic_pack)
        second = create_draft(session, clinic_pack)
    assert second.unchanged is True
    assert second.version_id == first.version_id


def test_new_version_replaces_the_old_and_rollback_restores_it(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    changed = _changed_price_pack(businesses_dir, tmp_path)
    with business_session(factory, clinic_pack.id) as session:
        first = create_draft(session, clinic_pack)
        approve(session, first.version_id, approved_by="Owner", evidence="sheet v1", now=NOW)
        activate(session, first.version_id, HashEmbedder(), now=NOW)

        second = create_draft(session, changed)
        report = review_report(session, second.version_id)
        assert '- changed: price:physiotherapy\n  - amount: "30" → "32"' in report
        assert "## Documents\n- no changes" in report

        approve(session, second.version_id, approved_by="Owner", evidence="sheet v2", now=NOW)
        activate(session, second.version_id, HashEmbedder(), now=NOW)
        assert active_version_id(session, clinic_pack.id) == second.version_id
        assert _status(session, first.version_id) == "retired"

        restored = rollback(session, clinic_pack.id, HashEmbedder(), now=NOW)
        assert restored == first.version_id
        assert active_version_id(session, clinic_pack.id) == first.version_id
        assert _status(session, second.version_id) == "retired"


def test_report_shows_document_metadata_changes(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = create_draft(session, clinic_pack)
        approve(session, first.version_id, approved_by="Owner", evidence="sheet v1", now=NOW)
        activate(session, first.version_id, HashEmbedder(), now=NOW)
        second = create_draft(session, _expiring_parking_pack(businesses_dir, tmp_path))
        report = review_report(session, second.version_id)
    assert '- changed: parking-en\n  - valid_until: — → "2027-01-01"' in report
    assert "text changed" not in report


def test_first_report_lists_every_value(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        report = review_report(session, create_draft(session, clinic_pack).version_id)
    assert '- added: price:physiotherapy\n  - amount: "30"' in report
    assert '  - conditions.it: "a seduta"' in report
    assert '- added: parking-en\n  - language: "en"' in report


def test_activation_records_the_embedder_and_refuses_another(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        approve(session, draft.version_id, approved_by="Owner", evidence="signed sheet", now=NOW)
        activate(session, draft.version_id, HashEmbedder(), now=NOW)
        check_embedder(session, clinic_pack.id, HashEmbedder())
        with pytest.raises(EmbedderMismatchError, match="reembed"):
            check_embedder(session, clinic_pack.id, OtherEmbedder())
        assert reembed_active(session, clinic_pack.id, OtherEmbedder()) > 0
        check_embedder(session, clinic_pack.id, OtherEmbedder())


def test_rollback_re_embeds_with_the_current_embedder(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = create_draft(session, clinic_pack)
        approve(session, first.version_id, approved_by="Owner", evidence="sheet v1", now=NOW)
        activate(session, first.version_id, HashEmbedder(), now=NOW)
        second = create_draft(session, _changed_price_pack(businesses_dir, tmp_path))
        approve(session, second.version_id, approved_by="Owner", evidence="sheet v2", now=NOW)
        activate(session, second.version_id, OtherEmbedder(), now=NOW)
        rollback(session, clinic_pack.id, OtherEmbedder(), now=NOW)
        check_embedder(session, clinic_pack.id, OtherEmbedder())


def test_rollback_without_history_is_an_error(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with (
        business_session(factory, clinic_pack.id) as session,
        pytest.raises(KnowledgeStateError, match="no previous approved version"),
    ):
        rollback(session, clinic_pack.id, HashEmbedder(), now=NOW)


# Further behaviour: what each change stores and records, what a refused or failed call leaves
# alone, how vectors are recomputed, and what the review report says.


class NegatedEmbedder(HashEmbedder):
    """Another model with vectors of its own: OtherEmbedder only renames the hash vectors."""

    identity = "negated-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[-value for value in vector] for vector in super().embed(texts)]


class WrongSizeEmbedder:
    dimension = 2
    identity = "wrong-size-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class InjectedError(RuntimeError):
    """A failure injected between two steps of an operation."""


HOUR = timedelta(hours=1)


def _activate(
    session: Session, pack: BusinessPack, *, embedder: Embedder | None = None, now: datetime = NOW
) -> uuid.UUID:
    draft = create_draft(session, pack)
    approve(session, draft.version_id, approved_by="Owner", evidence="signed sheet", now=now)
    activate(session, draft.version_id, embedder or HashEmbedder(), now=now)
    return draft.version_id


def _chunk_count(session: Session, version_id: uuid.UUID, *, unembedded: bool = False) -> int:
    query = select(func.count()).select_from(Chunk).where(Chunk.knowledge_version_id == version_id)
    if unembedded:
        query = query.where(Chunk.embedding.is_(None))
    return session.scalar(query) or 0


def _vectors(session: Session, version_id: uuid.UUID) -> list[list[float]]:
    """The vectors of a version's chunks, in an order that does not change."""
    rows = session.scalars(
        select(Chunk.embedding).where(Chunk.knowledge_version_id == version_id).order_by(Chunk.id)
    )
    return [list(row) for row in rows]


def _negated(vectors: list[list[float]]) -> list[list[float]]:
    return [[-value for value in vector] for vector in vectors]


def _committed(
    factory: sessionmaker[Session], business_id: str, version_id: uuid.UUID
) -> tuple[str, str | None, int, int]:
    """Status, embedder, chunks and chunks without a vector, as a new transaction sees them."""
    with business_session(factory, business_id) as session:
        row = session.execute(
            select(KnowledgeVersion.status, KnowledgeVersion.embedder).where(
                KnowledgeVersion.id == version_id
            )
        ).one()
        return (
            row.status,
            row.embedder,
            _chunk_count(session, version_id),
            _chunk_count(session, version_id, unembedded=True),
        )


def _approval(session: Session, version_id: uuid.UUID) -> tuple[Any, ...]:
    """Who approved a version, on what evidence and when, as stored."""
    row = session.execute(
        select(
            KnowledgeVersion.approved_by,
            KnowledgeVersion.approval_evidence,
            KnowledgeVersion.approved_at,
        ).where(KnowledgeVersion.id == version_id)
    ).one()
    return tuple(row)


def _dates(session: Session, version_id: uuid.UUID) -> tuple[Any, ...]:
    """When a version was activated and when it was retired, as stored."""
    row = session.execute(
        select(KnowledgeVersion.activated_at, KnowledgeVersion.retired_at).where(
            KnowledgeVersion.id == version_id
        )
    ).one()
    return tuple(row)


def test_a_draft_stores_every_document_chunk_and_record_of_the_pack(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        version_id = create_draft(session, clinic_pack).version_id
        documents = set(
            session.scalars(
                select(Document.doc_key).where(Document.knowledge_version_id == version_id)
            )
        )
        chunks = _chunk_count(session, version_id)
        records = {
            (kind, key)
            for kind, key in session.execute(
                select(Record.kind, Record.record_key).where(
                    Record.knowledge_version_id == version_id
                )
            )
        }
    expected_chunks = sum(
        len(chunk_markdown(document.title, document.body)) for document in clinic_pack.documents
    )
    assert expected_chunks > len(clinic_pack.documents)  # some documents have several sections
    assert documents == {document.key for document in clinic_pack.documents}
    assert chunks == expected_chunks
    assert records == (
        {("location", "main")}
        | {("service", service.id) for service in clinic_pack.services}
        | {("doctor", doctor.id) for doctor in clinic_pack.doctors}
        | {("price", price.service_id) for price in clinic_pack.prices}
        | {("orientation", entry.id) for entry in clinic_pack.orientation}
        | {("orientation_map", "approval")}
    )


def test_approval_needs_evidence_as_well(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        with pytest.raises(KnowledgeStateError, match="evidence of approval"):
            approve(session, draft.version_id, approved_by="Owner", evidence="  ", now=NOW)
        assert version_status(session, draft.version_id) is VersionStatus.DRAFT


def test_a_version_is_approved_only_once(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        approve(session, draft.version_id, approved_by=" Owner ", evidence=" sheet v1 ", now=NOW)
        assert _approval(session, draft.version_id) == ("Owner", "sheet v1", NOW)
        with pytest.raises(KnowledgeStateError, match="is not draft"):
            approve(
                session,
                draft.version_id,
                approved_by="Other",
                evidence="sheet v2",
                now=NOW + timedelta(hours=1),
            )
        assert _approval(session, draft.version_id) == ("Owner", "sheet v1", NOW)


def test_activation_retirement_and_rollback_are_dated(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    hour = timedelta(hours=1)
    with business_session(factory, clinic_pack.id) as session:
        first = _activate(session, clinic_pack, now=NOW + hour)
        second = _activate(
            session, _changed_price_pack(businesses_dir, tmp_path), now=NOW + 2 * hour
        )
        assert _dates(session, first) == (NOW + hour, NOW + 2 * hour)
        assert _dates(session, second) == (NOW + 2 * hour, None)

        rollback(session, clinic_pack.id, HashEmbedder(), now=NOW + 3 * hour)
        assert _dates(session, first) == (NOW + 3 * hour, None)
        assert _dates(session, second) == (NOW + 2 * hour, NOW + 3 * hour)


def test_nothing_is_embedded_before_activation(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
        chunks = _chunk_count(session, draft.version_id)
        assert chunks > 0
        with pytest.raises(KnowledgeStateError, match="not approved"):
            activate(session, draft.version_id, HashEmbedder(), now=NOW)
        approve(session, draft.version_id, approved_by="Owner", evidence="signed sheet", now=NOW)
        assert _chunk_count(session, draft.version_id, unembedded=True) == chunks


def test_a_failed_activation_changes_nothing(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = _activate(session, clinic_pack)
        second = create_draft(session, _changed_price_pack(businesses_dir, tmp_path))
        approve(session, second.version_id, approved_by="Owner", evidence="sheet v2", now=NOW)
        with pytest.raises(KnowledgeStateError, match="embedding dimension"):
            activate(session, second.version_id, WrongSizeEmbedder(), now=NOW)
        assert active_version_id(session, clinic_pack.id) == first
        assert version_status(session, second.version_id) is VersionStatus.APPROVED


def test_an_activation_that_fails_after_retiring_the_old_version_changes_nothing(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = _activate(session, clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        second = create_draft(session, _changed_price_pack(businesses_dir, tmp_path)).version_id
        approve(session, second, approved_by="Owner", evidence="sheet v2", now=NOW)
    seen: list[VersionStatus | None] = []

    def fail_after_the_retirement(session: Session, *args: Any, **kwargs: Any) -> None:
        seen.append(version_status(session, first))
        raise InjectedError

    monkeypatch.setattr("frontdesk.knowledge.versions._transition", fail_after_the_retirement)
    with pytest.raises(InjectedError), business_session(factory, clinic_pack.id) as session:
        activate(session, second, HashEmbedder(), now=NOW + HOUR)

    assert seen == [VersionStatus.RETIRED]  # the failure came after the old version was retired
    assert _committed(factory, clinic_pack.id, first)[0] == "active"
    status, embedder, chunks, unembedded = _committed(factory, clinic_pack.id, second)
    assert (status, embedder) == ("approved", None)
    assert chunks > 0
    assert unembedded == chunks


def test_a_rollback_that_fails_leaves_the_active_version_and_the_candidate_as_they_were(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = _activate(session, clinic_pack, now=NOW)
    with business_session(factory, clinic_pack.id) as session:
        second = _activate(session, _changed_price_pack(businesses_dir, tmp_path), now=NOW + HOUR)
    with (
        pytest.raises(KnowledgeStateError, match="embedding dimension"),
        business_session(factory, clinic_pack.id) as session,
    ):
        rollback(session, clinic_pack.id, WrongSizeEmbedder(), now=NOW + 2 * HOUR)

    assert _committed(factory, clinic_pack.id, second)[0] == "active"
    status, embedder, chunks, unembedded = _committed(factory, clinic_pack.id, first)
    assert (status, embedder, unembedded) == ("retired", "hash-v1", 0)
    assert chunks > 0


def test_a_reembedding_that_fails_leaves_the_vectors_in_place(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        version_id = _activate(session, clinic_pack)
    with (
        pytest.raises(KnowledgeStateError, match="embedding dimension"),
        business_session(factory, clinic_pack.id) as session,
    ):
        reembed_active(session, clinic_pack.id, WrongSizeEmbedder())

    status, embedder, chunks, unembedded = _committed(factory, clinic_pack.id, version_id)
    assert (status, embedder, unembedded) == ("active", "hash-v1", 0)
    assert chunks > 0


def test_another_business_cannot_see_or_change_a_version(
    factory: sessionmaker[Session], clinic_pack: BusinessPack, shop_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, clinic_pack)
    with business_session(factory, shop_pack.id) as session:
        with pytest.raises(KnowledgeStateError, match="is not draft"):
            approve(session, draft.version_id, approved_by="Owner", evidence="sheet", now=NOW)
        with pytest.raises(KnowledgeStateError, match="does not exist"):
            review_report(session, draft.version_id)
    with (
        pytest.raises(DBAPIError, match="row-level security"),
        business_session(factory, shop_pack.id) as session,
    ):
        create_draft(session, clinic_pack)

    with business_session(factory, clinic_pack.id) as session:
        assert version_status(session, draft.version_id) is VersionStatus.DRAFT


def test_an_unchanged_pack_matches_a_live_version_but_not_a_retired_one(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    changed = _changed_price_pack(businesses_dir, tmp_path)
    with business_session(factory, clinic_pack.id) as session:
        first = _activate(session, clinic_pack)
        same = create_draft(session, clinic_pack)
        assert (same.unchanged, same.version_id) == (True, first)

        second = _activate(session, changed)
        # The first content comes back after its version was retired: that needs a new approval.
        again = create_draft(session, clinic_pack)
        assert again.unchanged is False
        assert again.version_id not in {first, second}


def test_rollback_restores_the_most_recently_retired_version(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    packs = [
        clinic_pack,
        _changed_price_pack(businesses_dir, tmp_path / "price"),
        _expiring_parking_pack(businesses_dir, tmp_path / "parking"),
    ]
    with business_session(factory, clinic_pack.id) as session:
        first, second, third = [
            _activate(session, pack, now=NOW + timedelta(hours=hour))
            for hour, pack in enumerate(packs)
        ]
        restored = rollback(session, clinic_pack.id, HashEmbedder(), now=NOW + timedelta(hours=3))
        assert restored == second
        assert active_version_id(session, clinic_pack.id) == second
        assert version_status(session, first) is VersionStatus.RETIRED
        assert version_status(session, third) is VersionStatus.RETIRED


def test_reembedding_recomputes_every_vector(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        version_id = _activate(session, clinic_pack)
        before = _vectors(session, version_id)
        assert reembed_active(session, clinic_pack.id, NegatedEmbedder()) == len(before) > 0
        assert _vectors(session, version_id) == _negated(before)


def test_a_pack_without_documents_has_no_chunks_to_embed(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    records_only = clinic_pack.model_copy(update={"documents": []})
    with business_session(factory, clinic_pack.id) as session:
        version_id = _activate(session, records_only)
        assert active_version_id(session, clinic_pack.id) == version_id
        assert _chunk_count(session, version_id) == 0
        assert reembed_active(session, clinic_pack.id, NegatedEmbedder()) == 0


def test_rollback_recomputes_the_vectors_of_the_restored_version(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = _activate(session, clinic_pack)
        before = _vectors(session, first)
        assert before
        _activate(
            session, _changed_price_pack(businesses_dir, tmp_path), embedder=NegatedEmbedder()
        )
        assert rollback(session, clinic_pack.id, NegatedEmbedder(), now=NOW) == first
        assert _vectors(session, first) == _negated(before)


def test_a_business_without_knowledge_has_nothing_to_mismatch_or_reembed(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        check_embedder(session, clinic_pack.id, HashEmbedder())
        with pytest.raises(KnowledgeStateError, match="no active knowledge version"):
            reembed_active(session, clinic_pack.id, HashEmbedder())


@pytest.mark.parametrize(
    ("embedder", "identity"),
    [
        pytest.param(None, "hash-v1", id="default-embedder"),
        pytest.param(OtherEmbedder(), "other-v1", id="given-embedder"),
    ],
)
def test_activate_pack_activates_a_pack_for_other_tests(
    activate_pack: Callable[..., sessionmaker[Session]],
    clinic_pack: BusinessPack,
    embedder: Embedder | None,
    identity: str,
) -> None:
    app_factory = activate_pack(clinic_pack, embedder)
    with business_session(app_factory, clinic_pack.id) as session:
        version_id = active_version_id(session, clinic_pack.id)
        assert version_id is not None
        embedded_by = session.scalar(
            select(KnowledgeVersion.embedder).where(KnowledgeVersion.id == version_id)
        )
        assert embedded_by == identity
        assert _chunk_count(session, version_id) > 0
        assert _chunk_count(session, version_id, unembedded=True) == 0


def test_the_approval_report_names_the_orientation_map_version_and_its_doctor(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        report = review_report(session, create_draft(session, clinic_pack).version_id)
    lines = report.splitlines()
    start = lines.index("- added: orientation_map:approval")
    assert lines[start + 1 : start + 4] == [
        '  - approved_by: "Dr. Arben Demo (synthetic)"',
        '  - approved_on: "2026-10-01"',
        '  - version: "demo-2026-10"',
    ]


@pytest.mark.parametrize(
    ("field", "before", "after"),
    [
        ("version", "demo-2026-10", "demo-2026-11"),
        ("approved_by", "Dr. Arben Demo (synthetic)", "Dr. Elira Demo (synthetic)"),
    ],
)
def test_a_new_map_version_or_approving_doctor_is_a_new_draft_and_shows_in_the_report(
    factory: sessionmaker[Session], clinic_pack: BusinessPack, field: str, before: str, after: str
) -> None:
    """The map's approval is part of the knowledge version: changing it needs a new approval."""
    approval = clinic_pack.orientation_map
    assert approval is not None
    changed = clinic_pack.model_copy(
        update={"orientation_map": approval.model_copy(update={field: after})}
    )
    with business_session(factory, clinic_pack.id) as session:
        first = create_draft(session, clinic_pack)
        approve(session, first.version_id, approved_by="Owner", evidence="sheet v1", now=NOW)
        activate(session, first.version_id, HashEmbedder(), now=NOW)

        second = create_draft(session, changed)
        report = review_report(session, second.version_id)
    assert second.unchanged is False
    assert second.version_id != first.version_id
    assert f'- changed: orientation_map:approval\n  - {field}: "{before}" → "{after}"' in report
