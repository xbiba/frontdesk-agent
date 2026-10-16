"""The knowledge in the prompt (Albi, 2026-10-16): a small pack's approved documents go whole to the
reception agent, in the conversation's language, as a search would give them."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session
from frontdesk.knowledge.in_prompt import documents_for, knowledge_chars
from frontdesk.knowledge.retrieval import valid_chunks
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack

pytestmark = pytest.mark.db

Activate = Callable[..., sessionmaker[Session]]
TODAY = date(2026, 10, 20)


def test_the_size_counts_every_document_in_every_language(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        chunks = valid_chunks(session, business_id=clinic_pack.id, today=TODAY)
        expected = sum(len(chunk.heading) + len(chunk.content) for chunk in chunks)
        assert {chunk.language for chunk in chunks} == {"sq", "it", "en"}
        assert knowledge_chars(session, clinic_pack.id) == expected > 0


def test_a_business_without_knowledge_has_none_in_the_prompt(
    activate_pack: Activate, clinic_pack: BusinessPack, app_url: str
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, "no-such-business") as session:
        assert knowledge_chars(session, "no-such-business") == 0
        assert documents_for(session, "no-such-business", TODAY, "it").text == ""


def test_the_documents_come_in_the_conversation_s_language(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        italian = documents_for(session, clinic_pack.id, TODAY, "it")
        english = documents_for(session, clinic_pack.id, TODAY, "en")
    assert italian.sources == ("first-visit-it", "physiotherapy-it")
    assert english.sources == ("parking-en",)
    assert english.text.startswith("[Parking and access \u203a Parking]\nKlinika Demo has six")


def test_a_language_with_no_valid_document_gets_every_language(
    activate_pack: Activate, businesses_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "demo-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", target)
    parking = target / "knowledge" / "parking-en.md"
    text = parking.read_text(encoding="utf-8")
    parking.write_text(
        text.replace("language: en\n", "language: en\nvalid_until: 2026-10-19\n"), encoding="utf-8"
    )
    pack = load_pack(target)
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        found = documents_for(session, pack.id, TODAY, "en")
    assert "parking-en" not in found.sources  # out of its period
    assert set(found.sources) == {
        "first-visit-it",
        "payments-sq",
        "physiotherapy-it",
        "physiotherapy-sq",
    }
