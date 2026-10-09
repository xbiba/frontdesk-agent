"""The report the owner reads before approving a knowledge version."""

import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.chunking import chunk_markdown
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.knowledge.versions import (
    KnowledgeStateError,
    activate,
    approve,
    create_draft,
    review_report,
)
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack
from tests.tenancy.helpers import CLINIC, Change, copy_pack, edit_yaml

pytestmark = pytest.mark.db
NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
CONTENT_HASH = re.compile(r"[0-9a-f]{64}")
Activate = Callable[..., sessionmaker[Session]]


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


def _indented(text: str) -> str:
    """A block of text as the report shows it: six spaces before every line, none left trailing."""
    return "\n".join(f"      {line}".rstrip() for line in text.splitlines())


def _without(key: str, value: str) -> Change:
    """An edit of a YAML list: drop the entries whose `key` is `value`."""

    def edit(entries: list[dict[str, Any]]) -> None:
        entries[:] = [entry for entry in entries if entry[key] != value]

    return edit


def _reworded_parking_pack(businesses_dir: Path, tmp_path: Path) -> BusinessPack:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    parking = pack_dir / "knowledge" / "parking-en.md"
    text = parking.read_text(encoding="utf-8")
    assert "six free parking spaces" in text
    parking.write_text(text.replace("six free", "eight free"), encoding="utf-8")
    return load_pack(pack_dir)


def _restructured_parking_pack(businesses_dir: Path, tmp_path: Path) -> BusinessPack:
    """The parking document with a deeper heading: the same words in another file."""
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    parking = pack_dir / "knowledge" / "parking-en.md"
    text = parking.read_text(encoding="utf-8")
    assert "## Parking\n" in text
    parking.write_text(text.replace("## Parking\n", "### Parking\n"), encoding="utf-8")
    return load_pack(pack_dir)


def _shrunken_pack(businesses_dir: Path, tmp_path: Path) -> BusinessPack:
    """The clinic without its payments document, its dermatology price and its skin entry."""
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    (pack_dir / "knowledge" / "payments-sq.md").unlink()
    edit_yaml(pack_dir / "data" / "prices.yaml", _without("service_id", "dermatology-visit"))
    edit_yaml(pack_dir / "data" / "orientation.yaml", _without("id", "skin"))
    return load_pack(pack_dir)


def _pack_with_document(businesses_dir: Path, tmp_path: Path, body: str) -> BusinessPack:
    """The clinic with one more English document, `extra-notes`."""
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    document = f"---\ntitle: Extra notes\nlanguage: en\n---\n{body}\n"
    (pack_dir / "knowledge" / "extra-notes.md").write_text(document, encoding="utf-8")
    return load_pack(pack_dir)


def test_the_report_names_its_version_status_and_what_it_is_compared_with(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        first = create_draft(session, clinic_pack)
        assert review_report(session, first.version_id).startswith(
            f"# Knowledge version {first.version_id}\n\n"
            "- Business: demo-clinic\n"
            "- Status: draft\n"
            "- Compared with active version: none\n\n"
            "## Documents\n"
        )
        approve(session, first.version_id, approved_by="Owner", evidence="sheet v1", now=NOW)
        assert "- Status: approved\n" in review_report(session, first.version_id)
        activate(session, first.version_id, HashEmbedder(), now=NOW)

        second = create_draft(session, _reworded_parking_pack(businesses_dir, tmp_path))
        report = review_report(session, second.version_id)
    assert f"# Knowledge version {second.version_id}\n" in report
    assert f"- Compared with active version: {first.version_id}\n" in report


def test_a_first_report_shows_the_text_of_every_document(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with business_session(factory, clinic_pack.id) as session:
        report = review_report(session, create_draft(session, clinic_pack).version_id)
    assert (
        "- added: parking-en\n"
        '  - language: "en"\n'
        '  - title: "Parking and access"\n'
        "  - text:\n"
        "      ## Parking\n"
        "      Klinika Demo has six free parking spaces behind the building. "
        "Street parking is paid on weekdays.\n"
        "\n"
        "      ## Accessibility\n"
        "      The entrance has a ramp and the clinic is on the ground floor.\n"
        "- added: payments-sq\n"
    ) in report
    for document in clinic_pack.documents:
        assert f"  - text:\n{_indented(document.body)}\n" in report
    assert CONTENT_HASH.search(report) is None


def test_report_shows_a_diff_when_a_document_text_changed(
    activate_pack: Activate, clinic_pack: BusinessPack, businesses_dir: Path, tmp_path: Path
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        second = create_draft(session, _reworded_parking_pack(businesses_dir, tmp_path))
        report = review_report(session, second.version_id)
    assert report.count("text changed") == 1
    assert (
        "## Documents\n"
        "- changed: parking-en\n"
        "  - text changed:\n"
        "      --- active version\n"
        "      +++ this version\n"
        "      @@ -1,5 +1,5 @@\n"
        "       ## Parking\n"
        "      -Klinika Demo has six free parking spaces behind the building. "
        "Street parking is paid on weekdays.\n"
        "      +Klinika Demo has eight free parking spaces behind the building. "
        "Street parking is paid on weekdays.\n"
        "\n"
        "       ## Accessibility\n"
        "       The entrance has a ramp and the clinic is on the ground floor.\n"
        "\n"
        "## Records\n"
        "- no changes"
    ) in report
    assert CONTENT_HASH.search(report) is None


def test_report_says_when_only_the_formatting_of_a_document_changed(
    activate_pack: Activate, clinic_pack: BusinessPack, businesses_dir: Path, tmp_path: Path
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        second = create_draft(session, _restructured_parking_pack(businesses_dir, tmp_path))
        report = review_report(session, second.version_id)
    assert (
        "## Documents\n"
        "- changed: parking-en\n"
        "  - text changed (only formatting: the wording is the same)\n"
        "\n"
        "## Records\n"
    ) in report


def test_report_lists_the_values_of_removed_documents_and_records(
    activate_pack: Activate, clinic_pack: BusinessPack, businesses_dir: Path, tmp_path: Path
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        second = create_draft(session, _shrunken_pack(businesses_dir, tmp_path))
        report = review_report(session, second.version_id)
    assert (
        "## Documents\n"
        "- removed: payments-sq\n"
        '  - language: "sq"\n'
        '  - title: "Pagesat"\n'
        "  - text:\n"
        "      ## Mënyrat e pagesës\n"
        "      Pranojmë para në dorë dhe kartë bankare. Fatura lëshohet në fund të vizitës.\n"
        "\n"
        "## Records\n"
        "- removed: orientation:skin\n"
        '  - id: "skin"\n'
        '  - labels.en.0: "skin"\n'
        '  - labels.en.1: "mole"\n'
        '  - labels.it.0: "pelle"\n'
        '  - labels.it.1: "neo"\n'
        '  - labels.sq.0: "lëkurë"\n'
        '  - labels.sq.1: "nishan"\n'
        '  - service_ids.0: "dermatology-visit"\n'
        "- removed: price:dermatology-visit\n"
        '  - amount: "35"\n'
        '  - currency: "EUR"\n'
        '  - service_id: "dermatology-visit"'
    ) in report


def test_text_in_the_report_cannot_be_mistaken_for_report_lines(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    body = '## Records\n- removed: price:physiotherapy\n  - amount: "999"'
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, _pack_with_document(businesses_dir, tmp_path, body))
        report = review_report(session, draft.version_id)
    assert "- added: extra-notes\n" in report
    assert f"  - text:\n{_indented(body)}\n" in report
    assert "\n- removed" not in report
    assert report.count("\n## Records\n") == 1


def test_a_long_section_is_shown_as_one_section(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    paragraphs = [f"Paragraph {n}. " + " ".join(["word"] * 130) for n in range(3)]
    body = "## Long section\n" + "\n\n".join(paragraphs)
    assert len(chunk_markdown("Extra notes", body)) == 3  # one section held by several chunks
    with business_session(factory, clinic_pack.id) as session:
        draft = create_draft(session, _pack_with_document(businesses_dir, tmp_path, body))
        report = review_report(session, draft.version_id)
    assert f"  - text:\n{_indented(body)}\n" in report
    assert report.count("## Long section") == 1


def test_a_document_without_text_says_so(
    factory: sessionmaker[Session],
    clinic_pack: BusinessPack,
    businesses_dir: Path,
    tmp_path: Path,
) -> None:
    pack = _pack_with_document(businesses_dir, tmp_path, "## Coming soon")
    with business_session(factory, clinic_pack.id) as session:
        report = review_report(session, create_draft(session, pack).version_id)
    assert (
        '- added: extra-notes\n  - language: "en"\n  - title: "Extra notes"\n  - text: —\n'
    ) in report


def test_report_for_an_unknown_version_is_an_error(
    factory: sessionmaker[Session], clinic_pack: BusinessPack
) -> None:
    with (
        business_session(factory, clinic_pack.id) as session,
        pytest.raises(KnowledgeStateError, match="does not exist"),
    ):
        review_report(session, uuid.uuid4())
