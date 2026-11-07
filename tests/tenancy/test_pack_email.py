"""The `email:` section of a pack (email drafts, 2026-11-07): the mailbox, the signature, the words
of the placeholder, and the tester mode that drafts only for the senders it lists."""

from pathlib import Path

import pytest

from frontdesk.tenancy.loader import PackError, load_pack
from tests.tenancy.helpers import CLINIC, copy_pack, edit_yaml


def test_a_pack_has_no_email_section_by_default(businesses_dir: Path) -> None:
    assert load_pack(businesses_dir / CLINIC).email is None


def test_the_email_section_is_read_with_its_defaults(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(
        pack_dir / "business.yaml",
        lambda d: d.update(
            email={"mailbox": "Info@Example.com", "testers": ["Tester@Example.com"]}
        ),
    )
    email = load_pack(pack_dir).email
    assert email is not None
    assert email.mailbox == "info@example.com"
    assert email.signature == "[FIRMA]"
    assert email.todo == "DA COMPLETARE"
    assert email.testers_only is True
    assert email.drafts_for("Tester <tester@example.com>") is True
    assert email.drafts_for("someone@example.com") is False


def test_open_to_everyone_drafts_for_any_sender(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(
        pack_dir / "business.yaml",
        lambda d: d.update(email={"mailbox": "info@example.com", "testers_only": False}),
    )
    email = load_pack(pack_dir).email
    assert email is not None
    assert email.drafts_for("someone@example.com") is True


@pytest.mark.parametrize(
    "section",
    [
        {"mailbox": "not-an-address"},
        {"mailbox": "info@example.com", "testers": ["nobody"]},
        {"mailbox": "info@example.com", "signature": " "},
        {"mailbox": "info@example.com", "unknown": 1},
    ],
)
def test_a_bad_email_section_is_refused(
    businesses_dir: Path, tmp_path: Path, section: dict[str, object]
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda d: d.update(email=section))
    with pytest.raises(PackError, match="email"):
        load_pack(pack_dir)
