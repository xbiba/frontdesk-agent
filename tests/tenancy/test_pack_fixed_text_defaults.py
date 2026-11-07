"""The fixed texts a pack may leave out and the engine gives (2026-11-07): what a failure says
(`unavailable`) and the offer of a person's contact (`contact_offer`), in each of its languages.
A pack's own text wins; a blank one is refused like any other."""

from pathlib import Path

import pytest

from frontdesk.tenancy.loader import PackError, load_pack
from frontdesk.tenancy.pack import DEFAULT_FIXED_TEXTS, FixedTextKey
from tests.tenancy.helpers import CLINIC, copy_pack, edit_yaml

DEFAULTED = (FixedTextKey.UNAVAILABLE, FixedTextKey.CONTACT_OFFER)


@pytest.mark.parametrize("key", DEFAULTED)
def test_a_pack_without_the_text_gets_the_engine_s_in_each_language(
    tmp_path: Path, businesses_dir: Path, key: FixedTextKey
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(
        pack_dir / "business.yaml",
        lambda data: [texts.pop(key.value, None) for texts in data["fixed_texts"].values()],
    )
    pack = load_pack(pack_dir)
    for language in pack.languages:
        assert pack.fixed_texts[language][key] == DEFAULT_FIXED_TEXTS[key][language]


@pytest.mark.parametrize("key", DEFAULTED)
def test_the_pack_s_own_text_wins(tmp_path: Path, businesses_dir: Path, key: FixedTextKey) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(
        pack_dir / "business.yaml",
        lambda data: data["fixed_texts"]["it"].update({key.value: "Nostro."}),
    )
    assert load_pack(pack_dir).fixed_texts["it"][key] == "Nostro."


@pytest.mark.parametrize("key", DEFAULTED)
def test_a_blank_text_is_refused(tmp_path: Path, businesses_dir: Path, key: FixedTextKey) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(
        pack_dir / "business.yaml", lambda data: data["fixed_texts"]["it"].update({key.value: "  "})
    )
    with pytest.raises(PackError, match=f"missing fixed text '{key.value}' for language 'it'"):
        load_pack(pack_dir)


def test_the_defaults_give_no_contact() -> None:
    """The offer only offers: the number comes when the person asks for it."""
    for texts in DEFAULT_FIXED_TEXTS.values():
        for text in texts.values():
            assert not any(char.isdigit() for char in text)
