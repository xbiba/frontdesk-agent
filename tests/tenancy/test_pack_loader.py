import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from frontdesk.tenancy.loader import PackError, load_pack
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, Profile

PROMISES = (
    "contatter",
    "ti risponder",
    "passo la conversazione",
    "vedrà il tuo messaggio",
    "will contact",
    "will reply",
    "will see your message",
    "passing this conversation",
    "do t'ju kontakt",
    "po ia kaloj",
    "do ta shohë",
)


def _copy(businesses_dir: Path, name: str, tmp_path: Path) -> Path:
    target = tmp_path / name
    shutil.copytree(businesses_dir / name, target)
    return target


def _edit(path: Path, change: Callable[[Any], None]) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_demo_clinic_loads(clinic_pack: BusinessPack) -> None:
    assert clinic_pack.profile is Profile.RECEPTIONIST
    assert clinic_pack.features.orientation is True
    assert clinic_pack.integrations.calendar is None
    assert {"orthopedics-visit", "physiotherapy"} <= {s.id for s in clinic_pack.services}
    assert {d.key for d in clinic_pack.documents} >= {"parking-en", "physiotherapy-sq"}


def test_demo_shop_has_no_calendar_and_no_clinic_data(shop_pack: BusinessPack) -> None:
    assert shop_pack.profile is Profile.SUPPORT_SALES
    assert shop_pack.integrations.calendar is None
    assert shop_pack.services == []
    assert shop_pack.orientation == []
    assert [o.reference for o in shop_pack.orders] == ["ORD-10427", "ORD-10458"]


@pytest.mark.parametrize("name", ["demo-clinic", "demo-shop"])
def test_demo_texts_promise_nothing_and_handoff_gives_a_contact(
    businesses_dir: Path, name: str
) -> None:
    pack = load_pack(businesses_dir / name)
    for language, texts in pack.fixed_texts.items():
        assert pack.location.phone in texts[FixedTextKey.HANDOFF], language
        for key, text in texts.items():
            assert not any(promise in text.lower() for promise in PROMISES), (language, key)


def test_opening_days_use_day_codes(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = _copy(businesses_dir, "demo-shop", tmp_path)
    _edit(
        pack_dir / "data" / "location.yaml",
        lambda data: data["opening_hours"][0].update(days="weekdays"),
    )
    with pytest.raises(PackError, match="days"):
        load_pack(pack_dir)


def test_missing_fixed_text_names_business_language_and_key(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = _copy(businesses_dir, "demo-clinic", tmp_path)
    _edit(pack_dir / "business.yaml", lambda data: data["fixed_texts"]["it"].pop("handoff"))
    with pytest.raises(
        PackError, match=r"demo-clinic.*missing fixed text 'handoff' for language 'it'"
    ):
        load_pack(pack_dir)


def test_unknown_service_reference_is_rejected(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = _copy(businesses_dir, "demo-clinic", tmp_path)
    _edit(
        pack_dir / "data" / "doctors.yaml", lambda data: data[0]["service_ids"].append("cardiology")
    )
    with pytest.raises(PackError, match="unknown service 'cardiology'"):
        load_pack(pack_dir)


def test_orientation_needs_the_receptionist_profile(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = _copy(businesses_dir, "demo-shop", tmp_path)
    _edit(pack_dir / "business.yaml", lambda data: data["features"].update(orientation=True))
    with pytest.raises(PackError, match="feature 'orientation'"):
        load_pack(pack_dir)


def test_calendar_integration_needs_the_receptionist_profile(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = _copy(businesses_dir, "demo-shop", tmp_path)
    _edit(
        pack_dir / "business.yaml",
        lambda data: data.update(
            integrations={"calendar": {"credentials_ref": "secret/x", "calendars": {"a": "b"}}}
        ),
    )
    with pytest.raises(PackError, match="calendar integration needs profile 'receptionist'"):
        load_pack(pack_dir)


def test_knowledge_document_needs_front_matter(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = _copy(businesses_dir, "demo-shop", tmp_path)
    (pack_dir / "knowledge" / "broken.md").write_text("# No front matter\n", encoding="utf-8")
    with pytest.raises(PackError, match="front matter"):
        load_pack(pack_dir)


def test_missing_directory_is_a_pack_error(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="not found"):
        load_pack(tmp_path / "nowhere")
