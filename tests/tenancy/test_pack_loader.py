import re
from pathlib import Path

import pytest

from frontdesk.safety.rules import promise_phrase
from frontdesk.tenancy.loader import PackError, load_pack
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, Profile
from tests.tenancy.helpers import CLINIC, SHOP, copy_pack, edit_yaml

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
            assert promise_phrase(text) is None, (language, key)


@pytest.mark.parametrize("name", ["demo-clinic", "demo-shop"])
def test_demo_documents_promise_nothing_the_channel_cannot_do(
    businesses_dir: Path, name: str
) -> None:
    """A reply quotes its document: a promise in the document becomes the bot's (P96)."""
    for document in load_pack(businesses_dir / name).documents:
        assert promise_phrase(document.body) is None, document.key


def test_opening_days_use_day_codes(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, "demo-shop", tmp_path)
    edit_yaml(
        pack_dir / "data" / "location.yaml",
        lambda data: data["opening_hours"][0].update(days="weekdays"),
    )
    with pytest.raises(PackError, match="days"):
        load_pack(pack_dir)


def test_missing_fixed_text_names_business_language_and_key(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, "demo-clinic", tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda data: data["fixed_texts"]["it"].pop("handoff"))
    with pytest.raises(
        PackError, match=r"demo-clinic.*missing fixed text 'handoff' for language 'it'"
    ):
        load_pack(pack_dir)


def test_unknown_service_reference_is_rejected(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, "demo-clinic", tmp_path)
    edit_yaml(
        pack_dir / "data" / "doctors.yaml", lambda data: data[0]["service_ids"].append("cardiology")
    )
    with pytest.raises(PackError, match="unknown service 'cardiology'"):
        load_pack(pack_dir)


def test_orientation_needs_the_receptionist_profile(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, "demo-shop", tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda data: data["features"].update(orientation=True))
    with pytest.raises(PackError, match="feature 'orientation'"):
        load_pack(pack_dir)


def test_calendar_integration_needs_the_receptionist_profile(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, "demo-shop", tmp_path)
    edit_yaml(
        pack_dir / "business.yaml",
        lambda data: data.update(
            integrations={"calendar": {"credentials_ref": "secret/x", "calendars": {"a": "b"}}}
        ),
    )
    with pytest.raises(PackError, match="calendar integration needs profile 'receptionist'"):
        load_pack(pack_dir)


def test_knowledge_document_needs_front_matter(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, "demo-shop", tmp_path)
    (pack_dir / "knowledge" / "broken.md").write_text("# No front matter\n", encoding="utf-8")
    with pytest.raises(PackError, match="front matter"):
        load_pack(pack_dir)


def test_missing_directory_is_a_pack_error(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="not found"):
        load_pack(tmp_path / "nowhere")


# --- the shipped demo packs expose every data file -----------------------------------------------


def test_demo_clinic_exposes_every_data_file(clinic_pack: BusinessPack) -> None:
    assert clinic_pack.location.phone == "+355 00 000 0000"
    assert [h.days for h in clinic_pack.location.opening_hours] == ["mon-fri", "sat"]
    assert [d.id for d in clinic_pack.doctors] == [
        "dr-arben-demo",
        "dr-giulia-esempio",
        "ft-elira-shembull",
    ]
    assert {p.service_id: p.amount for p in clinic_pack.prices} == {
        "orthopedics-visit": "40",
        "physiotherapy": "30",
        "dermatology-visit": "35",
    }
    assert [o.id for o in clinic_pack.orientation] == ["back", "knee", "skin"]
    assert [d.key for d in clinic_pack.documents] == [
        "first-visit-it",
        "parking-en",
        "payments-sq",
        "physiotherapy-it",
        "physiotherapy-sq",
    ]


def test_demo_shop_exposes_every_data_file(shop_pack: BusinessPack) -> None:
    assert shop_pack.location.phone == "+39 00 0000 0000"
    assert [h.days for h in shop_pack.location.opening_hours] == ["tue-sat"]
    assert [d.key for d in shop_pack.documents] == ["returns-en", "returns-it", "shipping-it"]
    assert shop_pack.orders[0].status == {"it": "spedito", "en": "shipped"}


# --- file errors name the file (path inside the pack), and the line and column for YAML ---------


def test_missing_business_file_is_named(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / "business.yaml").unlink()
    with pytest.raises(PackError, match=r"^missing file: business\.yaml$"):
        load_pack(pack_dir)


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "just text\n"])
def test_business_file_must_be_a_mapping(
    businesses_dir: Path, tmp_path: Path, content: str
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / "business.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(PackError, match=r"^business\.yaml: expected a mapping$"):
        load_pack(pack_dir)


@pytest.mark.parametrize(
    ("file", "content", "message"),
    [
        pytest.param(
            "business.yaml",
            "id: demo-shop\nname: Bottega Demo: the shop\n",
            "business.yaml: invalid YAML at line 2, column 19: mapping values are not allowed here",
            id="business-file",
        ),
        pytest.param(
            "data/location.yaml",
            "address: Via Roma 1\nphone: +39: 00\n",
            "data/location.yaml: invalid YAML at line 2, column 11: "
            "mapping values are not allowed here",
            id="data-file",
        ),
        pytest.param(
            "data/orders.yaml",
            "- reference: ORD-10427\n---\n- reference: ORD-10458\n",
            "data/orders.yaml: invalid YAML at line 2, column 1: "
            "expected a single document in the stream, but found another document",
            id="context-and-problem",
        ),
    ],
)
def test_invalid_yaml_names_the_file_line_and_column(
    businesses_dir: Path, tmp_path: Path, file: str, content: str, message: str
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / file).write_text(content, encoding="utf-8")
    with pytest.raises(PackError, match=f"^{re.escape(message)}$"):
        load_pack(pack_dir)


def test_yaml_error_without_a_position_still_names_the_file(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / "data" / "location.yaml").write_text("address: \x00x\n", encoding="utf-8")
    with pytest.raises(
        PackError, match=r"^data/location\.yaml: invalid YAML: unacceptable character #x0000"
    ):
        load_pack(pack_dir)


def test_invalid_front_matter_names_the_file_and_the_line_in_it(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    document = "---\ntitle: Returns: how to\nlanguage: en\n---\nBody\n"
    (pack_dir / "knowledge" / "returns-en.md").write_text(document, encoding="utf-8")
    message = (
        "knowledge/returns-en.md: invalid YAML at line 2, column 15: "
        "mapping values are not allowed here"
    )
    with pytest.raises(PackError, match=f"^{re.escape(message)}$"):
        load_pack(pack_dir)


@pytest.mark.parametrize(
    ("file", "text"),
    [
        pytest.param(
            "knowledge/returns-it.md",
            "---\ntitle: Resi\nlanguage: it\n---\nPerché\n",
            id="knowledge-document",
        ),
        pytest.param(
            "data/location.yaml",
            "address: Perché\nphone: x\nemail: y\nopening_hours: []\n",
            id="data-file",
        ),
    ],
)
def test_file_that_is_not_utf8_is_named(
    businesses_dir: Path, tmp_path: Path, file: str, text: str
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / file).write_bytes(text.encode("latin-1"))
    with pytest.raises(PackError, match=rf"^{re.escape(file)}: not valid UTF-8 text"):
        load_pack(pack_dir)


def test_byte_order_mark_is_ignored(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    for relative in ("business.yaml", "data/orders.yaml", "knowledge/returns-en.md"):
        path = pack_dir / relative
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_pack(pack_dir) == load_pack(businesses_dir / SHOP)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("Intro\n---\ntitle: X\nlanguage: it\n---\nBody\n", id="no-opening-delimiter"),
        pytest.param(
            "---\ntitle: X\nlanguage: it\nBody, never closed\n", id="no-closing-delimiter"
        ),
    ],
)
def test_front_matter_needs_an_opening_and_a_closing_delimiter(
    businesses_dir: Path, tmp_path: Path, document: str
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / "knowledge" / "shipping-it.md").write_text(document, encoding="utf-8")
    message = r"^knowledge/shipping-it\.md: knowledge documents start with YAML front matter$"
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


@pytest.mark.parametrize("front", ["- a\n- b\n", ""], ids=["a-list", "empty"])
def test_front_matter_must_be_a_mapping(businesses_dir: Path, tmp_path: Path, front: str) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / "knowledge" / "shipping-it.md").write_text(
        f"---\n{front}---\nBody\n", encoding="utf-8"
    )
    message = r"^knowledge/shipping-it\.md: front matter must be a mapping$"
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


# --- a key written twice is an error, because YAML would keep only the last value ----------------


@pytest.mark.parametrize(
    ("file", "content", "message"),
    [
        pytest.param(
            "business.yaml",
            "id: demo-shop\nname: Bottega Demo\nname: Another name\n",
            "business.yaml: invalid YAML at line 3, column 1: "
            "found duplicate key 'name' (first at line 2)",
            id="business-file",
        ),
        pytest.param(
            "data/location.yaml",
            "address: Via Roma 1\nphone: +39 00 0000 0000\nemail: a@b.example\n"
            "phone: +39 11 1111 1111\n",
            "data/location.yaml: invalid YAML at line 4, column 1: "
            "found duplicate key 'phone' (first at line 2)",
            id="data-file",
        ),
        pytest.param(
            "data/orders.yaml",
            "- reference: ORD-10427\n  postal_code: '20121'\n  postal_code: '20122'\n",
            "data/orders.yaml: invalid YAML at line 3, column 3: "
            "found duplicate key 'postal_code' (first at line 2)",
            id="inside-a-list-item",
        ),
        pytest.param(
            "knowledge/returns-it.md",
            "---\ntitle: Resi\nlanguage: it\ntitle: Altro\n---\nBody\n",
            "knowledge/returns-it.md: invalid YAML at line 4, column 1: "
            "found duplicate key 'title' (first at line 2)",
            id="front-matter",
        ),
    ],
)
def test_a_duplicate_key_names_the_file_the_line_and_the_key(
    businesses_dir: Path, tmp_path: Path, file: str, content: str, message: str
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / file).write_text(content, encoding="utf-8")
    with pytest.raises(PackError, match=f"^{re.escape(message)}$"):
        load_pack(pack_dir)


def test_a_second_handoff_text_cannot_replace_the_contact_text(
    businesses_dir: Path, tmp_path: Path
) -> None:
    # Without the check the second line wins, and the pack would load with a promise.
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    business = pack_dir / "business.yaml"
    lines = business.read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("    handoff:"))
    lines.insert(first + 1, '    handoff: "La reception ti contatterà presto."')
    business.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(
        PackError,
        match=rf"^business\.yaml: invalid YAML at line {first + 2}, column 5: "
        rf"found duplicate key 'handoff' \(first at line {first + 1}\)$",
    ):
        load_pack(pack_dir)


def test_a_key_that_overrides_a_merged_one_is_not_a_duplicate(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / "data" / "location.yaml").write_text(
        'address: "Via dell\'Esempio 1"\nphone: "+39 00 0000 0000"\nemail: "a@b.example"\n'
        "opening_hours:\n"
        '  - &weekday {days: "tue-fri", opens: "10:00", closes: "19:30"}\n'
        '  - {<<: *weekday, days: "sat"}\n',
        encoding="utf-8",
    )
    hours = load_pack(pack_dir).location.opening_hours
    assert [(h.days, h.opens) for h in hours] == [("tue-fri", "10:00"), ("sat", "10:00")]


# --- errors that are not YAML errors still name the file -----------------------------------------


def test_an_impossible_date_is_a_pack_error_naming_the_file(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    prices = pack_dir / "data" / "prices.yaml"
    prices.write_text(
        prices.read_text(encoding="utf-8") + "  valid_until: 2026-02-30\n", encoding="utf-8"
    )
    with pytest.raises(PackError, match=r"^data/prices\.yaml: invalid value: .*day"):
        load_pack(pack_dir)


@pytest.mark.parametrize("file", ["data/location.yaml", "business.yaml", "knowledge/returns-it.md"])
def test_a_file_that_cannot_be_read_is_a_pack_error_naming_it(
    businesses_dir: Path, tmp_path: Path, file: str
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    (pack_dir / file).unlink()
    (pack_dir / file).mkdir()  # a directory where the file should be: reading it fails
    with pytest.raises(PackError, match=rf"^{re.escape(file)}: cannot be read \(Is a directory\)$"):
        load_pack(pack_dir)
