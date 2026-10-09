"""Value rules: shown text says something, a price range and a validity window are possible."""

import re
from datetime import date
from pathlib import Path

import pytest

from frontdesk.tenancy.loader import PackError, load_pack
from tests.tenancy.helpers import CLINIC, SHOP, Change, copy_pack, edit_document, edit_yaml

BLANK = [
    pytest.param(CLINIC, "business.yaml", lambda d: d.update(name=" "), "name", id="business-name"),
    pytest.param(
        CLINIC,
        "data/location.yaml",
        lambda d: d.update(address=""),
        "location.address",
        id="address",
    ),
    pytest.param(
        CLINIC,
        "data/location.yaml",
        lambda d: d.update(phone="  "),
        "location.phone",
        id="phone",
    ),
    pytest.param(
        CLINIC,
        "data/location.yaml",
        lambda d: d.update(email=""),
        "location.email",
        id="email",
    ),
    pytest.param(
        CLINIC,
        "data/services.yaml",
        lambda d: d[0]["names"].update(sq=""),
        "services.0.names.sq",
        id="service-name",
    ),
    pytest.param(
        CLINIC,
        "data/doctors.yaml",
        lambda d: d[0].update(name="\t"),
        "doctors.0.name",
        id="doctor-name",
    ),
    pytest.param(
        SHOP,
        "data/orders.yaml",
        lambda d: d[0]["status"].update(it="  "),
        "orders.0.status.it",
        id="order-status-text",
    ),
]


@pytest.mark.parametrize(("name", "file", "change", "field"), BLANK)
def test_names_contacts_and_statuses_must_not_be_blank(
    businesses_dir: Path, tmp_path: Path, name: str, file: str, change: Change, field: str
) -> None:
    pack_dir = copy_pack(businesses_dir, name, tmp_path)
    edit_yaml(pack_dir / file, change)
    message = rf"\n{re.escape(field)}\n\s+Value error, must not be blank"
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


def test_order_status_needs_at_least_one_text(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    edit_yaml(pack_dir / "data" / "orders.yaml", lambda data: data[0].update(status={}))
    with pytest.raises(
        PackError, match=r"\norders\.0\.status\n\s+Dictionary should have at least 1 item"
    ):
        load_pack(pack_dir)


def test_price_range_must_not_end_below_its_start(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "data" / "prices.yaml", lambda data: data[1].update(max_amount="20"))
    message = (
        r"\nprices\.1\n\s+Value error, price for 'physiotherapy': "
        r"max_amount 20 is below amount 30"
    )
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


@pytest.mark.parametrize("max_amount", ["30", "35.50", "100"])
def test_price_range_compares_numbers_and_may_end_where_it_starts(
    businesses_dir: Path, tmp_path: Path, max_amount: str
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "data" / "prices.yaml", lambda data: data[1].update(max_amount=max_amount))
    assert load_pack(pack_dir).prices[1].max_amount == max_amount


REVERSED = {"valid_from": date(2027, 1, 1), "valid_until": date(2026, 12, 31)}


def test_price_validity_must_not_end_before_it_starts(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "data" / "prices.yaml", lambda data: data[0].update(REVERSED))
    message = (
        r"\nprices\.0\n\s+Value error, price for 'orthopedics-visit': "
        r"valid_from 2027-01-01 is after valid_until 2026-12-31"
    )
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


def test_document_validity_must_not_end_before_it_starts(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    edit_document(pack_dir / "knowledge" / "returns-en.md", lambda meta: meta.update(REVERSED))
    message = (
        r"\ndocuments\.0\n\s+Value error, document 'returns-en': "
        r"valid_from 2027-01-01 is after valid_until 2026-12-31"
    )
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


@pytest.mark.parametrize(
    "window",
    [
        {"valid_from": date(2026, 10, 1), "valid_until": date(2026, 10, 1)},
        {"valid_from": date(2026, 10, 1)},
        {"valid_until": date(2026, 10, 1)},
    ],
    ids=["single-day", "no-end", "no-start"],
)
def test_validity_may_be_a_single_day_or_open_ended(
    businesses_dir: Path, tmp_path: Path, window: dict[str, date]
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "data" / "prices.yaml", lambda data: data[0].update(window))
    edit_document(pack_dir / "knowledge" / "parking-en.md", lambda meta: meta.update(window))
    pack = load_pack(pack_dir)
    parking = next(d for d in pack.documents if d.key == "parking-en")
    assert pack.prices[0].model_dump(include=set(window)) == window
    assert parking.model_dump(include=set(window)) == window


@pytest.mark.parametrize(
    "zone", ["Europe/Tiranee", "Tirane", "", "../etc/passwd", "/etc/localtime"]
)
def test_timezone_must_be_a_known_iana_zone(
    businesses_dir: Path, tmp_path: Path, zone: str
) -> None:
    """The business-local day picks the prices and documents that apply: a typo must not load."""
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda data: data.update(timezone=zone))
    message = rf"\ntimezone\n\s+Value error, unknown time zone '{re.escape(zone)}'"
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


@pytest.mark.parametrize("zone", ["europe/tirane", "EUROPE/TIRANE", "Europe/tirane", "utc"])
def test_timezone_names_are_matched_case_sensitively(
    businesses_dir: Path, tmp_path: Path, zone: str
) -> None:
    """A Mac opens these zone files (its file system ignores case) and Linux does not.

    The container image runs on Linux, so a mis-cased name that loads on a laptop would only fail
    when the container starts. The pack refuses it everywhere.
    """
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda data: data.update(timezone=zone))
    message = rf"\ntimezone\n\s+Value error, unknown time zone '{re.escape(zone)}'"
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


@pytest.mark.parametrize(
    "zone", ["UTC", "Europe/Tirane", "Europe/Rome", "America/Argentina/Buenos_Aires"]
)
def test_timezone_may_be_any_iana_zone(businesses_dir: Path, tmp_path: Path, zone: str) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda data: data.update(timezone=zone))
    assert load_pack(pack_dir).timezone == zone
