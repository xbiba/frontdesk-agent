"""Each consistency rule and schema constraint of a business pack, pinned by its message."""

import copy
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from frontdesk.tenancy.loader import PackError, load_pack
from frontdesk.tenancy.pack import BusinessPack, FixedTextKey
from tests.tenancy.helpers import (
    CLINIC,
    SHOP,
    Change,
    copy_pack,
    edit_document,
    edit_yaml,
    write_yaml,
)

ORIENTATION_RULE = "feature 'orientation' needs profile 'receptionist' and an orientation map"


def _duplicate_first(data: Any) -> None:
    data.append(copy.deepcopy(data[0]))


def _calendar_with(**settings: Any) -> Change:
    calendar = {"credentials_ref": "adc", "calendars": {"dr-arben-demo": "c1"}, **settings}
    return lambda data: data.update(integrations={"calendar": calendar})


def _calendar_for(*doctor_ids: str) -> Change:
    calendars = {doctor_id: f"calendar-{doctor_id}" for doctor_id in doctor_ids}
    return lambda data: data.update(
        integrations={"calendar": {"credentials_ref": "secret/x", "calendars": calendars}}
    )


RULES = [
    pytest.param(
        SHOP,
        "business.yaml",
        lambda d: d.update(default_language="sq"),
        "default language 'sq' is not in languages",
        id="default-language-is-offered",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d["fixed_texts"]["en"].update(greeting="   "),
        "missing fixed text 'greeting' for language 'en'",
        id="blank-fixed-text",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d["fixed_texts"]["en"].update(next_step="   "),
        "missing fixed text 'next_step' for language 'en'",
        id="blank-next-step",  # optional, but not blank
    ),
    pytest.param(
        SHOP,
        "business.yaml",
        lambda d: d["fixed_texts"].pop("en"),
        "missing fixed text 'disclosure' for language 'en'",
        id="language-without-fixed-texts",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d["fixed_texts"]["it"].update(handoff="La reception ti contatterà presto."),
        "fixed text 'handoff' for language 'it' must contain the location phone or email",
        id="handoff-gives-a-contact",
    ),
    pytest.param(
        SHOP,
        "business.yaml",
        lambda d: d["fixed_texts"]["en"].update(handoff="Write to us."),
        "fixed text 'handoff' for language 'en' must contain the location phone or email",
        id="handoff-gives-a-contact-in-every-language",
    ),
    pytest.param(
        CLINIC,
        "data/location.yaml",
        lambda d: d.update(phone="+355 00 111 1111", email="reception@klinika-demo.example"),
        "fixed text 'handoff' for language 'sq' must contain the location phone or email",
        id="handoff-names-the-current-contact",
    ),
    pytest.param(
        CLINIC,
        "data/services.yaml",
        lambda d: d[0]["names"].pop("en"),
        "service 'orthopedics-visit' has no name in 'en'",
        id="service-name-in-every-language",
    ),
    pytest.param(
        CLINIC,
        "data/orientation.yaml",
        lambda d: d[0]["labels"].pop("en"),
        "orientation 'back' has no labels in 'en'",
        id="orientation-labels-in-every-language",
    ),
    pytest.param(
        CLINIC,
        "data/orientation.yaml",
        lambda d: d[0]["labels"].update(it=[" ", ""]),
        "orientation 'back' has no labels in 'it'",
        id="orientation-labels-not-blank",
    ),
    pytest.param(
        CLINIC,
        "data/prices.yaml",
        lambda d: d[1]["conditions"].pop("en"),
        "price for 'physiotherapy' has no condition in 'en'",
        id="price-condition-in-every-language",
    ),
    pytest.param(
        CLINIC,
        "data/prices.yaml",
        lambda d: d[1]["conditions"].update(sq="  "),
        "price for 'physiotherapy' has no condition in 'sq'",
        id="price-condition-not-blank",
    ),
    pytest.param(
        CLINIC,
        "data/services.yaml",
        _duplicate_first,
        "duplicate service id 'orthopedics-visit'",
        id="duplicate-service",
    ),
    pytest.param(
        CLINIC,
        "data/doctors.yaml",
        _duplicate_first,
        "duplicate doctor id 'dr-arben-demo'",
        id="duplicate-doctor",
    ),
    pytest.param(
        CLINIC,
        "data/orientation.yaml",
        _duplicate_first,
        "duplicate orientation id 'back'",
        id="duplicate-orientation",
    ),
    pytest.param(
        CLINIC,
        "data/prices.yaml",
        _duplicate_first,
        "duplicate price id 'orthopedics-visit'",
        id="duplicate-price",
    ),
    pytest.param(
        SHOP,
        "data/orders.yaml",
        _duplicate_first,
        "duplicate order id 'ORD-10427'",
        id="duplicate-order",
    ),
    pytest.param(
        CLINIC,
        "data/prices.yaml",
        lambda d: d.append({"service_id": "cardiology", "currency": "EUR", "amount": "50"}),
        "price references unknown service 'cardiology'",
        id="price-to-unknown-service",
    ),
    pytest.param(
        CLINIC,
        "data/orientation.yaml",
        lambda d: d[0]["service_ids"].append("cardiology"),
        "orientation 'back' references unknown service 'cardiology'",
        id="orientation-to-unknown-service",
    ),
    pytest.param(
        CLINIC,
        "data/services.yaml",
        lambda d: d.clear(),
        "profile 'receptionist' needs at least one service",
        id="receptionist-needs-a-service",
    ),
    pytest.param(
        CLINIC,
        "data/orientation.yaml",
        lambda d: d.clear(),
        ORIENTATION_RULE,
        id="orientation-needs-a-map",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d.update(profile="support-sales"),
        ORIENTATION_RULE,
        id="orientation-needs-the-receptionist-profile",
    ),
    pytest.param(
        SHOP,
        "data/orders.yaml",
        lambda d: d.clear(),
        "feature 'order_status' needs profile 'support-sales' and orders",
        id="order-status-needs-orders",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        _calendar_for("nobody"),
        "calendar references unknown doctor 'nobody'",
        id="calendar-to-unknown-doctor",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d["fixed_texts"]["it"].update(next_step="Per prenotare scrivici qui."),
        "fixed text 'next_step' for language 'it' must contain the location phone or email",
        id="next-step-gives-a-contact",  # it may stand in for the handoff of a booking
    ),
]


@pytest.mark.parametrize(("name", "file", "change", "message"), RULES)
def test_each_consistency_rule_rejects_its_violation(
    businesses_dir: Path, tmp_path: Path, name: str, file: str, change: Change, message: str
) -> None:
    pack_dir = copy_pack(businesses_dir, name, tmp_path)
    edit_yaml(pack_dir / file, change)
    with pytest.raises(PackError, match=re.escape(message)):
        load_pack(pack_dir)


@pytest.mark.parametrize(
    "handoff",
    [
        pytest.param("Scrivici a info@klinika-demo.example.", id="email-only"),
        pytest.param("Chiama il +355 00 000 0000.", id="phone-only"),
        pytest.param("Scrivici a INFO@Klinika-Demo.example.", id="email-in-another-case"),
    ],
)
def test_a_handoff_may_give_either_the_phone_or_the_email(
    businesses_dir: Path, tmp_path: Path, handoff: str
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda d: d["fixed_texts"]["it"].update(handoff=handoff))
    assert load_pack(pack_dir).fixed_texts["it"][FixedTextKey.HANDOFF] == handoff


def test_order_status_needs_the_support_sales_profile(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda data: data["features"].update(order_status=True))
    order = {"reference": "ORD-10001", "postal_code": "1000", "status": {"en": "shipped"}}
    write_yaml(pack_dir / "data" / "orders.yaml", [order])
    with pytest.raises(
        PackError, match="feature 'order_status' needs profile 'support-sales' and orders"
    ):
        load_pack(pack_dir)


def test_document_in_a_language_the_pack_does_not_offer_is_rejected(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    edit_document(pack_dir / "knowledge" / "returns-en.md", lambda meta: meta.update(language="sq"))
    with pytest.raises(PackError, match="document 'returns-en' is in unsupported language 'sq'"):
        load_pack(pack_dir)


def test_conditions_and_labels_are_required_only_in_the_languages_the_pack_offers(
    clinic_pack: BusinessPack,
) -> None:
    data = clinic_pack.model_dump()
    data["languages"] = ["it", "en"]
    data["default_language"] = "it"
    data["documents"] = [doc for doc in data["documents"] if doc["language"] != "sq"]
    for price in data["prices"]:
        price["conditions"].pop("sq", None)
    for entry in data["orientation"]:
        entry["labels"].pop("sq")
    pack = BusinessPack.model_validate(data)
    assert pack.languages == ["it", "en"]
    assert "sq" not in pack.prices[1].conditions
    assert all("sq" not in entry.labels for entry in pack.orientation)


def test_duplicate_document_keys_are_rejected(shop_pack: BusinessPack) -> None:
    # Keys come from file names, so only a pack built in code can repeat one.
    data = shop_pack.model_dump()
    data["documents"].append(dict(data["documents"][0]))
    with pytest.raises(ValidationError, match="duplicate document id 'returns-en'"):
        BusinessPack.model_validate(data)


def test_calendar_integration_for_known_doctors_loads(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", _calendar_for("dr-arben-demo", "ft-elira-shembull"))
    calendar = load_pack(pack_dir).integrations.calendar
    assert calendar is not None
    assert calendar.credentials_ref == "secret/x"
    assert calendar.calendars == {
        "dr-arben-demo": "calendar-dr-arben-demo",
        "ft-elira-shembull": "calendar-ft-elira-shembull",
    }
    # The defaults of the bookings design (§3): a marker word, 14 days, 2 hours' notice, 3 options.
    assert (calendar.availability_marker, calendar.horizon_days) == ("Disponibile", 14)
    assert calendar.min_notice_minutes == 120
    assert (calendar.buffer_minutes, calendar.max_options) == (0, 3)


def test_bookable_doctors_are_those_with_a_calendar_and_their_bookable_services(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    assert load_pack(pack_dir).bookable_doctors() == {}
    edit_yaml(pack_dir / "business.yaml", _calendar_for("dr-giulia-esempio", "ft-elira-shembull"))
    assert load_pack(pack_dir).bookable_doctors() == {
        "dr-giulia-esempio": ("dermatology-visit",),  # the ultrasound needs a consult first
        "ft-elira-shembull": ("physiotherapy", "osteopathy"),
    }


def test_a_calendar_for_a_doctor_with_no_bookable_service_is_refused(
    businesses_dir: Path, tmp_path: Path
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "data/services.yaml", lambda d: d[3].update(bookable_online=False))
    edit_yaml(pack_dir / "business.yaml", _calendar_for("dr-giulia-esempio"))
    with pytest.raises(PackError, match="calendar for 'dr-giulia-esempio' has no bookable service"):
        load_pack(pack_dir)


SCHEMA = [
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d.update(id="Demo Clinic"),
        r"\nid\n\s+String should match pattern",
        id="id-pattern",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        lambda d: d.update(colour="blue"),
        r"\ncolour\n\s+Extra inputs are not permitted",
        id="unknown-key-in-business-file",
    ),
    pytest.param(
        CLINIC,
        "data/services.yaml",
        lambda d: d[0].update(bonus=1),
        r"\nservices\.0\.bonus\n\s+Extra inputs are not permitted",
        id="unknown-key-in-a-record",
    ),
    pytest.param(
        SHOP,
        "data/orders.yaml",
        lambda d: d[0].update(reference="ORD-1"),
        r"\norders\.0\.reference\n\s+String should match pattern",
        id="order-reference-pattern",
    ),
    pytest.param(
        SHOP,
        "data/orders.yaml",
        lambda d: d[0].update(postal_code="AB123"),
        r"\norders\.0\.postal_code\n\s+String should match pattern",
        id="postal-code-pattern",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        _calendar_with(availability_marker="  "),
        r"\nintegrations\.calendar\.availability_marker\n\s+Value error, must not be blank",
        id="blank-availability-marker",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        _calendar_with(horizon_days=0),
        r"\nintegrations\.calendar\.horizon_days\n\s+Input should be greater than or equal to 1",
        id="horizon-of-at-least-a-day",
    ),
    pytest.param(
        CLINIC,
        "business.yaml",
        _calendar_with(max_options=6),
        r"\nintegrations\.calendar\.max_options\n\s+Input should be less than or equal to 5",
        id="at-most-five-options",
    ),
]


@pytest.mark.parametrize(("name", "file", "change", "message"), SCHEMA)
def test_each_schema_constraint_rejects_its_violation(
    businesses_dir: Path, tmp_path: Path, name: str, file: str, change: Change, message: str
) -> None:
    pack_dir = copy_pack(businesses_dir, name, tmp_path)
    edit_yaml(pack_dir / file, change)
    with pytest.raises(PackError, match=message):
        load_pack(pack_dir)


@pytest.mark.parametrize("value", ["8:00", "24:00", "09:60", "9am"])
def test_opening_times_are_24_hour_hh_mm(businesses_dir: Path, tmp_path: Path, value: str) -> None:
    pack_dir = copy_pack(businesses_dir, SHOP, tmp_path)
    edit_yaml(
        pack_dir / "data" / "location.yaml",
        lambda data: data["opening_hours"][0].update(opens=value),
    )
    with pytest.raises(
        PackError, match=r"\nlocation\.opening_hours\.0\.opens\n\s+String should match pattern"
    ):
        load_pack(pack_dir)


@pytest.mark.parametrize("field", ["amount", "max_amount"])
@pytest.mark.parametrize("value", ["40,5", "4O", "-5", "40.5", ""])
def test_amounts_are_plain_decimals_with_two_places(
    businesses_dir: Path, tmp_path: Path, field: str, value: str
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "data" / "prices.yaml", lambda data: data[0].update({field: value}))
    with pytest.raises(PackError, match=rf"\nprices\.0\.{field}\n\s+String should match pattern"):
        load_pack(pack_dir)


def test_pack_records_are_frozen(clinic_pack: BusinessPack) -> None:
    with pytest.raises(ValidationError, match="Instance is frozen"):
        clinic_pack.location.phone = "+355 11 111 1111"


# --- The orientation map's provenance (spec §7, final review P94) --------------------------------------


def test_the_demo_map_records_its_version_and_approving_doctor(clinic_pack: BusinessPack) -> None:
    approval = clinic_pack.orientation_map
    assert approval is not None
    assert (approval.version, approval.approved_by) == (
        "demo-2026-10",
        "Dr. Arben Demo (synthetic)",
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            lambda d: d.pop("orientation_map"),
            "feature 'orientation' needs orientation_map: the map's version and approving doctor",
            id="map-without-provenance",
        ),
        pytest.param(
            lambda d: d["orientation_map"].update(approved_by="  "),
            "orientation_map.approved_by",
            id="blank-approver",
        ),
        pytest.param(
            lambda d: d["orientation_map"].pop("version"),
            "orientation_map.version",
            id="no-version",
        ),
        pytest.param(
            lambda d: d["orientation_map"].update(version=" "),
            "orientation_map.version",
            id="blank-version",
        ),
        pytest.param(
            lambda d: d["orientation_map"].pop("approved_on"),
            "orientation_map.approved_on",
            id="no-approval-date",
        ),
    ],
)
def test_orientation_needs_the_version_and_the_approving_doctor_of_its_map(
    businesses_dir: Path, tmp_path: Path, change: Change, message: str
) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", change)
    with pytest.raises(PackError, match=re.escape(message)):
        load_pack(pack_dir)


def test_a_map_approval_without_a_map_is_refused(businesses_dir: Path, tmp_path: Path) -> None:
    pack_dir = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(pack_dir / "business.yaml", lambda d: d["features"].update(orientation=False))
    edit_yaml(pack_dir / "data" / "orientation.yaml", lambda d: d.clear())
    with pytest.raises(PackError, match="orientation_map describes a map"):
        load_pack(pack_dir)
