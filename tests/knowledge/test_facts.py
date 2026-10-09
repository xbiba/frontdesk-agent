import uuid
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, NamedTuple

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.knowledge.facts import (
    Catalog,
    doctor_evidence,
    field_label,
    load_catalog,
    localized_days,
    location_evidence,
    orientation_evidence,
    price_evidence,
    service_evidence,
)
from frontdesk.knowledge.versions import active_version_id, create_draft
from frontdesk.tenancy.pack import BusinessPack, Language

TODAY = date(2026, 10, 7)
Activate = Callable[..., sessionmaker[Session]]


def _edited(pack: BusinessPack, **changes: Any) -> BusinessPack:
    """The pack with fields replaced and validated again: test data must obey the pack rules."""
    return BusinessPack.model_validate({**pack.model_dump(), **changes})


def _with_price(pack: BusinessPack, service_id: str, **changes: Any) -> BusinessPack:
    prices = [
        {**price.model_dump(), **changes} if price.service_id == service_id else price.model_dump()
        for price in pack.prices
    ]
    return _edited(pack, prices=prices)


@pytest.mark.db
def test_location_keeps_exact_values(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        evidence = location_evidence(session, clinic_pack.id, "it")
    assert evidence is not None
    assert evidence.kind is EvidenceKind.RECORD
    assert "Indirizzo: Rruga e Shembullit 1" in evidence.content
    assert evidence.values["hours"] == "lun-ven 08:00-19:00; sab 09:00-14:00"
    assert "Orari: lun-ven 08:00-19:00; sab 09:00-14:00" in evidence.content


@pytest.mark.db
def test_price_renders_amount_and_condition(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        evidence = price_evidence(session, clinic_pack.id, "physiotherapy", "it", TODAY)
    assert evidence is not None
    assert evidence.content == "Fisioterapia: 30 EUR (a seduta)"
    assert evidence.values == {"price": "30 EUR"}


@pytest.mark.db
def test_services_without_a_price_have_no_price_evidence(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        assert price_evidence(session, clinic_pack.id, "ultrasound", "it", TODAY) is None
        assert price_evidence(session, clinic_pack.id, "unknown", "it", TODAY) is None


@pytest.mark.db
def test_service_and_doctor_evidence(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        services = service_evidence(session, clinic_pack.id, ["ultrasound", "nope"], "en")
        doctors = doctor_evidence(session, clinic_pack.id, ["dr-arben-demo"], "en")
    assert len(services) == 1
    assert "Requires a prior consultation" in services[0].content
    assert "Dr.ssa Giulia Esempio" in services[0].content
    assert doctors[0].content == "Dr. Arben Demo: Orthopedic visit"


@pytest.mark.db
def test_orientation_comes_from_the_map(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        evidence = orientation_evidence(session, clinic_pack.id, "back", "sq")
        missing = orientation_evidence(session, clinic_pack.id, "heart", "sq")
    assert evidence is not None
    assert "Vizitë ortopedike" in evidence.content
    assert "Osteopati" in evidence.content
    assert missing is None


@pytest.mark.db
def test_catalog_lists_allowed_ids(activate_pack: Activate, clinic_pack: BusinessPack) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        catalog = load_catalog(session, clinic_pack.id)
    assert catalog is not None
    assert set(catalog.services) == {s.id for s in clinic_pack.services}
    assert catalog.doctors["dr-arben-demo"] == "Dr. Arben Demo"
    assert catalog.problem_categories["knee"]["it"] == ["ginocchio"]


@pytest.mark.db
def test_no_active_knowledge_means_no_catalog(app_url: str, clinic_pack: BusinessPack) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, clinic_pack.id) as session:
        assert load_catalog(session, clinic_pack.id) is None
        assert location_evidence(session, clinic_pack.id, "sq") is None


# Days and labels


@pytest.mark.parametrize(
    ("language", "names"),
    [
        ("sq", ["Hën", "Mar", "Mër", "Enj", "Pre", "Sht", "Die"]),
        ("it", ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]),
        ("en", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    ],
)
def test_every_day_has_a_name_in_each_language(language: Language, names: list[str]) -> None:
    week = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    assert [localized_days(day, language) for day in week] == names


@pytest.mark.parametrize(
    ("days", "language", "expected"),
    [
        ("mon-fri", "it", "lun-ven"),
        ("tue-sat", "sq", "Mar-Sht"),
        ("sat", "en", "Sat"),
        ("sat-mon", "it", "sab-lun"),  # printed as the pack wrote it, never reordered
    ],
)
def test_day_ranges_are_joined_as_the_pack_wrote_them(
    days: str, language: Language, expected: str
) -> None:
    assert localized_days(days, language) == expected


@pytest.mark.parametrize(
    ("key", "language", "expected"),
    [
        ("address", "it", "Indirizzo"),
        ("address", "sq", "Adresa"),
        ("phone", "en", "Phone"),
        ("hours", "en", "Opening hours"),
        ("price", "sq", "Çmimi"),
    ],
)
def test_field_labels_follow_the_language(key: str, language: Language, expected: str) -> None:
    assert field_label(key, language) == expected


# Location


@pytest.mark.db
@pytest.mark.parametrize(
    ("language", "title", "labels", "hours"),
    [
        (
            "sq",
            "Adresa dhe kontaktet",
            ("Adresa", "Telefoni", "Email", "Orari"),
            "Hën-Pre 08:00-19:00; Sht 09:00-14:00",
        ),
        (
            "it",
            "Indirizzo e contatti",
            ("Indirizzo", "Telefono", "Email", "Orari"),
            "lun-ven 08:00-19:00; sab 09:00-14:00",
        ),
        (
            "en",
            "Address and contacts",
            ("Address", "Phone", "Email", "Opening hours"),
            "Mon-Fri 08:00-19:00; Sat 09:00-14:00",
        ),
    ],
)
def test_location_is_written_in_the_customers_language(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    language: Language,
    title: str,
    labels: tuple[str, str, str, str],
    hours: str,
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        evidence = location_evidence(session, clinic_pack.id, language)
    assert evidence is not None
    location = clinic_pack.location
    values = {
        "address": location.address,
        "phone": location.phone,
        "email": location.email,
        "hours": hours,
    }
    assert (evidence.source_ref, evidence.title) == ("location:main", title)
    assert evidence.values == values
    assert evidence.content == "\n".join(
        f"{label}: {value}" for label, value in zip(labels, values.values(), strict=True)
    )
    # a caller showing one value writes `field_label(key): values[key]`; it must match the content
    assert labels == tuple(field_label(key, language) for key in evidence.values)


# Prices


@pytest.mark.db
@pytest.mark.parametrize(
    ("language", "title", "content"),
    [
        ("sq", "Çmimi", "Fizioterapi: 30 EUR (për seancë)"),
        ("it", "Prezzo", "Fisioterapia: 30 EUR (a seduta)"),
        ("en", "Price", "Physiotherapy: 30 EUR (per session)"),
    ],
)
def test_a_price_is_written_in_the_customers_language(
    activate_pack: Activate, clinic_pack: BusinessPack, language: Language, title: str, content: str
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        evidence = price_evidence(session, clinic_pack.id, "physiotherapy", language, TODAY)
    assert evidence is not None
    assert evidence.kind is EvidenceKind.RECORD
    assert (evidence.source_ref, evidence.title, evidence.content) == (
        "price:physiotherapy",
        title,
        content,
    )


@pytest.mark.db
@pytest.mark.parametrize(
    ("changes", "shown"),
    [
        ({"max_amount": "55"}, "40-55 EUR"),
        ({"amount": "9", "max_amount": "10"}, "9-10 EUR"),  # 10 is above 9 as a number, not as text
        ({"amount": "40.50"}, "40.50 EUR"),
        ({"currency": "ALL", "amount": "4000"}, "4000 ALL"),
        ({"max_amount": "40"}, "40 EUR"),
        ({"max_amount": "40.00"}, "40 EUR"),
    ],
    ids=[
        "range",
        "range across digit counts",
        "cents kept",
        "other currency",
        "collapsed range",
        "collapsed range in cents",
    ],
)
def test_amounts_are_shown_as_the_pack_wrote_them(
    activate_pack: Activate, clinic_pack: BusinessPack, changes: dict[str, str], shown: str
) -> None:
    pack = _with_price(clinic_pack, "orthopedics-visit", **changes)
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        evidence = price_evidence(session, pack.id, "orthopedics-visit", "en", TODAY)
    assert evidence is not None
    assert evidence.values == {"price": shown}
    assert evidence.content == f"Orthopedic visit: {shown}"


@pytest.mark.db
@pytest.mark.parametrize(
    ("window", "current"),
    [
        ({"valid_from": TODAY + timedelta(days=1)}, False),
        ({"valid_from": TODAY}, True),
        ({"valid_until": TODAY - timedelta(days=1)}, False),
        ({"valid_until": TODAY}, True),
        (
            {"valid_from": TODAY - timedelta(days=30), "valid_until": TODAY + timedelta(days=30)},
            True,
        ),
    ],
    ids=["starts tomorrow", "starts today", "ended yesterday", "ends today", "inside its window"],
)
def test_a_price_applies_from_its_first_to_its_last_day(
    activate_pack: Activate, clinic_pack: BusinessPack, window: dict[str, date], current: bool
) -> None:
    pack = _with_price(clinic_pack, "physiotherapy", **window)
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        evidence = price_evidence(session, pack.id, "physiotherapy", "it", TODAY)
    assert (evidence is not None) is current


@pytest.mark.db
def test_the_caller_decides_what_today_is(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = _with_price(clinic_pack, "physiotherapy", valid_from=TODAY, valid_until=TODAY)
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        found = {
            offset: price_evidence(
                session, pack.id, "physiotherapy", "it", TODAY + timedelta(days=offset)
            )
            for offset in (-1, 0, 1)
        }
    assert {offset: item is not None for offset, item in found.items()} == {
        -1: False,
        0: True,
        1: False,
    }


# Services, doctors and orientation


@pytest.mark.db
@pytest.mark.parametrize(
    ("language", "title", "content"),
    [
        (
            "sq",
            "Shërbimi",
            "Ekografi\nKohëzgjatja: 20 minuta\nKërkon konsultë paraprake\n"
            "Mjekët: Dr.ssa Giulia Esempio",
        ),
        (
            "it",
            "Servizio",
            "Ecografia\nDurata: 20 minuti\nRichiede un consulto preliminare\n"
            "Medici: Dr.ssa Giulia Esempio",
        ),
        (
            "en",
            "Service",
            "Ultrasound scan\nDuration: 20 minutes\nRequires a prior consultation\n"
            "Doctors: Dr.ssa Giulia Esempio",
        ),
    ],
)
def test_a_service_lists_its_duration_consult_rule_and_doctors(
    activate_pack: Activate, clinic_pack: BusinessPack, language: Language, title: str, content: str
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        [evidence] = service_evidence(session, clinic_pack.id, ["ultrasound"], language)
    assert evidence.kind is EvidenceKind.RECORD
    assert (evidence.source_ref, evidence.title, evidence.content) == (
        "service:ultrasound",
        title,
        content,
    )


@pytest.mark.db
def test_a_service_that_needs_no_consult_says_nothing_about_one(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        [evidence] = service_evidence(session, clinic_pack.id, ["physiotherapy"], "sq")
    assert evidence.content == "Fizioterapi\nKohëzgjatja: 45 minuta\nMjekët: Elira Shembull"


@pytest.mark.db
def test_a_service_nobody_offers_lists_no_doctors(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = _edited(clinic_pack, doctors=[])
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        [evidence] = service_evidence(session, pack.id, ["physiotherapy"], "it")
    assert evidence.content == "Fisioterapia\nDurata: 45 minuti"


@pytest.mark.db
def test_services_come_back_in_the_order_asked(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    asked = ["osteopathy", "dermatology-visit", "physiotherapy"]
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        found = service_evidence(session, clinic_pack.id, asked, "en")
    assert [item.source_ref for item in found] == [f"service:{service}" for service in asked]


@pytest.mark.db
@pytest.mark.parametrize(
    ("language", "title", "content"),
    [
        ("sq", "Mjeku", "Dr.ssa Giulia Esempio: Vizitë dermatologjike, Ekografi"),
        ("it", "Medico", "Dr.ssa Giulia Esempio: Visita dermatologica, Ecografia"),
        ("en", "Doctor", "Dr.ssa Giulia Esempio: Dermatology visit, Ultrasound scan"),
    ],
)
def test_a_doctor_lists_the_services_they_offer(
    activate_pack: Activate, clinic_pack: BusinessPack, language: Language, title: str, content: str
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        [evidence] = doctor_evidence(session, clinic_pack.id, ["dr-giulia-esempio"], language)
    assert evidence.kind is EvidenceKind.RECORD
    assert (evidence.source_ref, evidence.title, evidence.content) == (
        "doctor:dr-giulia-esempio",
        title,
        content,
    )


@pytest.mark.db
def test_doctors_come_back_in_the_order_asked_and_unknown_ones_are_skipped(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    asked = ["ft-elira-shembull", "nope", "dr-arben-demo"]
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        found = doctor_evidence(session, clinic_pack.id, asked, "it")
    assert [item.source_ref for item in found] == [
        "doctor:ft-elira-shembull",
        "doctor:dr-arben-demo",
    ]


@pytest.mark.db
@pytest.mark.parametrize(
    ("language", "title", "labels", "services"),
    [
        ("sq", "Shërbimet që merren me këtë", "shpinë, kurriz", "Vizitë ortopedike, Osteopati"),
        ("it", "Servizi che se ne occupano", "schiena, lombare", "Visita ortopedica, Osteopatia"),
        ("en", "Services that handle this", "back, lower back", "Orthopedic visit, Osteopathy"),
    ],
)
def test_orientation_lists_the_mapped_services_in_the_customers_language(
    activate_pack: Activate,
    clinic_pack: BusinessPack,
    language: Language,
    title: str,
    labels: str,
    services: str,
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        evidence = orientation_evidence(session, clinic_pack.id, "back", language)
    assert evidence is not None
    assert evidence.kind is EvidenceKind.RECORD
    assert (evidence.source_ref, evidence.title) == ("orientation:back", title)
    assert evidence.content == f"{labels}: {services}"
    assert evidence.values == {"services": services}


# The catalog and the order of records


@pytest.mark.db
def test_catalog_carries_every_name_and_label(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with business_session(factory, clinic_pack.id) as session:
        catalog = load_catalog(session, clinic_pack.id)
        version_id = active_version_id(session, clinic_pack.id)
    assert catalog is not None
    assert (catalog.business_id, catalog.version_id) == (clinic_pack.id, version_id)
    assert catalog.services == {s.id: dict(s.names) for s in clinic_pack.services}
    assert catalog.doctors == {d.id: d.name for d in clinic_pack.doctors}
    assert catalog.problem_categories == {e.id: e.labels for e in clinic_pack.orientation}


@pytest.mark.db
def test_records_are_read_in_id_order_whatever_the_order_in_the_pack(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    doctors = [
        {**doctor.model_dump(), "service_ids": [*doctor.service_ids, "physiotherapy"]}
        if doctor.id == "dr-arben-demo"
        else doctor.model_dump()
        for doctor in reversed(clinic_pack.doctors)
    ]
    pack = _edited(
        clinic_pack,
        doctors=doctors,
        services=list(reversed(clinic_pack.services)),
        orientation=list(reversed(clinic_pack.orientation)),
    )
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        # The unique index on (version, kind, key) hands rows back sorted by key on its own.
        # Without it a sequential scan returns them as inserted: only ORDER BY sorts them then.
        for statement in ("SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"):
            session.execute(text(statement))
        catalog = load_catalog(session, pack.id)
        [service] = service_evidence(session, pack.id, ["physiotherapy"], "en")
    assert catalog is not None
    assert list(catalog.services) == sorted(catalog.services)
    assert list(catalog.doctors) == sorted(catalog.doctors)
    assert list(catalog.problem_categories) == sorted(catalog.problem_categories)
    assert service.content.endswith("Doctors: Dr. Arben Demo, Elira Shembull")


# Versions and tenants


class _Snapshot(NamedTuple):
    version_id: uuid.UUID | None
    price: Evidence | None
    service: list[Evidence]
    catalog: Catalog | None


def _snapshot(factory: sessionmaker[Session], business_id: str) -> _Snapshot:
    with business_session(factory, business_id) as session:
        return _Snapshot(
            active_version_id(session, business_id),
            price_evidence(session, business_id, "physiotherapy", "it", TODAY),
            service_evidence(session, business_id, ["physiotherapy"], "it"),
            load_catalog(session, business_id),
        )


@pytest.mark.db
def test_facts_follow_the_active_version(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    first = _snapshot(factory, clinic_pack.id)
    edited = _with_price(clinic_pack, "physiotherapy", amount="35")
    with business_session(factory, clinic_pack.id) as session:
        create_draft(session, edited)
    assert _snapshot(factory, clinic_pack.id) == first  # the draft waits for approval
    activate_pack(edited)
    second = _snapshot(factory, clinic_pack.id)
    assert first.version_id is not None
    assert second.version_id is not None
    assert second.version_id != first.version_id
    assert first.price is not None
    assert second.price is not None
    assert (first.price.values, second.price.values) == ({"price": "30 EUR"}, {"price": "35 EUR"})
    assert (first.price.version, second.price.version) == (
        str(first.version_id),
        str(second.version_id),
    )
    # a record both versions share is listed once, not once per version
    listing = "Fisioterapia\nDurata: 45 minuti\nMedici: Elira Shembull"
    assert [item.content for item in first.service] == [listing]
    assert [item.content for item in second.service] == [listing]
    assert first.catalog is not None
    assert second.catalog is not None
    assert (first.catalog.version_id, second.catalog.version_id) == (
        first.version_id,
        second.version_id,
    )


@pytest.mark.db
def test_a_draft_is_not_a_source_of_facts(app_url: str, clinic_pack: BusinessPack) -> None:
    factory = make_session_factory(make_engine(app_url))
    with business_session(factory, clinic_pack.id) as session:
        create_draft(session, clinic_pack)
        business = clinic_pack.id
        assert load_catalog(session, business) is None
        assert location_evidence(session, business, "it") is None
        assert price_evidence(session, business, "physiotherapy", "it", TODAY) is None
        assert service_evidence(session, business, ["physiotherapy"], "it") == []
        assert doctor_evidence(session, business, ["dr-arben-demo"], "it") == []
        assert orientation_evidence(session, business, "back", "it") is None


@pytest.mark.db
def test_other_businesses_are_invisible(
    activate_pack: Activate, clinic_pack: BusinessPack, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    activate_pack(shop_pack)
    with business_session(factory, clinic_pack.id) as session:
        assert load_catalog(session, shop_pack.id) is None
        assert location_evidence(session, shop_pack.id, "it") is None
    with business_session(factory, shop_pack.id) as session:
        catalog = load_catalog(session, shop_pack.id)
        location = location_evidence(session, shop_pack.id, "it")
    assert catalog is not None  # a shop has no services, doctors or orientation: still a catalog
    assert (catalog.services, catalog.doctors, catalog.problem_categories) == ({}, {}, {})
    assert location is not None
    assert location.values["address"] == shop_pack.location.address
    assert location.values["hours"] == "mar-sab 10:00-19:30"
