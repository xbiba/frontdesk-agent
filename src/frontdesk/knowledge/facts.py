"""Exact facts from approved records (ADR-0009): rendered per language, values kept verbatim."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from frontdesk.db.models import Record
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.knowledge.versions import active_version_id
from frontdesk.tenancy.pack import Doctor, Language, Location, OrientationEntry, Price, Service

_LABELS: dict[str, dict[str, str]] = {
    "location": {
        "sq": "Adresa dhe kontaktet",
        "it": "Indirizzo e contatti",
        "en": "Address and contacts",
    },
    "address": {"sq": "Adresa", "it": "Indirizzo", "en": "Address"},
    "phone": {"sq": "Telefoni", "it": "Telefono", "en": "Phone"},
    "email": {"sq": "Email", "it": "Email", "en": "Email"},
    "hours": {"sq": "Orari", "it": "Orari", "en": "Opening hours"},
    "price": {"sq": "Çmimi", "it": "Prezzo", "en": "Price"},
    "service": {"sq": "Shërbimi", "it": "Servizio", "en": "Service"},
    "doctor": {"sq": "Mjeku", "it": "Medico", "en": "Doctor"},
    "doctors": {"sq": "Mjekët", "it": "Medici", "en": "Doctors"},
    "duration": {"sq": "Kohëzgjatja", "it": "Durata", "en": "Duration"},
    "minutes": {"sq": "minuta", "it": "minuti", "en": "minutes"},
    "requires_consult": {
        "sq": "Kërkon konsultë paraprake",
        "it": "Richiede un consulto preliminare",
        "en": "Requires a prior consultation",
    },
    "orientation": {
        "sq": "Shërbimet që merren me këtë",
        "it": "Servizi che se ne occupano",
        "en": "Services that handle this",
    },
}

_DAYS: dict[str, dict[str, str]] = {
    "mon": {"sq": "Hën", "it": "lun", "en": "Mon"},
    "tue": {"sq": "Mar", "it": "mar", "en": "Tue"},
    "wed": {"sq": "Mër", "it": "mer", "en": "Wed"},
    "thu": {"sq": "Enj", "it": "gio", "en": "Thu"},
    "fri": {"sq": "Pre", "it": "ven", "en": "Fri"},
    "sat": {"sq": "Sht", "it": "sab", "en": "Sat"},
    "sun": {"sq": "Die", "it": "dom", "en": "Sun"},
}


@dataclass(frozen=True)
class Catalog:
    business_id: str
    version_id: uuid.UUID
    services: dict[str, dict[str, str]]
    doctors: dict[str, str]
    problem_categories: dict[str, dict[str, list[str]]]


def load_catalog(session: Session, business_id: str) -> Catalog | None:
    version_id = active_version_id(session, business_id)
    if version_id is None:
        return None
    services = [Service.model_validate(r.payload) for r in _rows(session, version_id, "service")]
    doctors = [Doctor.model_validate(r.payload) for r in _rows(session, version_id, "doctor")]
    entries = [
        OrientationEntry.model_validate(r.payload)
        for r in _rows(session, version_id, "orientation")
    ]
    return Catalog(
        business_id=business_id,
        version_id=version_id,
        services={
            s.id: {str(language): name for language, name in s.names.items()} for s in services
        },
        doctors={d.id: d.name for d in doctors},
        problem_categories={e.id: {k: list(v) for k, v in e.labels.items()} for e in entries},
    )


def localized_days(days: str, language: Language) -> str:
    return "-".join(_DAYS[code][language] for code in days.split("-"))


def field_label(key: str, language: Language) -> str:
    return _LABELS[key][language]


def location_evidence(session: Session, business_id: str, language: Language) -> Evidence | None:
    version_id = active_version_id(session, business_id)
    row = _row(session, version_id, "location", "main") if version_id else None
    if version_id is None or row is None:
        return None
    location = Location.model_validate(row.payload)
    hours = "; ".join(
        f"{localized_days(h.days, language)} {h.opens}-{h.closes}" for h in location.opening_hours
    )
    values = {
        "address": location.address,
        "phone": location.phone,
        "email": location.email,
        "hours": hours,
    }
    content = "\n".join(f"{_LABELS[key][language]}: {value}" for key, value in values.items())
    return _record("location:main", version_id, _LABELS["location"][language], content, values)


def price_evidence(
    session: Session, business_id: str, service_id: str, language: Language, today: date
) -> Evidence | None:
    version_id = active_version_id(session, business_id)
    if version_id is None:
        return None
    price_row = _row(session, version_id, "price", service_id)
    service_row = _row(session, version_id, "service", service_id)
    if price_row is None or service_row is None:
        return None
    price = Price.model_validate(price_row.payload)
    if (price.valid_from and price.valid_from > today) or (
        price.valid_until and price.valid_until < today
    ):
        return None
    service = Service.model_validate(service_row.payload)
    # "40" up to "40.00" is one price, not a range: compare the ends as numbers
    is_range = price.max_amount is not None and Decimal(price.max_amount) != Decimal(price.amount)
    amount = (
        f"{price.amount}-{price.max_amount} {price.currency}"
        if is_range
        else f"{price.amount} {price.currency}"
    )
    condition = price.conditions.get(language)
    content = f"{service.names.get(language, service.id)}: {amount}"
    if condition:
        content += f" ({condition})"
    return _record(
        f"price:{service_id}", version_id, _LABELS["price"][language], content, {"price": amount}
    )


def service_evidence(
    session: Session, business_id: str, service_ids: Sequence[str], language: Language
) -> list[Evidence]:
    version_id = active_version_id(session, business_id)
    if version_id is None:
        return []
    doctors = [Doctor.model_validate(r.payload) for r in _rows(session, version_id, "doctor")]
    found: list[Evidence] = []
    for service_id in service_ids:
        row = _row(session, version_id, "service", service_id)
        if row is None:
            continue
        service = Service.model_validate(row.payload)
        lines = [
            service.names.get(language, service.id),
            f"{_LABELS['duration'][language]}: {service.duration_minutes} {_LABELS['minutes'][language]}",
        ]
        if service.requires_consult:
            lines.append(_LABELS["requires_consult"][language])
        names = [d.name for d in doctors if service.id in d.service_ids]
        if names:
            lines.append(f"{_LABELS['doctors'][language]}: " + ", ".join(names))
        found.append(
            _record(
                f"service:{service.id}",
                version_id,
                _LABELS["service"][language],
                "\n".join(lines),
                {},
            )
        )
    return found


def doctor_evidence(
    session: Session, business_id: str, doctor_ids: Sequence[str], language: Language
) -> list[Evidence]:
    version_id = active_version_id(session, business_id)
    if version_id is None:
        return []
    services = {
        s.id: s
        for s in (Service.model_validate(r.payload) for r in _rows(session, version_id, "service"))
    }
    found: list[Evidence] = []
    for doctor_id in doctor_ids:
        row = _row(session, version_id, "doctor", doctor_id)
        if row is None:
            continue
        doctor = Doctor.model_validate(row.payload)
        offered = [services[s].names.get(language, s) for s in doctor.service_ids if s in services]
        content = f"{doctor.name}: " + ", ".join(offered)
        found.append(
            _record(f"doctor:{doctor.id}", version_id, _LABELS["doctor"][language], content, {})
        )
    return found


def orientation_evidence(
    session: Session, business_id: str, category_id: str, language: Language
) -> Evidence | None:
    version_id = active_version_id(session, business_id)
    row = _row(session, version_id, "orientation", category_id) if version_id else None
    if version_id is None or row is None:
        return None
    entry = OrientationEntry.model_validate(row.payload)
    services = {
        s.id: s
        for s in (Service.model_validate(r.payload) for r in _rows(session, version_id, "service"))
    }
    names = [services[s].names.get(language, s) for s in entry.service_ids if s in services]
    labels = ", ".join(entry.labels.get(language, []))
    content = f"{labels}: " + ", ".join(names)
    return _record(
        f"orientation:{entry.id}",
        version_id,
        _LABELS["orientation"][language],
        content,
        {"services": ", ".join(names)},
    )


def _rows(session: Session, version_id: uuid.UUID, kind: str) -> list[Record]:
    return list(
        session.scalars(
            select(Record)
            .where(Record.knowledge_version_id == version_id, Record.kind == kind)
            .order_by(Record.record_key)
        )
    )


def _row(session: Session, version_id: uuid.UUID, kind: str, key: str) -> Record | None:
    return session.scalar(
        select(Record).where(
            Record.knowledge_version_id == version_id, Record.kind == kind, Record.record_key == key
        )
    )


def _record(
    source_ref: str, version_id: uuid.UUID, title: str, content: str, values: dict[str, str]
) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.RECORD,
        source_ref=source_ref,
        version=str(version_id),
        title=title,
        content=content,
        values=values,
    )
