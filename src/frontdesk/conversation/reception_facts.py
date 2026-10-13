"""The clinic's facts as the reception agent reads them (Albi, 2026-10-12): the records of the
active knowledge version, in the conversation's language. The agent may state these and nothing
else; its tools add the free times and the approved documents."""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from frontdesk.knowledge.facts import (
    Catalog,
    department_names,
    doctor_evidence,
    location_evidence,
    orientation_evidence,
    price_evidence,
    service_evidence,
)
from frontdesk.tenancy.pack import BusinessPack, Language


def render_facts(
    session: Session,
    pack: BusinessPack,
    catalog: Catalog,
    language: Language,
    today: date,
    *,
    chat_booking: bool,
) -> str:
    """Location and contacts, services, prices, doctors, the problem map, and what can be booked
    in the chat; one section each, in the language's own words for every field."""
    sections = [f"Clinic: {pack.name}"]
    location = location_evidence(session, pack.id, language)
    if location is not None:
        sections.append(f"Location, contacts and opening hours:\n{location.content}")
    services = service_evidence(session, pack.id, sorted(catalog.services), language)
    if services:
        lines = (item.content.replace("\n", "; ") for item in services)
        sections.append("Services:\n" + "\n".join(f"- {line}" for line in lines))
    prices = [price_evidence(session, pack.id, item, language, today) for item in catalog.services]
    if any(prices):
        lines = (item.content for item in prices if item is not None)
        sections.append("Prices:\n" + "\n".join(f"- {line}" for line in lines))
    departments = department_names(session, pack.id, language)
    if departments:
        sections.append(
            "Departments of the clinic (by name only: it has them, and more services than those "
            "priced above; for their prices, times and doctors, reception):\n"
            + "\n".join(f"- {name}" for name in departments)
        )
    doctors = doctor_evidence(session, pack.id, sorted(catalog.doctors), language)
    if doctors:
        sections.append("Doctors:\n" + "\n".join(f"- {item.content}" for item in doctors))
    problems = [
        orientation_evidence(session, pack.id, item, language)
        for item in sorted(catalog.problem_categories)
    ]
    if any(problems):
        lines = (item.content for item in problems if item is not None)
        sections.append(
            "Which service for which problem (the only health guidance you may give):\n"
            + "\n".join(f"- {line}" for line in lines)
        )
    sections.append(_bookable(pack, catalog, language, chat_booking=chat_booking))
    return "\n\n".join(sections)


def _bookable(
    pack: BusinessPack, catalog: Catalog, language: Language, *, chat_booking: bool
) -> str:
    offered = [
        f"{catalog.services.get(service, {}).get(language, service)} "
        f"({catalog.doctors.get(doctor, doctor)}; doctor_id {doctor}, service_id {service})"
        for doctor, services in pack.bookable_doctors().items()
        for service in services
    ]
    if not chat_booking or not offered:
        return "Bookings: by phone with reception only."
    return "Bookable here in the chat, with the booking tools:\n" + "\n".join(
        f"- {line}" for line in offered
    )
