"""The business's information as the email drafts read it, written from the pack (design §3.3):
the index of its departments and services, then the facts and the documents valid today.

Every sentence a draft cites must be found, word for word, in this text (frontdesk.email.checks).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final

from frontdesk.tenancy.pack import BusinessPack, FixedTextKey, KnowledgeDocument, Language, Price

# The pack's texts that also hold for an email: what to do in an emergency, no advice by message,
# how to reach a person and how to book.
EMAIL_FIXED_TEXTS: Final = (
    FixedTextKey.EMERGENCY,
    FixedTextKey.CLINICAL,
    FixedTextKey.HANDOFF,
    FixedTextKey.NEXT_STEP,
)


def _names(names: Mapping[Language, str]) -> str:
    return " / ".join(names.values())


def index_text(pack: BusinessPack) -> str:
    """What the business has, by name: a department without a price is still a department."""
    lines = [
        "INDEX (departments and services; prices, doctors and hours only where the "
        "information gives them):"
    ]
    lines += [f"- Department: {_names(item.names)}" for item in pack.departments]
    lines += [
        f"- Service with price and booking: {_names(item.names)} ({item.duration_minutes} minutes)"
        for item in pack.services
    ]
    return "\n".join(lines)


def knowledge_text(pack: BusinessPack, today: date) -> str:
    """The facts of the pack and its documents valid on `today`, as plain text."""
    location = pack.location
    parts = [f"# {pack.name}", f"Address: {location.address}", f"Phone: {location.phone}"]
    if location.whatsapp:
        parts.append(f"WhatsApp: {location.whatsapp}")
    parts.append(f"Email: {location.email}")
    parts += [
        f"Opening hours: {hours.days} {hours.opens}-{hours.closes}"
        for hours in location.opening_hours
    ]
    if pack.departments:
        parts.append("\n## Departments")
        parts += [f"- {_names(item.names)}" for item in pack.departments]
    if pack.services:
        parts.append("\n## Services")
        parts += [_service_line(pack, item.id, today) for item in pack.services]
    parts.append("\n## Fixed texts")
    for language in pack.languages:
        texts = pack.fixed_texts.get(language, {})
        parts += [f"- ({language}) {texts[key]}" for key in EMAIL_FIXED_TEXTS if key in texts]
    documents = [item for item in pack.documents if _valid(item, today)]
    if documents:
        parts.append("\n## Documents")
        parts += [f"\n### {item.title} ({item.language})\n{item.body}" for item in documents]
    return "\n".join(parts)


def _service_line(pack: BusinessPack, service_id: str, today: date) -> str:
    service = next(item for item in pack.services if item.id == service_id)
    line = f"- {_names(service.names)}: {service.duration_minutes} minutes"
    prices = [
        _price(item)
        for item in pack.prices
        if item.service_id == service_id and _current(item, today)
    ]
    if prices:
        line += "; price " + "; ".join(prices)
    doctors = [doctor.name for doctor in pack.doctors if service_id in doctor.service_ids]
    if doctors:
        line += "; doctors: " + ", ".join(doctors)
    return line


def _price(price: Price) -> str:
    amount = price.amount if price.max_amount is None else f"{price.amount}-{price.max_amount}"
    conditions = " / ".join(price.conditions.values())
    return f"{amount} {price.currency}" + (f" ({conditions})" if conditions else "")


def _current(price: Price, today: date) -> bool:
    return (price.valid_from is None or price.valid_from <= today) and (
        price.valid_until is None or today <= price.valid_until
    )


def _valid(document: KnowledgeDocument, today: date) -> bool:
    return (document.valid_from is None or document.valid_from <= today) and (
        document.valid_until is None or today <= document.valid_until
    )
