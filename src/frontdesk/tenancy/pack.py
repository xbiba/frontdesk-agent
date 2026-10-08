"""Business pack: everything that makes one business different, kept outside the code (spec §3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["sq", "it", "en"]

_ID = r"^[a-z0-9][a-z0-9-]{0,62}$"
_TIME = r"^([01]\d|2[0-3]):[0-5]\d$"
_DAY = "(mon|tue|wed|thu|fri|sat|sun)"
_DAYS = rf"^{_DAY}(-{_DAY})?$"
_AMOUNT = r"^\d+(\.\d{2})?$"


class Profile(StrEnum):
    RECEPTIONIST = "receptionist"
    SUPPORT_SALES = "support-sales"


class FixedTextKey(StrEnum):
    DISCLOSURE = "disclosure"
    GREETING = "greeting"
    EMERGENCY = "emergency"
    CLINICAL = "clinical"
    PERSONAL_DATA = "personal_data"
    HANDOFF = "handoff"
    NO_INFORMATION = "no_information"
    CLARIFICATION = "clarification"
    ATTACHMENT = "attachment"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Features(Frozen):
    orientation: bool = False
    order_status: bool = False


class CalendarIntegration(Frozen):
    credentials_ref: str = Field(min_length=1)
    calendars: dict[str, str] = Field(min_length=1)


class Integrations(Frozen):
    calendar: CalendarIntegration | None = None


class OpeningHours(Frozen):
    days: str = Field(pattern=_DAYS)
    opens: str = Field(pattern=_TIME)
    closes: str = Field(pattern=_TIME)


class Location(Frozen):
    address: str
    phone: str
    email: str
    opening_hours: list[OpeningHours] = Field(min_length=1)


class Service(Frozen):
    id: str = Field(pattern=_ID)
    names: dict[Language, str]
    duration_minutes: int = Field(gt=0, le=480)
    bookable_online: bool = False
    requires_consult: bool = False


class Doctor(Frozen):
    id: str = Field(pattern=_ID)
    name: str
    service_ids: list[str] = Field(min_length=1)


class Price(Frozen):
    service_id: str
    currency: Literal["EUR", "ALL"]
    amount: str = Field(pattern=_AMOUNT)
    max_amount: str | None = Field(default=None, pattern=_AMOUNT)
    conditions: dict[Language, str] = Field(default_factory=dict)
    valid_from: date | None = None
    valid_until: date | None = None


class OrientationEntry(Frozen):
    id: str = Field(pattern=_ID)
    labels: dict[Language, list[str]]
    service_ids: list[str] = Field(min_length=1)


class Order(Frozen):
    reference: str = Field(pattern=r"^ORD-\d{4,8}$")
    postal_code: str = Field(pattern=r"^\d{4,6}$")
    status: dict[Language, str]
    eta: date | None = None


class KnowledgeDocument(Frozen):
    key: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    language: Language
    body: str = Field(min_length=1)
    public_url: str | None = None
    valid_from: date | None = None
    valid_until: date | None = None


class BusinessPack(Frozen):
    id: str = Field(pattern=_ID)
    name: str
    profile: Profile
    isolation_level: Literal["T1", "T2", "T3"]
    default_language: Language
    languages: list[Language] = Field(min_length=1)
    timezone: str
    features: Features = Features()
    integrations: Integrations = Integrations()
    fixed_texts: dict[Language, dict[FixedTextKey, str]]
    location: Location
    services: list[Service] = Field(default_factory=list)
    doctors: list[Doctor] = Field(default_factory=list)
    prices: list[Price] = Field(default_factory=list)
    orientation: list[OrientationEntry] = Field(default_factory=list)
    orders: list[Order] = Field(default_factory=list)
    documents: list[KnowledgeDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> BusinessPack:
        problems = [
            *_language_problems(self),
            *_reference_problems(self),
            *_profile_problems(self),
        ]
        if problems:
            raise ValueError(f"business pack '{self.id}': " + "; ".join(problems))
        return self


def _language_problems(pack: BusinessPack) -> list[str]:
    problems: list[str] = []
    if pack.default_language not in pack.languages:
        problems.append(f"default language '{pack.default_language}' is not in languages")
    for language in pack.languages:
        texts = pack.fixed_texts.get(language, {})
        problems += [
            f"missing fixed text '{key}' for language '{language}'"
            for key in FixedTextKey
            if not texts.get(key, "").strip()
        ]
    problems += [
        f"service '{service.id}' has no name in '{language}'"
        for service in pack.services
        for language in pack.languages
        if language not in service.names
    ]
    problems += [
        f"document '{document.key}' is in unsupported language '{document.language}'"
        for document in pack.documents
        if document.language not in pack.languages
    ]
    return problems


def _reference_problems(pack: BusinessPack) -> list[str]:
    groups = {
        "service": [s.id for s in pack.services],
        "doctor": [d.id for d in pack.doctors],
        "orientation": [o.id for o in pack.orientation],
        "document": [d.key for d in pack.documents],
        "price": [p.service_id for p in pack.prices],
        "order": [o.reference for o in pack.orders],
    }
    problems = [
        f"duplicate {kind} id '{value}'"
        for kind, values in groups.items()
        for value in _duplicates(values)
    ]
    services = set(groups["service"])
    problems += [
        f"doctor '{doctor.id}' references unknown service '{service_id}'"
        for doctor in pack.doctors
        for service_id in doctor.service_ids
        if service_id not in services
    ]
    problems += [
        f"price references unknown service '{price.service_id}'"
        for price in pack.prices
        if price.service_id not in services
    ]
    problems += [
        f"orientation '{entry.id}' references unknown service '{service_id}'"
        for entry in pack.orientation
        for service_id in entry.service_ids
        if service_id not in services
    ]
    return problems


def _profile_problems(pack: BusinessPack) -> list[str]:
    problems: list[str] = []
    receptionist = pack.profile is Profile.RECEPTIONIST
    if receptionist and not pack.services:
        problems.append("profile 'receptionist' needs at least one service")
    if pack.features.orientation and not (receptionist and pack.orientation):
        problems.append("feature 'orientation' needs profile 'receptionist' and an orientation map")
    if pack.features.order_status and not (pack.profile is Profile.SUPPORT_SALES and pack.orders):
        problems.append("feature 'order_status' needs profile 'support-sales' and orders")
    calendar = pack.integrations.calendar
    if calendar is not None:
        if not receptionist:
            problems.append("calendar integration needs profile 'receptionist'")
        doctors = {d.id for d in pack.doctors}
        problems += [
            f"calendar references unknown doctor '{doctor_id}'"
            for doctor_id in calendar.calendars
            if doctor_id not in doctors
        ]
    return problems


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)
