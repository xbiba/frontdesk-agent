"""Business pack: everything that makes one business different, kept outside the code (ADR-0007)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from enum import StrEnum
from functools import cache
from typing import Annotated, Final, Literal
from zoneinfo import available_timezones

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Language = Literal["sq", "it", "en"]

_ID = r"^[a-z0-9][a-z0-9-]{0,62}$"
_TIME = r"^([01]\d|2[0-3]):[0-5]\d$"
_DAY = "(mon|tue|wed|thu|fri|sat|sun)"
_DAYS = rf"^{_DAY}(-{_DAY})?$"
_AMOUNT = r"^\d+(\.\d{2})?$"


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


# Text that is shown to people (contacts, names, an order's status) must say something.
NonBlankStr = Annotated[str, AfterValidator(_not_blank)]


@cache
def _iana_zone_names() -> frozenset[str]:
    # Scanning the zone files takes about ten milliseconds, and every pack load asks.
    return frozenset(available_timezones())


def _known_time_zone(value: str) -> str:
    # Matched against the list of names, not by opening a zone file: a Mac opens "europe/tirane"
    # because its file system ignores case, while the Linux container image does not.
    if value not in _iana_zone_names():
        raise ValueError(f"unknown time zone '{value}' (use an IANA name such as Europe/Tirane)")
    return value


# The business-local day decides which prices and documents apply, so the zone must exist.
TimeZoneName = Annotated[str, AfterValidator(_known_time_zone)]


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
    NEXT_STEP = "next_step"


# A pack may leave these out. The next step (how to book) ends an answer once per conversation;
# without one, an answer ends with the answer.
OPTIONAL_FIXED_TEXTS: Final = frozenset({FixedTextKey.NEXT_STEP})


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
    address: NonBlankStr
    phone: NonBlankStr
    email: NonBlankStr
    opening_hours: list[OpeningHours] = Field(min_length=1)


class Service(Frozen):
    id: str = Field(pattern=_ID)
    names: dict[Language, NonBlankStr]
    duration_minutes: int = Field(gt=0, le=480)
    bookable_online: bool = False
    requires_consult: bool = False


class Doctor(Frozen):
    id: str = Field(pattern=_ID)
    name: NonBlankStr
    service_ids: list[str] = Field(min_length=1)


class Price(Frozen):
    service_id: str
    currency: Literal["EUR", "ALL"]
    amount: str = Field(pattern=_AMOUNT)
    max_amount: str | None = Field(default=None, pattern=_AMOUNT)
    conditions: dict[Language, str] = Field(default_factory=dict)
    valid_from: date | None = None
    valid_until: date | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> Price:
        problems: list[str] = []
        if self.max_amount is not None and Decimal(self.max_amount) < Decimal(self.amount):
            problems.append(f"max_amount {self.max_amount} is below amount {self.amount}")
        problems += _window_problems(self.valid_from, self.valid_until)
        if problems:
            raise ValueError(f"price for '{self.service_id}': " + "; ".join(problems))
        return self


class OrientationMap(Frozen):
    """Which version of the orientation map the bot uses and the doctor who approved it (spec §7;
    final review P94). It goes into the knowledge version with the map, so the approval report
    shows it and a new version or approver needs a new approval."""

    version: NonBlankStr
    approved_by: NonBlankStr
    approved_on: date


class OrientationEntry(Frozen):
    id: str = Field(pattern=_ID)
    labels: dict[Language, list[str]]
    service_ids: list[str] = Field(min_length=1)


class Order(Frozen):
    reference: str = Field(pattern=r"^ORD-\d{4,8}$")
    postal_code: str = Field(pattern=r"^\d{4,6}$")
    status: dict[Language, NonBlankStr] = Field(min_length=1)
    eta: date | None = None


class KnowledgeDocument(Frozen):
    key: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    language: Language
    body: str = Field(min_length=1)
    public_url: str | None = None
    valid_from: date | None = None
    valid_until: date | None = None

    @model_validator(mode="after")
    def _check_window(self) -> KnowledgeDocument:
        problems = _window_problems(self.valid_from, self.valid_until)
        if problems:
            raise ValueError(f"document '{self.key}': " + "; ".join(problems))
        return self


class BusinessPack(Frozen):
    id: str = Field(pattern=_ID)
    name: NonBlankStr
    profile: Profile
    isolation_level: Literal["T1", "T2", "T3"]
    default_language: Language
    languages: list[Language] = Field(min_length=1)
    timezone: TimeZoneName
    features: Features = Features()
    integrations: Integrations = Integrations()
    fixed_texts: dict[Language, dict[FixedTextKey, str]]
    location: Location
    services: list[Service] = Field(default_factory=list)
    doctors: list[Doctor] = Field(default_factory=list)
    prices: list[Price] = Field(default_factory=list)
    orientation: list[OrientationEntry] = Field(default_factory=list)
    orientation_map: OrientationMap | None = None
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
            if (key not in OPTIONAL_FIXED_TEXTS or key in texts) and not texts.get(key, "").strip()
        ]
        if not _gives_a_contact(texts.get(FixedTextKey.HANDOFF, ""), pack.location):
            problems.append(
                f"fixed text '{FixedTextKey.HANDOFF}' for language '{language}' "
                "must contain the location phone or email"
            )
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
    problems += [
        f"orientation '{entry.id}' has no labels in '{language}'"
        for entry in pack.orientation
        for language in pack.languages
        if not any(label.strip() for label in entry.labels.get(language, []))
    ]
    problems += [
        f"price for '{price.service_id}' has no condition in '{language}'"
        for price in pack.prices
        if price.conditions  # a price with no conditions at all stays valid
        for language in pack.languages
        if not price.conditions.get(language, "").strip()
    ]
    return problems


def _gives_a_contact(text: str, location: Location) -> bool:
    """Whether a handoff text names how to reach a person. A blank text is reported on its own."""
    text = text.strip().casefold()
    return not text or any(
        contact.strip().casefold() in text for contact in (location.phone, location.email)
    )


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
    if pack.features.orientation and pack.orientation_map is None:
        problems.append(
            "feature 'orientation' needs orientation_map: the map's version and approving doctor"
        )
    if pack.orientation_map is not None and not pack.orientation:
        problems.append("orientation_map describes a map, but data/orientation.yaml has no entries")
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


def _window_problems(valid_from: date | None, valid_until: date | None) -> list[str]:
    if valid_from is not None and valid_until is not None and valid_from > valid_until:
        return [f"valid_from {valid_from} is after valid_until {valid_until}"]
    return []
