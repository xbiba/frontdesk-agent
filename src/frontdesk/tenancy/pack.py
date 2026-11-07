"""Business pack: everything that makes one business different, kept outside the code (ADR-0007)."""

from __future__ import annotations

import re
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
    NO_SLOTS = "no_slots"
    BOOKING_PRIVACY = "booking_privacy"


# A pack may leave these out. The next step (how to book) ends an answer once per conversation;
# without one, an answer ends with the answer. Without `no_slots`, a booking question with no free
# slot gets `no_information`.
# `booking_privacy` ends a booking made in the chat: where the privacy notice is, a link (Albi,
# 2026-10-12: nothing before the name, for a clean conversation). A pack that books from the chat
# must give it (M5).
OPTIONAL_FIXED_TEXTS: Final = frozenset(
    {FixedTextKey.NEXT_STEP, FixedTextKey.NO_SLOTS, FixedTextKey.BOOKING_PRIVACY}
)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Features(Frozen):
    orientation: bool = False
    order_status: bool = False


class CalendarIntegration(Frozen):
    """One calendar per bookable doctor (ADR-0003). The doctor's hours are the events titled with
    the marker; the other events are appointments, read only as busy time."""

    credentials_ref: str = Field(min_length=1)
    calendars: dict[str, str] = Field(min_length=1)
    availability_marker: NonBlankStr = "Disponibile"
    horizon_days: int = Field(default=14, ge=1, le=60)
    min_notice_minutes: int = Field(default=120, ge=0, le=10080)
    buffer_minutes: int = Field(default=0, ge=0, le=120)
    max_options: int = Field(default=3, ge=1, le=5)
    # The clinic confirms each appointment (cal.com's "requires confirmation", Albi, 2026-10-13):
    # the chat sends a request, and never says an appointment is confirmed.
    requires_confirmation: bool = False
    # M5: the patient books one of the free slots from the chat, at once (Albi, 2026-10-12).
    chat_booking: bool = False


class Integrations(Frozen):
    calendar: CalendarIntegration | None = None


class ChannelPolicy(Frozen):
    """What one channel lets the business do (ADR-0010). It can only switch features off: a turn
    has the business's features AND the channel's."""

    chat_booking: bool = True
    order_status: bool = True
    orientation: bool = True


_ORIGIN: Final = re.compile(r"^(https://[a-z0-9.-]+|http://(localhost|127\.0\.0\.1))(:\d{1,5})?$")
_HTTPS_URL: Final = re.compile(r"^https://[^\s\"'<>]+$")


def _origin(value: str) -> str:
    """A site's origin as a browser sends it: https, or plain http on this computer only."""
    if _ORIGIN.fullmatch(value) is None:
        raise ValueError(f"not an origin (https://host[:port]): {value!r}")
    return value


def _https_url(value: str) -> str:
    if _HTTPS_URL.fullmatch(value) is None:
        raise ValueError(f"privacy link must be an https URL: {value!r}")
    return value


Origin = Annotated[str, AfterValidator(_origin)]
HttpsUrl = Annotated[str, AfterValidator(_https_url)]


class WebTexts(Frozen):
    """The widget's own words in one language. The replies are the engine's."""

    title: NonBlankStr
    open: NonBlankStr  # the label of the button that opens the chat
    close: NonBlankStr
    placeholder: NonBlankStr
    send: NonBlankStr
    # Always above the input (design §4): automatic assistant, no personal or health data, the
    # reception's number. The widget adds the privacy link.
    notice: NonBlankStr
    person: NonBlankStr  # "Talk to a person": the reception's phone or WhatsApp
    privacy: NonBlankStr  # the privacy link's words


_COLOUR: Final = r"^#[0-9a-fA-F]{6}$"


class WebStyle(Frozen):
    """A client's look for the widget, from its own pack (Albi, 2026-11-07): the engine's widget is
    plain and neutral until the client styles it. Each value becomes a CSS custom property inside
    the widget's Shadow DOM, so it can change the widget and nothing else of the site."""

    accent: str | None = Field(default=None, pattern=_COLOUR)  # header, buttons, own messages
    on_accent: str | None = Field(default=None, pattern=_COLOUR)  # text on the accent
    # A font-family list of names only: no url(), no braces, nothing CSS could run.
    font: str | None = Field(default=None, pattern=r"^[A-Za-z0-9 ,'\"-]{1,120}$")
    position: Literal["right", "left"] = "right"
    logo_url: HttpsUrl | None = None


class WebChannel(Frozen):
    """The chat on the business's site (design of 2026-11-07). Only the listed sites may embed it
    (CORS and the Origin check). No login, so no booking and no order lookup: its policy turns
    both off. The privacy link may wait while the widget runs in tester mode."""

    origins: list[Origin] = Field(min_length=1)
    name: NonBlankStr
    privacy_url: HttpsUrl | None = None
    texts: dict[Language, WebTexts]
    style: WebStyle | None = None

    def allows(self, origin: str | None) -> bool:
        return origin is not None and origin in self.origins

    def policy(self) -> ChannelPolicy:
        return ChannelPolicy(chat_booking=False, order_status=False)


class Channels(Frozen):
    """The channels the business answers on, each with its policy (M4: Instagram). WhatsApp (W1)
    is the reception's: booking requests to decide, no model. The web chat (2026-11-07) has its
    fixed policy."""

    instagram: ChannelPolicy | None = None
    whatsapp: ChannelPolicy | None = None
    web: WebChannel | None = None


_ADDRESS: Final = re.compile(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+")


def _address(value: str) -> str:
    """An email address, lower case; refused unless it looks like one."""
    found = _ADDRESS.fullmatch(value.strip())
    if found is None:
        raise ValueError(f"not an email address: {value!r}")
    return found.group(0).lower()


def address_of(sender: str) -> str:
    """The address in a From header ("Name <a@b.c>" or "a@b.c"), lower case; "" when none."""
    found = _ADDRESS.search(sender)
    return found.group(0).lower() if found else ""


Address = Annotated[str, AfterValidator(_address)]


class EmailChannel(Frozen):
    """Reply drafts for the business's mailbox (email drafts, 2026-11-07). The engine never sends:
    it writes a draft in the conversation, for a person to check and send.

    Tester mode, the default, drafts only for the senders in `testers`: a real mailbox starts there
    and opens to everyone (`testers_only: false`) when the business's privacy notice names the
    model's provider (design §9, decision 3)."""

    mailbox: Address
    signature: NonBlankStr = "[FIRMA]"
    # The words of the placeholder for what the information does not say: "[DA COMPLETARE: ...]".
    todo: NonBlankStr = "DA COMPLETARE"
    testers_only: bool = True
    testers: list[Address] = Field(default_factory=list)

    def drafts_for(self, sender: str) -> bool:
        """Whether a message from `sender` (a From header) may get a draft."""
        return not self.testers_only or address_of(sender) in self.testers


class OpeningHours(Frozen):
    days: str = Field(pattern=_DAYS)
    opens: str = Field(pattern=_TIME)
    closes: str = Field(pattern=_TIME)


class Location(Frozen):
    address: NonBlankStr
    phone: NonBlankStr
    email: NonBlankStr
    # Reception's WhatsApp, when it has one: the agent gives it with the phone (Albi, 2026-10-12).
    whatsapp: NonBlankStr | None = None
    # A clinic may have no fixed hours: the first clinic is open when a doctor has availability in its
    # calendar (Albi, 2026-10-13). An all-day calendar block needs them, to know its hours.
    opening_hours: list[OpeningHours] = Field(default_factory=list)


class Service(Frozen):
    id: str = Field(pattern=_ID)
    names: dict[Language, NonBlankStr]
    duration_minutes: int = Field(gt=0, le=480)
    bookable_online: bool = False
    requires_consult: bool = False


class Department(Frozen):
    """A department of the clinic, by name only: the agent may say the clinic has it, and gives
    reception for its prices, times and doctors (Albi, 2026-10-13)."""

    id: str = Field(pattern=_ID)
    names: dict[Language, NonBlankStr]


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
    channels: Channels = Channels()
    email: EmailChannel | None = None
    fixed_texts: dict[Language, dict[FixedTextKey, str]]
    location: Location
    services: list[Service] = Field(default_factory=list)
    departments: list[Department] = Field(default_factory=list)
    doctors: list[Doctor] = Field(default_factory=list)
    prices: list[Price] = Field(default_factory=list)
    orientation: list[OrientationEntry] = Field(default_factory=list)
    orientation_map: OrientationMap | None = None
    orders: list[Order] = Field(default_factory=list)
    documents: list[KnowledgeDocument] = Field(default_factory=list)

    def bookable_doctors(self) -> dict[str, tuple[str, ...]]:
        """Each doctor with a calendar, with the services a patient can book online."""
        calendar = self.integrations.calendar
        if calendar is None:
            return {}
        bookable = {service.id for service in self.services if service.bookable_online}
        return {
            doctor.id: tuple(item for item in doctor.service_ids if item in bookable)
            for doctor in self.doctors
            if doctor.id in calendar.calendars
        }

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
        problems += [
            f"fixed text '{key}' for language '{language}' must contain the location phone or email"
            for key in (FixedTextKey.HANDOFF, FixedTextKey.NEXT_STEP)
            if (key is FixedTextKey.HANDOFF or key in texts)
            and not _gives_a_contact(texts.get(key, ""), pack.location)
        ]
    problems += _web_problems(pack)
    problems += [
        f"service '{service.id}' has no name in '{language}'"
        for service in pack.services
        for language in pack.languages
        if language not in service.names
    ]
    problems += [
        f"department '{department.id}' has no name in '{language}'"
        for department in pack.departments
        for language in pack.languages
        if language not in department.names
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


def _web_problems(pack: BusinessPack) -> list[str]:
    web = pack.channels.web
    if web is None:
        return []
    problems = [
        f"web channel has no texts in '{language}'"
        for language in pack.languages
        if language not in web.texts
    ]
    problems += [
        f"web notice in '{language}' must contain the location phone or email"
        for language, texts in web.texts.items()
        if not _gives_a_contact(texts.notice, pack.location)
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
        "department": [d.id for d in pack.departments],
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
        problems += [
            f"calendar for '{doctor_id}' has no bookable service"
            for doctor_id, services in pack.bookable_doctors().items()
            if not services
        ]
        if calendar.chat_booking:
            problems += [
                f"chat booking needs the fixed text 'booking_privacy' in '{language}'"
                for language in pack.languages
                if FixedTextKey.BOOKING_PRIVACY not in pack.fixed_texts.get(language, {})
            ]
    return problems


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _window_problems(valid_from: date | None, valid_until: date | None) -> list[str]:
    if valid_from is not None and valid_until is not None and valid_from > valid_until:
        return [f"valid_from {valid_from} is after valid_until {valid_until}"]
    return []
