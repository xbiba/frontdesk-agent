"""Closed vocabularies the classifier must answer with (ADR-0006)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from frontdesk.tenancy.pack import Profile


class Category(StrEnum):
    ADMINISTRATIVE = "administrative"
    ORIENTATION = "orientation"
    CLINICAL = "clinical"
    UNCERTAIN = "uncertain"
    EMERGENCY = "emergency"


class Intent(StrEnum):
    GREETING = "greeting"
    HUMAN = "human"
    LOCATION = "location"
    HOURS = "hours"
    CONTACT = "contact"
    PRICE = "price"
    SERVICE_INFO = "service_info"
    DOCTOR_INFO = "doctor_info"
    BOOKING = "booking"
    ORDER_STATUS = "order_status"
    GENERAL_QUESTION = "general_question"
    OTHER = "other"


class ClarifyField(StrEnum):
    """What a clarifying question may ask for; the question itself is written by code."""

    SERVICE = "service"
    DOCTOR = "doctor"
    ORDER_DETAILS = "order_details"


class SlotPart(StrEnum):
    """The part of the day a patient asks for: morning before 13:00, afternoon from 13:00."""

    MORNING = "morning"
    AFTERNOON = "afternoon"


ADMINISTRATIVE_INTENTS: Final = frozenset(
    {
        Intent.LOCATION,
        Intent.HOURS,
        Intent.CONTACT,
        Intent.PRICE,
        Intent.SERVICE_INFO,
        Intent.DOCTOR_INFO,
        Intent.BOOKING,
        Intent.ORDER_STATUS,
        Intent.GENERAL_QUESTION,
    }
)

PROFILE_INTENTS: Final[dict[Profile, frozenset[Intent]]] = {
    Profile.RECEPTIONIST: frozenset(Intent) - {Intent.ORDER_STATUS},
    Profile.SUPPORT_SALES: frozenset(
        {
            Intent.GREETING,
            Intent.HUMAN,
            Intent.LOCATION,
            Intent.HOURS,
            Intent.CONTACT,
            Intent.ORDER_STATUS,
            Intent.GENERAL_QUESTION,
            Intent.OTHER,
        }
    ),
}
