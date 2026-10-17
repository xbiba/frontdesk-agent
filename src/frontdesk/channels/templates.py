"""The booking request template (WhatsApp W1, spec D4): what the staff receives about a booking that
waits for the clinic, outside WhatsApp's 24-hour window, where only templates Meta approved go out.

Its four parameters are the day and time, the service, the doctor and the patient's name; never the
patient's phone, which stays on cal.com (Albi, 2026-10-16). Italian unless reception prefers
Albanian (FRONTDESK_WHATSAPP_STAFF_LANGUAGE, Albi, 2026-10-17); the Albanian copy waits for the
native speaker's review, like the others.
"""

from __future__ import annotations

from typing import Any, Final

BOOKING_REQUEST: Final = "frontdesk_booking_request"
_COPY: Final[dict[str, dict[str, Any]]] = {
    "it": {
        "text": "Nuova richiesta di prenotazione: {{1}}, {{2}} con {{3}}. Paziente: {{4}}.",
        "example": [
            "venerdì 23 ottobre, 10:00",
            "Trattamento osteopatico",
            "Dr. Salvatore Oliverio",
            "Mario Rossi",
        ],
        "buttons": ("Conferma", "Rifiuta"),
    },
    "sq": {
        "text": "Kërkesë e re për rezervim: {{1}}, {{2}} me {{3}}. Pacienti: {{4}}.",
        "example": [
            "e premte 23 tetor, 10:00",
            "Trajtim osteopatik",
            "Dr. Salvatore Oliverio",
            "Mario Rossi",
        ],
        "buttons": ("Konfirmo", "Refuzo"),
    },
}


def booking_request_definition(language: str) -> dict[str, Any]:
    """The template as Meta's message_templates endpoint takes it."""
    copy = _COPY[language]
    return {
        "name": BOOKING_REQUEST,
        "language": language,
        "category": "UTILITY",
        "components": [
            {"type": "BODY", "text": copy["text"], "example": {"body_text": [copy["example"]]}},
            {
                "type": "BUTTONS",
                "buttons": [{"type": "QUICK_REPLY", "text": text} for text in copy["buttons"]],
            },
        ],
    }
