"""The booking request template (W1, spec D4), as Meta takes it, in the staff's two languages."""

from frontdesk.channels.templates import BOOKING_REQUEST, booking_request_definition


def test_the_template_is_defined_as_the_spec_writes_it() -> None:
    assert BOOKING_REQUEST == "frontdesk_booking_request"
    assert booking_request_definition("it") == {
        "name": "frontdesk_booking_request",
        "language": "it",
        "category": "UTILITY",
        "components": [
            {
                "type": "BODY",
                "text": "Nuova richiesta di prenotazione: {{1}}, {{2}} con {{3}}. Paziente: {{4}}.",
                "example": {
                    "body_text": [
                        [
                            "venerdì 23 ottobre, 10:00",
                            "Trattamento osteopatico",
                            "Dr. Salvatore Oliverio",
                            "Mario Rossi",
                        ]
                    ]
                },
            },
            {
                "type": "BUTTONS",
                "buttons": [
                    {"type": "QUICK_REPLY", "text": "Conferma"},
                    {"type": "QUICK_REPLY", "text": "Rifiuta"},
                ],
            },
        ],
    }


def test_the_albanian_template_has_the_same_shape() -> None:
    definition = booking_request_definition("sq")
    body, buttons = definition["components"]
    assert (definition["name"], definition["language"]) == ("frontdesk_booking_request", "sq")
    assert body["text"] == "Kërkesë e re për rezervim: {{1}}, {{2}} me {{3}}. Pacienti: {{4}}."
    assert len(body["example"]["body_text"][0]) == 4
    assert [button["text"] for button in buttons["buttons"]] == ["Konfirmo", "Refuzo"]
