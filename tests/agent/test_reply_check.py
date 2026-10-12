"""The reception agent's reply may use only what it was given (Albi, 2026-10-12)."""

import pytest

from frontdesk.agent.reply_check import reply_problems

KNOWN = "Trattamento osteopatico: 40 EUR. Telefono: +355 69 443 9339. Sito: www.clinicamcu.com"


def test_a_reply_from_the_facts_passes() -> None:
    reply = (
        "Il trattamento costa 40 EUR. Per altro chiami il +355 69 443 9339 o veda clinicamcu.com"
    )
    assert reply_problems(reply, KNOWN, booked=False) == ()


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        pytest.param("", "empty", id="empty"),
        pytest.param("Costa 45 EUR.", "numbers", id="a-number-from-nowhere"),
        pytest.param("Veda www.altro.it", "links", id="a-link-from-nowhere"),
        pytest.param("Potrebbe trattarsi di una sciatica.", "diagnosis", id="a-diagnosis"),
        pytest.param("La richiameremo domani.", "promises", id="a-promise"),
        pytest.param("Prenotazione confermata per venerdì.", "booking", id="a-booking-not-made"),
    ],
)
def test_a_reply_beyond_the_facts_is_refused(reply: str, problem: str) -> None:
    assert any(problem in reason for reason in reply_problems(reply, KNOWN, booked=False))


def test_a_booking_the_tool_made_may_be_announced() -> None:
    assert reply_problems("Prenotazione confermata.", KNOWN, booked=True) == ()
