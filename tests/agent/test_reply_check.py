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
        pytest.param("La richiameremo domani.", "promise", id="a-promise"),
        pytest.param("Prenotazione confermata per venerdì.", "booking", id="a-booking-not-made"),
    ],
)
def test_a_reply_beyond_the_facts_is_refused(reply: str, problem: str) -> None:
    found = reply_problems(reply, KNOWN, booked=False)
    assert problem in {item.code for item in found}
    # The code names the kind; only the detail, for the model, holds the reply's values.
    assert all(item.code.isidentifier() for item in found)


def test_a_booking_the_tool_made_may_be_announced() -> None:
    assert reply_problems("Prenotazione confermata.", KNOWN, booked=True) == ()


SHOWN = frozenset({"09:00", "10:00", "12:00", "16:00"})
TOOL = "- 2026-10-23 (venerdì 23 ottobre): 09:00, 10:00, 12:00, 16:00\nOrari: lun-ven 09:00-18:00"


@pytest.mark.parametrize(
    ("reply", "codes"),
    [
        pytest.param("Venerdì 23 ho posto alle 9, 10, 12 e 16: quale preferisce?", set(), id="ok"),
        pytest.param("Alle 10:00 va bene?", set(), id="a-clock-time"),
        pytest.param("Apriamo dalle 9 alle 18.", set(), id="opening-hours-from-the-facts"),
        pytest.param("1) venerdì 23: alle 9 e alle 10\n2) venerdì 23: alle 16", set(), id="a-list"),
        pytest.param("Venerdì 23 alle 9 o alle 11?", {"times"}, id="a-time-not-free"),
        pytest.param("On Friday 23 at 9 and 13.", {"times"}, id="in-english"),
        pytest.param("On Friday 23 at 4 PM.", set(), id="a-12-hour-time"),
        pytest.param("On Friday 23 at 2:00 p.m.", {"times"}, id="a-12-hour-time-not-free"),
    ],
)
def test_each_time_named_is_one_the_tools_gave(reply: str, codes: set[str]) -> None:
    found = reply_problems(reply, TOOL, booked=False, times=SHOWN)
    assert {problem.code for problem in found} == codes
