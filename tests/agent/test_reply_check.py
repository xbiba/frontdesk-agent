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


@pytest.mark.parametrize(
    "reply",
    [
        "Dr. Oliverio is at the clinic from 23 October until 2 November 2026.",
        "He is here until 2 November, and the first free time is on Friday 23 at 10:00.",
        "Il dottor Oliverio è qui fino al 2 novembre, alle 10 di venerdì 23 ha posto.",
        "Dr. Oliverio është në klinikë nga 23 tetori deri më 2 nëntor 2026.",
    ],
)
def test_a_day_of_the_month_is_no_time(reply: str) -> None:
    # "from 23 October until 2 November" was read as 23:00 and 02:00, and a right reply was refused
    # twice (run of 2026-10-13).
    known = f"Il dottor Oliverio è in clinica dal 23 ottobre al 2 novembre 2026.\n{TOOL}"
    assert not list(reply_problems(reply, known, booked=False, times=SHOWN))


def test_an_evening_hour_is_read_whole() -> None:
    # "alle 20" was read as 02:00 and 00:00: an hour of 20 to 23 is one hour.
    assert not list(
        reply_problems("Venerdì 23 alle 20?", TOOL, booked=False, times=frozenset({"20:00"}))
    )


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("Klinika jonë ofron vetëm trajtime osteopatike.", id="sq-only"),
        pytest.param(
            "Non disponiamo delle analisi del sangue tra i nostri servizi.", id="it-not-had"
        ),
        pytest.param("Al momento offriamo solo il trattamento osteopatico.", id="it-only"),
        pytest.param("We only offer osteopathic treatment.", id="en-only"),
        pytest.param("We do not offer blood tests.", id="en-not-offered"),
        pytest.param("Nuk ofrojmë analiza gjaku.", id="sq-not-offered"),
    ],
)
def test_the_clinic_is_never_said_to_offer_only_what_the_facts_list(reply: str) -> None:
    # Knowing the one service it prices, the agent said the clinic had no other (run of 2026-10-13).
    assert {problem.code for problem in reply_problems(reply, KNOWN, booked=False)} == {"denies"}


@pytest.mark.parametrize(
    "reply",
    [
        "In chat posso prenotare solo il trattamento osteopatico; per il resto, la reception.",
        "Bookings in the chat are only for osteopathy.",
        "Venerdì ci sono solo posti di mattina.",
        "We do not have free times on Friday.",
        "Non posso darle consigli medici.",
        "Non ho informazioni sulle analisi del sangue: la reception può aiutarla.",
        "I don't have information on blood tests here.",
        "Nuk kam informacion për analizat e gjakut.",
    ],
)
def test_what_the_chat_books_or_what_is_free_may_be_only_that(reply: str) -> None:
    assert "denies" not in {problem.code for problem in reply_problems(reply, KNOWN, booked=False)}
