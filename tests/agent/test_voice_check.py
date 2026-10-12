"""The voice may change the words of a reply written by code, never its facts."""

import pytest

from frontdesk.agent.voice_check import voice_problems

HANDOFF = "Per parlare con una persona, chiami la reception al +355 69 443 9339."
FACTS = "Trattamento osteopatico: 40 EUR.\nPrenotabile online: sì."
NAMES = ("Dr. Salvatore Oliverio", "Dr. Arben Demo", "Trattamento osteopatico")


def problems(draft: str, voiced: str, *, facts: bool = False) -> tuple[str, ...]:
    return voice_problems(draft, voiced, language="it", names=NAMES, facts=facts)


def test_a_rewrite_that_keeps_the_facts_passes() -> None:
    voiced = "Certo! Per parlare con una persona può chiamare la reception al +355 69 443 9339."
    assert problems(HANDOFF, voiced) == ()
    assert problems(FACTS, "Il trattamento osteopatico costa 40 EUR e si prenota online.") == ()


def test_markdown_is_read_as_plain_text() -> None:
    assert problems(HANDOFF, f"**Certo!** {HANDOFF}") == ()


@pytest.mark.parametrize(
    ("voiced", "reason"),
    [
        pytest.param("", "empty", id="empty"),
        pytest.param("Chiami la reception al +355 69 443 9330.", "numbers", id="a-changed-number"),
        pytest.param(f"{HANDOFF} La visita costa 40 EUR.", "price", id="an-added-price"),
        pytest.param(f"{HANDOFF} Info: www.example.com", "links", id="an-added-link"),
        pytest.param(f"Per il mal di schiena: {HANDOFF}", "clinical", id="added-health-words"),
        pytest.param(
            f"Potrebbe trattarsi di una lombalgia. {HANDOFF}", "diagnostic", id="a-diagnosis"
        ),
        pytest.param(f"{HANDOFF} La richiameremo domani.", "promise", id="a-promise"),
        pytest.param(f"Il Dr. Arben Demo può aiutarla. {HANDOFF}", "names", id="an-added-doctor"),
        pytest.param(
            f"{HANDOFF} " + "Siamo sempre felici di aiutarla. " * 12, "too_long", id="long"
        ),
    ],
)
def test_a_rewrite_that_changes_a_fact_is_refused(voiced: str, reason: str) -> None:
    assert reason in problems(HANDOFF, voiced)


def test_a_name_left_out_is_refused() -> None:
    draft = "Se ne occupa: Trattamento osteopatico (Dr. Salvatore Oliverio)."
    assert "names" in problems(draft, "Se ne occupa il trattamento osteopatico.", facts=True)


def test_a_negation_may_not_turn_a_fact_around() -> None:
    voiced = "Il trattamento osteopatico costa 40 EUR e non si prenota online."
    assert "negation" in problems(FACTS, voiced, facts=True)
    # Words around a handoff carry no fact: a negation there changes none.
    assert problems(HANDOFF, f"Non saprei dirle di più. {HANDOFF}") == ()
