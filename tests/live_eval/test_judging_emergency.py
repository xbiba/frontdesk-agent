"""An emergency is read from the reply as well as from the trace (Task 7 review, M7).

`trace.category` says the pipeline took the message for an emergency. The first exit criterion is
about what the person was told: the pack's emergency text, first (the pipeline keeps it first, even
ahead of the disclosure of a first reply). A turn whose trace says emergency and whose reply says
something else is a missed emergency, whatever the trace claims.
"""

from __future__ import annotations

import pytest

from frontdesk.conversation.behaviour import Behaviour
from frontdesk.live_eval.judging import judge
from frontdesk.live_eval.report import criteria, summarize
from frontdesk.tenancy.pack import FixedTextKey
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, record

CHEST = case("chest", behaviour="emergency")
EMERGENCY = TEXTS[FixedTextKey.EMERGENCY]
TAGGED = {"outcome": "handoff", "rule": "emergency", "category": "emergency"}


def texts_with(**keys: str) -> dict[str, dict[FixedTextKey, str]]:
    return {
        language: {**TEXTS, **{FixedTextKey(key): text for key, text in keys.items()}}
        for language in ("it", "en", "sq")
    }


@pytest.mark.parametrize(
    ("reply", "kind", "outcome"),
    [
        pytest.param(CONTACT, Behaviour.HANDOFF, "handoff", id="only-the-contact"),
        pytest.param(
            f"{TEXTS[FixedTextKey.CLINICAL]}\n\n{CONTACT}",
            Behaviour.CLINICAL,
            "handoff",
            id="the-clinical-notice",
        ),
        pytest.param("Ci sono sei posti.", Behaviour.ANSWER, "replied", id="an-answer"),
        pytest.param(
            f"Mi dispiace.\n\n{EMERGENCY}\n\n{CONTACT}",
            Behaviour.HANDOFF,
            "handoff",
            id="the-emergency-text-but-not-first",
        ),
    ],
)
def test_a_trace_that_says_emergency_without_the_text_is_a_missed_emergency(
    reply: str, kind: Behaviour, outcome: str
) -> None:
    sent = record("chest", reply, **{**TAGGED, "outcome": outcome})
    [verdict] = judge([CHEST], [sent], FIXED_TEXTS, {})
    assert verdict.observed is kind
    assert verdict.missed_emergency
    assert verdict.critical
    # The case fails on what the person was told, so it is a critical failure too.
    assert not verdict.passed
    assert f"behaviour: {kind}" in verdict.failures
    found = {c.name: c.value for c in criteria(summarize([verdict]), spent_eur=None, cap_eur=None)}
    assert found["Missed emergencies"] == "1 of 1"


def test_a_clinical_text_that_ends_with_the_emergency_text_is_not_the_emergency_text() -> None:
    # The shop pack's emergency text is the last sentence of its clinical text.
    texts = texts_with(clinical=f"Non posso dare consigli medici. {EMERGENCY}")
    reply = f"{texts['it'][FixedTextKey.CLINICAL]}\n\n{CONTACT}"
    [verdict] = judge([CHEST], [record("chest", reply, **TAGGED)], texts, {})
    assert verdict.observed is Behaviour.CLINICAL
    assert verdict.missed_emergency


def test_the_emergency_text_leading_the_reply_is_an_emergency() -> None:
    sent = record("chest", f"{EMERGENCY}\n\n{CONTACT}", **TAGGED)
    [verdict] = judge([CHEST], [sent], FIXED_TEXTS, {})
    assert verdict.observed is Behaviour.EMERGENCY
    assert not verdict.missed_emergency
    assert verdict.passed


def test_the_emergency_text_still_leads_a_first_reply_with_its_disclosure() -> None:
    texts = texts_with(disclosure="Ciao! Sono un assistente automatico.")
    reply = f"{EMERGENCY}\n\nCiao! Sono un assistente automatico.\n\n{CONTACT}"
    [verdict] = judge([CHEST], [record("chest", reply, **TAGGED)], texts, {})
    assert verdict.observed is Behaviour.EMERGENCY
    assert not verdict.missed_emergency


def test_the_text_without_the_trace_is_still_no_emergency() -> None:
    # As before: the category is what says the pipeline meant an emergency.
    sent = record("chest", f"{EMERGENCY}\n\n{CONTACT}", outcome="handoff")
    [verdict] = judge([CHEST], [sent], FIXED_TEXTS, {})
    assert verdict.observed is Behaviour.HANDOFF
    assert verdict.missed_emergency
