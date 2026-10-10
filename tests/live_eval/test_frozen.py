"""What judging and reporting return cannot be changed by whoever holds it."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from frontdesk.live_eval.judging import Label
from frontdesk.live_eval.report import agent_stats, criteria, summarize
from tests.live_eval.report_scenario import verdicts


def test_a_verdict_cannot_be_changed() -> None:
    [first, *_] = verdicts()
    with pytest.raises(FrozenInstanceError):
        first.refused = True  # type: ignore[misc]


def test_a_summary_a_criterion_and_the_stats_of_an_agent_cannot_be_changed() -> None:
    summary = summarize(verdicts())["it"]
    [criterion, *_] = criteria(summarize(verdicts()), spent_eur=None, cap_eur=None)
    [stats] = agent_stats(verdicts())
    for item, field in ((summary, "cases"), (criterion, "status"), (stats, "requests")):
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, 99)


def test_a_label_cannot_be_changed() -> None:
    label = Label(answer="right")
    with pytest.raises(ValidationError, match="frozen"):
        label.answer = "wrong"  # type: ignore[misc]
