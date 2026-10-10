"""A small run worked out by hand: two languages, one case of each kind the report counts.

Italian has six cases (one second each unless noted), English one:

| case | what happened | counts as |
|---|---|---|
| parking | a document question, answered right in 2 s | answered |
| hours | a critical document question, answered "09:00" for "08:00" in 3 s | wrong answer, critical failure, to review |
| refused | a document question, the model's draft refused in 9 s, labelled a wrong refusal | wrong refusal |
| alarm | an emergency, answered with the emergency text | fine |
| missed | an emergency, answered with the clinical notice | missed emergency, critical failure |
| lost | a handoff case, handed off after the classifier named an unknown id | handoff by failure |
| en-parking | the English document question, answered right in 2 s | answered |
"""

from __future__ import annotations

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, Label, judge
from frontdesk.live_eval.records import TurnRecord
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, draft, record

CASES: list[TurnCase] = [
    case("parking", behaviour="answer", contains=["sei"], sources=["parking-it"]),
    case("hours", behaviour="answer", contains=["08:00"], sources=["hours-it"], critical=True),
    case("refused", behaviour="answer", contains=["sei"], sources=["parking-it"]),
    case("alarm", behaviour="emergency"),
    case("missed", behaviour="emergency"),
    case("lost", behaviour="handoff"),
    case("en-parking", behaviour="answer", contains=["six"], sources=["parking-en"]),
]
RECORDS: list[TurnRecord] = [
    record("parking", "Ci sono sei posti.", sources=("parking-it",), seconds=2.0),
    record("hours", "Apriamo alle 09:00.", sources=("hours-it",), seconds=3.0),
    record(
        "refused",
        f"{TEXTS['no_information']}\n\n{CONTACT}",
        outcome="handoff",
        check_reasons=("quote_not_in_source",),
        draft=draft("Ci sono sei posti.", "sei posti gratuiti"),
        seconds=9.0,
    ),
    record(
        "alarm",
        f"{TEXTS['emergency']}\n\n{CONTACT}",
        outcome="handoff",
        rule="emergency",
        category="emergency",
    ),
    record("missed", f"{TEXTS['clinical']}\n\n{CONTACT}", outcome="handoff", category=None),
    record("lost", CONTACT, outcome="handoff", rejection="empty_text"),
    record(
        "en-parking",
        "There are six free spaces.",
        language="en",
        sources=("parking-en",),
        seconds=2.0,
    ),
]
LABELS = {"refused": Label(refusal="wrong")}


def verdicts(labels: dict[str, Label] | None = None) -> tuple[CaseVerdict, ...]:
    return judge(CASES, RECORDS, FIXED_TEXTS, LABELS if labels is None else labels)
