"""A responder outage, as a run: the model fails on every call.

Fifteen questions the documents answer, five in each language, none answered; an emergency and a
greeting per language that the pipeline still handles (a crash of the model does not touch them).
"""

from __future__ import annotations

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, judge
from frontdesk.live_eval.records import TurnRecord
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, record


def outage() -> tuple[CaseVerdict, ...]:
    """The responder fails on every call: 15 document questions in three languages, none answered,
    an emergency and a greeting per language that the pipeline still handles."""
    cases: list[TurnCase] = []
    records: list[TurnRecord] = []
    for language, expected in (("it", "sei"), ("en", "six"), ("sq", "gjashte")):
        for n in range(5):
            name = f"{language}-doc-{n}"
            cases.append(
                case(name, behaviour="answer", contains=[expected], sources=[f"parking-{language}"])
            )
            records.append(
                record(
                    name,
                    CONTACT,
                    outcome="handoff",
                    rejection="responder_error",
                    language=language,
                    seconds=0.4,
                )
            )
        cases.append(case(f"{language}-alarm", behaviour="emergency"))
        records.append(
            record(
                f"{language}-alarm",
                f"{TEXTS['emergency']}\n\n{CONTACT}",
                outcome="handoff",
                rule="emergency",
                category="emergency",
                language=language,
                seconds=0.1,
                calls=(),
            )
        )
        cases.append(case(f"{language}-hello", behaviour="answer", intent="greeting"))
        records.append(
            record(
                f"{language}-hello",
                "Ciao!",
                outcome="fixed",
                intent="greeting",
                language=language,
                seconds=0.3,
            )
        )
    return judge(cases, records, FIXED_TEXTS, {})
