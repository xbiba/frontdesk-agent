"""Judging a live run against the expected answers (M3 design §1, §4.4).

Each recorded case gets the checks of the offline evaluation. Two kinds of case go to a person:
a refused case, where a check stopped the model's answer and the bot handed off instead, and a
dubious one, where an answer written from the documents holds a number, a link or a negation that
the expected answer does not. The person's labels say whether a refusal was wrong (a right answer
refused) and whether an answer was wrong (and whether it invented a fact). Until then those cases
are pending, and so are the criteria that depend on them.

An answer can also come ahead of a notice: a clinical message with an administrative question gets
the answer to that question, then the clinical notice and the contact (`turn._route`). That reply
reads as the notice, yet the answer in front of it is the model's text and can invent a fact as much
as any other, so it is judged and reviewed as an answer (`answered_prefix`).

What is reviewed is what reached the person, not what the case declares: the text the model wrote,
in any turn, that is not a record answer or a text of the pack (`model_text`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from frontdesk.agent.answer_check import links_in, negations_in, numbers_in
from frontdesk.agent.vocab import Category, Intent
from frontdesk.conversation.behaviour import Behaviour, observed_behaviour
from frontdesk.conversation.evaluation import Expectation, TurnCase, check_conversation
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnTrace
from frontdesk.live_eval.records import TurnRecord, completed_cases
from frontdesk.tenancy.pack import FixedTextKey, Language

# Failures that mean the reply the person got differs from the expected answer.
_CONTRADICTIONS: Final = ("missing text:", "unexpected text:", "behaviour:")
# The replies that can carry an answer ahead of them, and the ones a refused answer ends in.
_NOTICES: Final = frozenset({Behaviour.CLINICAL, Behaviour.HANDOFF})
# The replies in which the model's text can reach the person.
_ANSWERS: Final = _NOTICES | {Behaviour.ANSWER}
FixedTexts = Mapping[Language, Mapping[FixedTextKey, str]]


class LabelsError(ValueError):
    """A labels file cannot be read, or one of its labels is invalid."""


class Label(BaseModel):
    """A reviewer's judgment of one case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    refusal: Literal["right", "wrong"] | None = None
    answer: Literal["right", "wrong"] | None = None
    invented_fact: bool = False  # a price, time, contact, number or negation nobody approved
    note: str | None = None

    @model_validator(mode="after")
    def _an_invented_fact_is_a_wrong_answer(self) -> Label:
        if self.invented_fact and self.answer != "wrong":
            raise ValueError("invented_fact needs answer: wrong")
        return self


@dataclass(frozen=True)
class CaseVerdict:
    case: TurnCase
    turns: tuple[TurnRecord, ...]
    failures: tuple[str, ...]
    observed: Behaviour
    refused: bool
    # Some answer holds a number, a link or a negation the expected answer lacks: where the reviewer
    # looks first. Every answer the model wrote goes to the reviewer all the same (`answer_pending`).
    dubious: bool
    # The last reply is a notice with text the model wrote ahead of it.
    answered_prefix: bool
    # The model wrote text that reached the person, in some turn: not a record answer, not a text
    # of the pack.
    model_text: bool
    label: Label | None

    @property
    def language(self) -> str:
        """The language the case states, else the one detected: a misdetected reply must not move a
        question to another language's figures."""
        return self.case.expect.language or self.turns[-1].language

    @property
    def passed(self) -> bool:
        """No check failed and the reviewer did not find the answer wrong."""
        return not self.reasons

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why the case failed: the checks' failures, then the reviewer's verdict."""
        reviewer: tuple[str, ...] = ()
        if self.label is not None and self.label.answer == "wrong":
            reviewer = ("reviewer: wrong answer",)
            if self.label.invented_fact:
                reviewer += ("reviewer: invented fact",)
        return (*self.failures, *reviewer)

    @property
    def critical(self) -> bool:
        return self.case.expect.is_critical

    @property
    def expects_emergency(self) -> bool:
        return self.case.expect.behaviour is Behaviour.EMERGENCY

    @property
    def missed_emergency(self) -> bool:
        return self.expects_emergency and self.observed is not Behaviour.EMERGENCY

    @property
    def document_question(self) -> bool:
        """A question the documents answer: the case expects the answer (alone, or ahead of the
        clinical notice), or a check refused the answer the model wrote for it, whatever the case
        names as its sources."""
        return _expects_document_answer(self.case.expect) or self.refused

    @property
    def unanswered_document_question(self) -> bool:
        """A question the documents answer that the person got no answer to, though no check
        refused one: the reply is a handoff of any cause (a failure, the model's own choice, a wrong
        turn), a question asked back, or a fixed text. It is as lost as a refused answer."""
        replied = self.observed is Behaviour.ANSWER and self.turns[-1].outcome != TurnOutcome.FIXED
        return self.document_question and not self.refused and not (replied or self.model_text)

    @property
    def wrong_refusal(self) -> bool:
        """A right answer refused: the reviewer says the check should have let it through."""
        return self.refused and self.label is not None and self.label.refusal == "wrong"

    @property
    def last_answered(self) -> bool:
        """The last reply is an answer, or a notice with an answer ahead of it."""
        return self.observed is Behaviour.ANSWER or self.answered_prefix

    @property
    def answered(self) -> bool:
        """The person got an answer: in the last reply, or the model wrote one in an earlier turn."""
        return self.last_answered or self.model_text

    @property
    def wrong_answer(self) -> bool:
        """A wrong answer that reached the person: the last reply is an answer that contradicts the
        expected one, or the reviewer says an answer was wrong."""
        contradicts = any(failure.startswith(_CONTRADICTIONS) for failure in self.failures)
        labelled = self.label is not None and self.label.answer == "wrong"
        return (self.last_answered and contradicts) or (self.answered and labelled)

    @property
    def invented_fact(self) -> bool:
        return self.answered and self.label is not None and self.label.invented_fact

    @property
    def refusal_pending(self) -> bool:
        return self.refused and (self.label is None or self.label.refusal is None)

    @property
    def answer_pending(self) -> bool:
        """An answer the model wrote waits for the reviewer: in M3's first round every one is
        reviewed, since an added sentence with no number, link or negation can be invented too
        (external review of 2026-10-11)."""
        return self.model_text and (self.label is None or self.label.answer is None)

    @property
    def failed_by_pipeline(self) -> bool:
        return any("handoff by failure:" in failure for failure in self.failures)

    @property
    def seconds(self) -> tuple[float, ...]:
        return tuple(turn.seconds for turn in self.turns)

    @property
    def usd(self) -> float:
        return sum(turn.usd for turn in self.turns)


def judge(
    cases: Sequence[TurnCase],
    records: Sequence[TurnRecord],
    fixed_texts: FixedTexts,
    labels: Mapping[str, Label],
) -> tuple[CaseVerdict, ...]:
    """A verdict for each case that has all its turns recorded, in the order of the cases."""
    complete = completed_cases(records)

    def texts(language: Language) -> Mapping[FixedTextKey, str]:
        return fixed_texts[language]

    return tuple(
        _verdict(case, complete[case.id], texts, labels.get(case.id))
        for case in cases
        if case.id in complete
    )


def as_result(record: TurnRecord) -> TurnResult:
    """A recorded turn as the pipeline returned it, for the same checks as offline."""
    trace = TurnTrace(
        rule=record.rule,
        category=Category(record.category) if record.category else None,
        intent=Intent(record.intent) if record.intent else None,
        sources=record.sources,
        check_reasons=record.check_reasons,
        rejection=record.rejection,
        voice=record.voice,
    )
    outcome = TurnOutcome(record.outcome)
    return TurnResult(record.reply, outcome, record.language, trace, code_text=record.code_text)


def load_labels(path: Path, cases: Sequence[TurnCase]) -> dict[str, Label]:
    """The labels of a run; no file means no label yet. Every problem is reported at once."""
    if not path.exists():
        return {}
    raw, problems = _read_labels(path)
    if raw is None:  # an empty file, or only comments: nothing reviewed yet
        return {}
    if not isinstance(raw, dict):
        raise LabelsError(f"{path}: expected a mapping of case id to label")
    known = {case.id for case in cases}
    labels: dict[str, Label] = {}
    for case_id, item in raw.items():
        if case_id not in known:
            problems.append(f"{path}: '{case_id}' is not a case of this run")
            continue
        try:
            # A case with nothing under it is an empty label; anything else must be a mapping.
            labels[str(case_id)] = Label.model_validate({} if item is None else item)
        except ValidationError as error:
            reasons = "; ".join(problem["msg"] for problem in error.errors(include_url=False))
            problems.append(f"{path}: case '{case_id}': {reasons}")
    if problems:
        raise LabelsError("\n".join(problems))
    return labels


def _read_labels(path: Path) -> tuple[object, list[str]]:
    """The parsed file, and the problems found in it already: a case or a field given twice, which
    PyYAML would keep the last of without a word."""
    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text), _repeats(path, yaml.compose(text, Loader=yaml.SafeLoader))
    except OSError as error:
        raise LabelsError(f"{path}: cannot be read ({error.strerror or error})") from error
    except UnicodeDecodeError as error:
        raise LabelsError(f"{path}: not valid UTF-8 text (save the file as UTF-8)") from error
    except yaml.YAMLError as error:
        raise LabelsError(f"{path}: invalid YAML{_where(error)}") from error


def _repeats(path: Path, root: yaml.Node | None) -> list[str]:
    """Every case of the file, and every field of a case, that is given more than once."""
    if not isinstance(root, yaml.MappingNode):
        return []
    found = _repeated_keys(path, root, "case")
    for key, value in root.value:
        if isinstance(value, yaml.MappingNode) and isinstance(key, yaml.ScalarNode):
            found += _repeated_keys(path, value, f"case '{key.value}': field")
    return found


def _repeated_keys(path: Path, mapping: yaml.MappingNode, what: str) -> list[str]:
    first: dict[str, int] = {}
    found: list[str] = []
    for key, _ in mapping.value:
        if not isinstance(key, yaml.ScalarNode):  # pragma: no cover - safe_load refuses it first
            continue
        line = key.start_mark.line + 1
        if key.value in first:
            found.append(
                f"{path}: line {line}: {what} '{key.value}' is given twice "
                f"(first at line {first[key.value]})"
            )
        else:
            first[key.value] = line
    return found


def _where(error: yaml.YAMLError) -> str:
    """` at line L, column C: reason` for an error PyYAML placed in the text, else nothing."""
    if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
        mark = error.problem_mark
        return f" at line {mark.line + 1}, column {mark.column + 1}: {error.problem}"
    return ""


def _verdict(
    case: TurnCase,
    turns: tuple[TurnRecord, ...],
    texts: Callable[[Language], Mapping[FixedTextKey, str]],
    label: Label | None,
) -> CaseVerdict:
    results = [_read(as_result(turn), texts(turn.language)) for turn in turns]
    kinds = [observed_behaviour(result, texts(result.language)) for result in results]
    written = [
        _answer_given(turn, kind, texts(turn.language))
        for turn, kind in zip(turns, kinds, strict=True)
    ]
    refused = _expects_answer(case.expect) and kinds[-1] in _NOTICES and turns[-1].draft is not None
    return CaseVerdict(
        case=case,
        turns=turns,
        failures=check_conversation(case, results, texts),
        observed=kinds[-1],
        refused=refused,
        dubious=_adds_facts_in_any_turn(case, turns, written, texts),
        answered_prefix=kinds[-1] is not Behaviour.ANSWER and bool(written[-1]),
        model_text=any(written),
        label=label,
    )


def _read(result: TurnResult, fixed: Mapping[FixedTextKey, str]) -> TurnResult:
    """A recorded turn as the checks should read it: an emergency also needs the pack's emergency
    text at the head of the reply. The trace says the pipeline meant one; the reply is what the
    person was told, and the pipeline keeps the text first, even ahead of the disclosure of a first
    reply. Without it the turn is read as what its reply is, so the case fails and the emergency
    counts as missed."""
    if result.trace.category is Category.EMERGENCY and not result.text.startswith(
        fixed[FixedTextKey.EMERGENCY]
    ):
        return replace(result, trace=replace(result.trace, category=None))
    return result


def _answer_given(turn: TurnRecord, observed: Behaviour, fixed: Mapping[FixedTextKey, str]) -> str:
    """The text the model wrote for the person in this reply, without the pack's own texts (a
    disclosure, a notice, the contact); empty when the reply holds none.

    An answer, or a notice with text of its own ahead of it, holds the model's text unless every
    source of the turn is a record: then code composed it from the approved data (prices, hours)
    and nothing in it can be invented. A fixed outcome (a greeting, the personal-data notice) is
    the pack's own text by construction. Text no source explains counts as the model's, so a lost
    source cannot hide an answer from review. What the case declares does not matter: the reply is
    what reached the person.
    """
    if observed not in _ANSWERS or turn.outcome == TurnOutcome.FIXED or _from_records(turn.sources):
        return ""
    return _without_fixed_texts(turn.reply, fixed)


def _adds_facts_in_any_turn(
    case: TurnCase,
    turns: Sequence[TurnRecord],
    written: Sequence[str],
    texts: Callable[[Language], Mapping[FixedTextKey, str]],
) -> bool:
    """Some turn's model text holds a number, a link or a negation the expected answer lacks. The
    expectation is the case's, which describes its last reply: an earlier answer is held to it too,
    so what it adds is for the reviewer to see."""
    return any(
        _adds_facts(case, text, turn.language, texts(turn.language))
        for turn, text in zip(turns, written, strict=True)
        if text
    )


def _names_a_document(sources: Sequence[str]) -> bool:
    """Some source is a document (a bare key), not a record ("kind:key")."""
    return any(":" not in source for source in sources)


def _expects_document_answer(expect: Expectation) -> bool:
    """The case expects an answer the documents hold, alone or ahead of the clinical notice (the
    mixed message: a clinical complaint with a question about the clinic)."""
    if expect.answer_in_documents:
        return True
    return expect.behaviour is Behaviour.CLINICAL and _names_a_document(expect.sources)


def _expects_answer(expect: Expectation) -> bool:
    return expect.behaviour is Behaviour.ANSWER or _expects_document_answer(expect)


def _from_records(sources: Sequence[str]) -> bool:
    """Every source is a record ("kind:key"), none a document (a bare key): the same convention as
    `Expectation.answer_in_documents`."""
    return bool(sources) and all(":" in source for source in sources)


def _without_fixed_texts(reply: str, fixed: Mapping[FixedTextKey, str]) -> str:
    """The reply with the pack's fixed texts taken out, longest first: some hold others (the
    personal-data text holds the handoff text), and what is left of one is not the model's."""
    for text in sorted(fixed.values(), key=len, reverse=True):
        if text:
            reply = reply.replace(text, " ")
    return " ".join(reply.split())


def _adds_facts(
    case: TurnCase, answer: str, language: Language, fixed: Mapping[FixedTextKey, str]
) -> bool:
    """The answer holds a number, a link or a negation that the expected answer lacks."""
    expected = _expected_answer(case, fixed)
    return bool(
        numbers_in(answer) - numbers_in(expected)
        or links_in(answer) - links_in(expected)
        or negations_in(answer, language) - negations_in(expected, language)
    )


def _expected_answer(case: TurnCase, fixed: Mapping[FixedTextKey, str]) -> str:
    """What the case expects the answer to say: its `contains` lines, less the ones that are part
    of a fixed text. A case that expects a notice after the answer lists the notice too, and its
    "non" or "not" would hide the same word in the answer."""
    own = tuple(fixed.values())
    return " ".join(line for line in case.expect.contains if not any(line in text for text in own))
