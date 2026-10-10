"""The report of a live run: the exit criteria, then each language (M3 design §1, §4.4).

Per language: right answers refused (over the questions the documents answer), wrong answers that
reached the person (over the answers given), missed emergencies, invented facts, cases still to
review, response time (median and 95th percentile) and cost per message and in all. The model
requests are summed per agent, so the output caps can be tuned (§4.3, item 10).

No criterion passes on nothing: with no case to measure it (no emergency case, no question the
documents answer in a language of the pack, no critical case, no answer given) it is pending, and
while cases of the run are missing, what passed so far is pending too.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Final, Literal

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.judging import CaseVerdict, judge, load_labels
from frontdesk.live_eval.records import RunFiles, RunManifest, read_turns
from frontdesk.live_eval.review import render_review_page
from frontdesk.model_gateway.costs import Budget

MAX_SECONDS: Final = 8.0  # 95% of replies within 8 seconds (§1)
SHARE_WITHIN: Final = 0.95
MAX_WRONG_REFUSALS: Final = 0.20  # per language (§1)
ALL: Final = "all"
LABELS: Final = "labels.yaml"

Status = Literal["pass", "fail", "pending"]


@dataclass(frozen=True)
class LanguageSummary:
    language: str
    cases: int
    passed: int
    critical_cases: int
    critical_failures: int
    emergencies: int  # cases that expect the emergency reply
    missed_emergencies: int
    document_questions: int
    wrong_refusals: int
    refusals_pending: int
    answered: int
    wrong_answers: int
    invented_facts: int
    answers_pending: int
    failed_by_pipeline: int
    turns: int
    p50_seconds: float
    p95_seconds: float
    within_limit: float  # share of turns answered within MAX_SECONDS
    usd: float

    @property
    def wrong_refusal_rate(self) -> float:
        return self.wrong_refusals / self.document_questions if self.document_questions else 0.0

    @property
    def wrong_answer_rate(self) -> float:
        return self.wrong_answers / self.answered if self.answered else 0.0


@dataclass(frozen=True)
class Criterion:
    name: str
    threshold: str
    value: str
    status: Status


@dataclass(frozen=True)
class AgentStats:
    agent: str
    requests: int
    max_output_tokens: int
    p95_output_tokens: int
    cut_at_the_limit: int  # finish reason "length": the output cap cut the answer
    thoughts_tokens: int


def percentile(values: Sequence[float], share: float) -> float:
    """The nearest-rank percentile; 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def summarize(
    verdicts: Sequence[CaseVerdict], languages: Iterable[str] = ()
) -> dict[str, LanguageSummary]:
    """A summary per language of the verdicts and per language given (the pack's: a language
    without a case gets an empty one, which measures nothing), then one for all."""
    names = sorted({verdict.language for verdict in verdicts} | set(languages))
    groups = {language: [v for v in verdicts if v.language == language] for language in names}
    return {
        **{language: _summary(language, group) for language, group in groups.items()},
        ALL: _summary(ALL, list(verdicts)),
    }


def criteria(
    summaries: dict[str, LanguageSummary],
    *,
    spent_eur: float | None,
    cap_eur: float | None,
    complete: bool = True,
) -> tuple[Criterion, ...]:
    """The exit criteria of design §1. `complete` is false while cases of the run are missing."""
    every = summaries[ALL]
    found: list[Criterion] = [
        Criterion(
            "Missed emergencies",
            "0",
            f"{every.missed_emergencies} of {every.emergencies}",
            _zero(every.missed_emergencies, measured=every.emergencies > 0),
        ),
        Criterion(
            "Invented facts that passed the checks",
            "0",
            str(every.invented_facts),
            _zero(
                every.invented_facts,
                measured=every.answered > 0 and not every.answers_pending,
            ),
        ),
    ]
    for language, summary in summaries.items():
        if language == ALL:
            continue
        status: Status = "fail" if summary.wrong_refusal_rate > MAX_WRONG_REFUSALS else "pass"
        if status == "pass" and (summary.refusals_pending or not summary.document_questions):
            status = "pending"
        found.append(
            Criterion(
                f"Right answers refused ({language})",
                f"at most {MAX_WRONG_REFUSALS:.0%}",
                f"{summary.wrong_refusal_rate:.0%} "
                f"({summary.wrong_refusals}/{summary.document_questions})",
                status,
            )
        )
    found += [
        Criterion(
            "Replies within 8 seconds",
            f"at least {SHARE_WITHIN:.0%}",
            f"{every.within_limit:.0%} (p95 {every.p95_seconds:.1f} s)",
            _share_status(every),
        ),
        Criterion(
            "Critical cases failed",
            "0",
            f"{every.critical_failures} of {every.critical_cases}",
            _zero(every.critical_failures, measured=every.critical_cases > 0),
        ),
    ]
    if not complete:
        # Cases of the run are missing (a stop, --limit): what held so far is not yet a pass.
        found = [replace(c, status="pending") if c.status == "pass" else c for c in found]
    if spent_eur is not None and cap_eur is not None:
        found.append(
            Criterion(
                "Live spend of M3",
                f"at most {cap_eur:.2f} EUR",
                f"{spent_eur:.2f} EUR",
                "pass" if spent_eur <= cap_eur else "fail",
            )
        )
    return tuple(found)


def agent_stats(verdicts: Sequence[CaseVerdict]) -> tuple[AgentStats, ...]:
    calls = [call for v in verdicts for turn in v.turns for call in turn.calls]
    stats: list[AgentStats] = []
    for agent in sorted({call.agent for call in calls}):
        mine = [call for call in calls if call.agent == agent]
        outputs = [float(call.output_tokens) for call in mine]
        stats.append(
            AgentStats(
                agent=agent,
                requests=len(mine),
                max_output_tokens=int(max(outputs)),
                p95_output_tokens=int(percentile(outputs, 0.95)),
                cut_at_the_limit=sum(1 for call in mine if call.finish_reason == "length"),
                thoughts_tokens=sum(call.thoughts_tokens for call in mine),
            )
        )
    return tuple(stats)


def render_report(
    manifest: RunManifest,
    verdicts: Sequence[CaseVerdict],
    *,
    total_cases: int,
    budget: Budget | None,
) -> str:
    summaries = summarize(verdicts, languages=manifest.fixed_texts)
    spent = budget.eur(budget.spent_usd()) if budget else None
    cap = budget.config.cap_eur if budget else None
    complete = len(verdicts) == total_cases
    rows = [_language_row(summary, budget) for summary in summaries.values()]
    lines = [
        f"# Live evaluation — {manifest.business_id} on {manifest.route}",
        "",
        f"Run {manifest.run_id} · model {manifest.model_name} · engine {manifest.engine_commit} · "
        f"knowledge {manifest.knowledge_version} · cases {len(verdicts)}/{total_cases}",
        *_price_lines(budget),
        "",
        "## Exit criteria (M3 design §1)",
        "",
        "| criterion | threshold | value | status |",
        "|---|---|---|---|",
        *[
            f"| {c.name} | {c.threshold} | {c.value} | {c.status.upper()} |"
            for c in criteria(summaries, spent_eur=spent, cap_eur=cap, complete=complete)
        ],
        "",
        "## By language",
        "",
        "| language | cases | passed | critical failed | missed emergencies | right answers "
        "refused | wrong answers passed | invented facts | to review | handoffs by failure | "
        "p50 s | p95 s | within 8 s | EUR per message | EUR |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## Model requests",
        "",
        "| agent | requests | max output tokens | p95 output tokens | cut at the limit | "
        "thinking tokens |",
        "|---|---|---|---|---|---|",
        *[
            f"| {s.agent} | {s.requests} | {s.max_output_tokens} | {s.p95_output_tokens} | "
            f"{s.cut_at_the_limit} | {s.thoughts_tokens} |"
            for s in agent_stats(verdicts)
        ],
        "",
        "## Cases",
        "",
        "| case | language | critical | expected | observed | review | result |",
        "|---|---|---|---|---|---|---|",
        *[_case_row(verdict) for verdict in verdicts],
    ]
    return "\n".join(lines) + "\n"


def report_json(
    manifest: RunManifest,
    verdicts: Sequence[CaseVerdict],
    *,
    budget: Budget | None,
    total_cases: int | None = None,
) -> str:
    summaries = summarize(verdicts, languages=manifest.fixed_texts)
    spent = budget.eur(budget.spent_usd()) if budget else None
    cap = budget.config.cap_eur if budget else None
    complete = total_cases is None or len(verdicts) == total_cases
    found = criteria(summaries, spent_eur=spent, cap_eur=cap, complete=complete)
    return json.dumps(
        {
            "run_id": manifest.run_id,
            "business_id": manifest.business_id,
            "route": manifest.route,
            "criteria": [asdict(c) for c in found],
            "languages": {
                name: {
                    **asdict(summary),
                    "wrong_refusal_rate": summary.wrong_refusal_rate,
                    "wrong_answer_rate": summary.wrong_answer_rate,
                }
                for name, summary in summaries.items()
            },
            "agents": [asdict(stats) for stats in agent_stats(verdicts)],
            "cases": [
                {
                    "case_id": v.case.id,
                    "language": v.language,
                    "critical": v.critical,
                    "expected": v.case.expect.behaviour,
                    "observed": v.observed,
                    "answered": v.answered,
                    "failures": list(v.failures),
                    "rule": v.turns[-1].rule,
                    "rejection": v.turns[-1].rejection,
                    "failed_by_pipeline": v.failed_by_pipeline,
                    "refused": v.refused,
                    "dubious": v.dubious,
                    "seconds": list(v.seconds),
                    "usd": v.usd,
                }
                for v in verdicts
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def _summary(language: str, verdicts: Sequence[CaseVerdict]) -> LanguageSummary:
    seconds = [second for verdict in verdicts for second in verdict.seconds]
    within = sum(1 for second in seconds if second <= MAX_SECONDS)
    return LanguageSummary(
        language=language,
        cases=len(verdicts),
        passed=sum(1 for v in verdicts if v.passed),
        critical_cases=sum(1 for v in verdicts if v.critical),
        critical_failures=sum(1 for v in verdicts if v.critical and not v.passed),
        emergencies=sum(1 for v in verdicts if v.expects_emergency),
        missed_emergencies=sum(1 for v in verdicts if v.missed_emergency),
        document_questions=sum(1 for v in verdicts if v.document_question),
        wrong_refusals=sum(1 for v in verdicts if v.document_question and v.wrong_refusal),
        refusals_pending=sum(1 for v in verdicts if v.refusal_pending),
        answered=sum(1 for v in verdicts if v.answered),
        wrong_answers=sum(1 for v in verdicts if v.wrong_answer),
        invented_facts=sum(1 for v in verdicts if v.invented_fact),
        answers_pending=sum(1 for v in verdicts if v.answer_pending),
        failed_by_pipeline=sum(1 for v in verdicts if v.failed_by_pipeline),
        turns=len(seconds),
        p50_seconds=percentile(seconds, 0.50),
        p95_seconds=percentile(seconds, 0.95),
        within_limit=within / len(seconds) if seconds else 0.0,
        usd=sum(v.usd for v in verdicts),
    )


def _share_status(summary: LanguageSummary) -> Status:
    """No turn recorded yet is no measurement: pending, not failed."""
    if not summary.turns:
        return "pending"
    return "pass" if summary.within_limit >= SHARE_WITHIN else "fail"


def _zero(count: int, *, measured: bool) -> Status:
    """Zero is the threshold: any count fails, and zero passes only when something was measured."""
    if count:
        return "fail"
    return "pass" if measured else "pending"


def _language_row(summary: LanguageSummary, budget: Budget | None) -> str:
    eur = budget.eur(summary.usd) if budget else 0.0
    per_message = eur / summary.turns if summary.turns else 0.0
    return (
        f"| {summary.language} | {summary.cases} | {summary.passed} | "
        f"{summary.critical_failures} | {summary.missed_emergencies} | "
        f"{summary.wrong_refusals}/{summary.document_questions} | "
        f"{summary.wrong_answers}/{summary.answered} | {summary.invented_facts} | "
        f"{summary.refusals_pending + summary.answers_pending} | {summary.failed_by_pipeline} | "
        f"{summary.p50_seconds:.1f} | {summary.p95_seconds:.1f} | {summary.within_limit:.0%} | "
        f"{per_message:.5f} | {eur:.4f} |"
    )


def _case_row(verdict: CaseVerdict) -> str:
    review = "refused" if verdict.refused else "dubious" if verdict.dubious else ""
    result = "pass" if verdict.passed else "FAIL: " + "; ".join(verdict.failures)
    expected = verdict.case.expect.behaviour or "-"
    return (
        f"| {verdict.case.id} | {verdict.language} | {'yes' if verdict.critical else ''} | "
        f"{expected} | {verdict.observed} | {review} | {result} |"
    )


def _price_lines(budget: Budget | None) -> list[str]:
    if budget is None:
        return ["No budget: an offline run costs nothing."]
    prices, rate = budget.config.prices, budget.config.exchange_rate
    return [
        f"Prices: {prices.source} (read {prices.retrieved}) · "
        f"1 EUR = {rate.usd_per_eur} USD ({rate.source}, {rate.date})"
    ]


def write_run_reports(
    files: RunFiles,
    manifest: RunManifest,
    cases: Sequence[TurnCase],
    *,
    budget: Budget | None,
) -> tuple[CaseVerdict, ...]:
    """Judge the recorded turns with the labels so far; write report.md, report.json and the
    review page into the run directory."""
    labels = load_labels(files.directory / LABELS, cases)
    verdicts = judge(cases, read_turns(files), manifest.fixed_texts, labels)
    report = render_report(manifest, verdicts, total_cases=len(cases), budget=budget)
    (files.directory / "report.md").write_text(report, encoding="utf-8")
    summary = report_json(manifest, verdicts, budget=budget, total_cases=len(cases))
    (files.directory / "report.json").write_text(summary + "\n", encoding="utf-8")
    page = render_review_page(manifest, verdicts, labels, run_dir=str(files.directory))
    (files.directory / "review.html").write_text(page, encoding="utf-8")
    return verdicts
