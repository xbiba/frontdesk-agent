"""`eval turns --live` on a live route, with a fake Gemini where the real one would be built (Task 8).

What the command shows before it spends, what it records, and how a run resumes. No test here calls
a model or reaches the network: the model is a function, and the free access check is a record
(`access_checks` in conftest.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontdesk.cli import runtime as cli_runtime
from frontdesk.cli.main import app
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.model_gateway.metering import MeteredModel
from frontdesk.model_gateway.routes import DataClass
from tests.cli.conftest import AccessChecks
from tests.cli.live_chat_helpers import PRICE, Install, bootstrapped, ledger, runner
from tests.cli.live_eval_helpers import (
    eval_live,
    eval_report,
    recorded_cases,
    write_cases,
    write_mixed_cases,
)
from tests.model_gateway.budget_helpers import ROUTE, write_budget

pytestmark = pytest.mark.db
# Four requests per turn at most, each held at its reserve: 32,000 tokens in and a cap of 1,200 out
# (the responder's, the larger) cost 0.044 USD at the test prices of 1 and 10 USD per million.
CEILING_PER_TURN_USD = 4 * 0.044


def test_a_live_run_shows_what_it_could_cost_pays_from_the_budget_and_records_everything(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)
    budget.record(run_id="earlier", route_id=ROUTE, kind="chat", usd=2.5)  # 2.00 EUR spent before
    run_dir = tmp_path / "runs" / "live-1"

    result = eval_live(env, budget, run_dir, write_mixed_cases(tmp_path))

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert (
        lines[0] == "vertex-eu-gemini-flash-lite (gemini-3.1-flash-lite): 3 cases, 3 turns to run"
    )
    ceiling = 3 * CEILING_PER_TURN_USD / 1.25
    assert lines[1] == (
        f"at most {ceiling:.2f} EUR; the budget has 8.00 of 10.00 EUR left"
    )  # 0.42: the ceiling is shown before the first call
    assert f"cases recorded: 3/3 in {run_dir}" in lines
    assert any(line.startswith("PASS     Live spend of M3: 2.02 EUR") for line in lines)
    assert sent == ["classifier", "classifier"]  # the emergency is decided by the rules alone
    assert recorded_cases(run_dir) == ["price-it", "chest-it", "price-tagged"]
    assert access_checks.calls == [(ROUTE, "demo-project")]
    # The ledger: what was there, the proof that it takes a line, then one line per paid request,
    # filed under the run's directory name.
    earlier, proof, *paid = ledger(budget)
    assert earlier["run"] == "earlier"
    assert (proof["kind"], proof["usd"], proof["run"]) == ("preflight", 0.0, "live-1")
    assert [(entry["kind"], entry["run"]) for entry in paid] == [("chat", "live-1")] * 2
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (manifest["route"], manifest["model_name"]) == (ROUTE, "gemini-3.1-flash-lite")
    assert manifest["budget_file"] == str(budget.path)
    assert (manifest["min_support_similarity"], manifest["min_retrieval_similarity"]) == (0.5, 0.0)
    for report in ("report.md", "report.json", "review.html"):
        assert (run_dir / report).exists()
    with budget.exclusive():  # the run gave the budget back
        pass


def test_a_run_may_be_cut_short_by_limit_and_resumes_where_it_stopped(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    cases, run_dir = write_mixed_cases(tmp_path), tmp_path / "runs" / "live-1"

    first = eval_live(env, budget, run_dir, cases, "--limit", "1")
    assert first.exit_code == 0, first.output
    assert "1 cases, 1 turns to run" in first.stdout
    assert "cases recorded: 1/3" in first.stdout
    assert "PENDING  Missed emergencies" in first.stdout  # what held so far is not yet a pass
    assert recorded_cases(run_dir) == ["price-it"]

    second = eval_live(env, budget, run_dir, cases)
    assert second.exit_code == 0, second.output
    assert "2 cases, 2 turns to run" in second.stdout
    assert "cases recorded: 3/3" in second.stdout
    assert recorded_cases(run_dir) == ["price-it", "chest-it", "price-tagged"]
    assert sent == ["classifier", "classifier"]  # no call of a finished case was made again
    assert len(access_checks.calls) == 2
    assert [entry["kind"] for entry in ledger(budget)] == [
        "preflight",
        "chat",
        "preflight",
        "chat",
    ]


def test_a_run_with_nothing_left_to_run_asks_nothing_checks_nothing_and_writes_nothing(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    cases, run_dir = write_cases(tmp_path, 1), tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, cases).exit_code == 0
    lines_before = ledger(budget)

    again = eval_live(env, budget, run_dir, cases, yes=False)  # no --yes, and no question

    assert again.exit_code == 0, again.output
    assert "0 cases, 0 turns to run" in again.stdout
    assert "Start the run?" not in again.stdout
    assert len(access_checks.calls) == 1  # the first run's
    assert ledger(budget) == lines_before


def test_the_run_starts_only_when_the_person_says_so(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    cases, run_dir = write_cases(tmp_path, 1), tmp_path / "runs" / "live-1"

    declined = eval_live(env, budget, run_dir, cases, yes=False, input="n\n")

    assert declined.exit_code == 1
    assert "at most" in declined.stdout  # the ceiling was shown first
    assert "Start the run?" in declined.stdout
    assert sent == []
    assert not run_dir.exists()  # a run that did not start left nothing behind
    assert not budget.ledger_path.exists()  # not even the proof
    with budget.exclusive():
        pass

    accepted = eval_live(env, budget, run_dir, cases, yes=False, input="y\n")

    assert accepted.exit_code == 0, accepted.output
    assert recorded_cases(run_dir) == ["price-1"]
    assert sent == ["classifier"]
    assert len(access_checks.calls) == 2  # each start proved the route again


def test_tags_choose_the_cases_and_the_report_is_judged_over_the_same_cases(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    cases, run_dir = write_mixed_cases(tmp_path), tmp_path / "runs" / "live-1"

    tagged = eval_live(env, budget, run_dir, cases, "--tag", "pricing")

    assert tagged.exit_code == 0, tagged.output
    assert "1 cases, 1 turns to run" in tagged.stdout
    assert "cases recorded: 1/1" in tagged.stdout
    assert recorded_cases(run_dir) == ["price-tagged"]
    # Recomputed with the tag the run was started with it is the same report; without, the cases
    # the run never meant to run are missing from it.
    assert "PASS     Replies within 8 seconds" in eval_report(run_dir, "--tag", "pricing").stdout
    assert "PENDING  Replies within 8 seconds" in eval_report(run_dir).stdout


def test_a_tag_no_case_has_is_an_error_before_anything_is_written(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    cases, run_dir = write_mixed_cases(tmp_path), tmp_path / "runs" / "live-1"

    result = eval_live(env, budget, run_dir, cases, "--tag", "nothing")

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"error: {cases}: no case has the tags nothing"]
    assert (sent, access_checks.calls) == ([], [])
    assert not run_dir.exists()
    assert not budget.ledger_path.exists()


def test_a_directory_that_holds_another_run_is_refused_before_anything_is_checked_or_paid(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    cases, run_dir = write_cases(tmp_path, 1), tmp_path / "runs" / "live-1"
    offline = ["eval", "turns", "--env-file", str(env), "--live", "--route", "offline-demo"]
    offline += ["--cases", str(cases), "--run-dir", str(run_dir), "--yes"]
    assert runner.invoke(app, offline).exit_code == 0  # a dry run claims the directory

    result = eval_live(env, budget, run_dir, cases)  # the live route is another run

    assert result.exit_code == 1
    [line] = result.stderr.splitlines()
    assert line.startswith(f"error: {run_dir} holds another run (it differs in route")
    assert line.endswith("choose a new --run-dir")
    assert (sent, access_checks.calls) == ([], [])
    assert not budget.ledger_path.exists()


def test_a_live_run_builds_its_pipeline_for_evaluation_data_with_every_request_metered(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[dict[str, object], TurnRuntime]] = []
    real = cli_runtime.build_turn_runtime

    def spy(deployment: object, **options: object) -> TurnRuntime:
        runtime = real(deployment, **options)  # type: ignore[arg-type]
        built.append((options, runtime))
        return runtime

    monkeypatch.setattr(cli_runtime, "build_turn_runtime", spy)
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    result = eval_live(env, write_budget(tmp_path), tmp_path / "live-1", write_cases(tmp_path, 1))
    assert result.exit_code == 0, result.output
    [(options, runtime)] = built
    assert options["data_class"] is DataClass.EVALUATION
    assert runtime.keep_drafts is False  # the runner turns it on for its own copy
    for agent in (runtime.classifier, runtime.responder):
        assert isinstance(agent.model, MeteredModel)


def test_a_missing_project_is_an_error_line_and_leaves_nothing_behind(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database, project=None)
    budget = write_budget(tmp_path)
    run_dir = tmp_path / "runs" / "live-1"

    result = eval_live(env, budget, run_dir, write_cases(tmp_path, 1))

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "error: FRONTDESK_GCP_PROJECT is required for the vertex-eu-gemini-flash-lite route"
    ]
    assert access_checks.calls == []
    assert not run_dir.exists()
    assert not budget.ledger_path.exists()


def test_a_cases_file_that_is_wrong_is_reported_before_anything_is_written(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    broken = tmp_path / "broken.yaml"
    broken.write_text("- {id: a, message: x, expect: {outcome: maybe}}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "live-1"

    result = eval_live(env, budget, run_dir, broken)

    assert result.exit_code == 1
    assert str(broken) in result.stderr
    assert "case 'a': expect.outcome" in result.stderr
    assert access_checks.calls == []
    assert not run_dir.exists()
    assert not budget.ledger_path.exists()


def test_a_manifest_that_is_not_one_is_an_error_line_and_nothing_runs(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, run_dir = write_budget(tmp_path), tmp_path / "runs" / "live-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{not json", encoding="utf-8")

    result = eval_live(env, budget, run_dir, write_cases(tmp_path, 1))

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"error: {run_dir / 'run.json'}: not a run manifest"]
    assert (sent, access_checks.calls) == ([], [])
    assert not budget.ledger_path.exists()


def test_a_record_of_turns_that_is_damaged_is_an_error_line_and_nothing_is_paid(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    access_checks: AccessChecks,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget, cases = write_budget(tmp_path), write_cases(tmp_path, 2)
    run_dir = tmp_path / "runs" / "live-1"
    assert eval_live(env, budget, run_dir, cases, "--limit", "1").exit_code == 0
    turns = run_dir / "turns.jsonl"
    turns.write_text("this is no record\n" + turns.read_text(encoding="utf-8"), encoding="utf-8")
    requests_before, lines_before = len(sent), ledger(budget)

    result = eval_live(env, budget, run_dir, cases)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"error: {turns}: line 1 is not a turn record"]
    assert len(sent) == requests_before  # nothing more was sent
    assert ledger(budget) == lines_before  # and nothing more was counted
    assert len(access_checks.calls) == 1  # the damaged run was found before the route was proved
