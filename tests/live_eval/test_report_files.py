"""Writing the report and the review page into the run directory, and judging again with labels."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from frontdesk.live_eval.judging import LabelsError
from frontdesk.live_eval.records import RunFileError, RunFiles, TurnRecord, append_case
from frontdesk.live_eval.report import render_report, report_json, write_run_reports
from frontdesk.live_eval.review import render_review_page
from tests.live_eval.helpers import case, manifest, record
from tests.live_eval.report_scenario import CASES, LABELS, RECORDS, verdicts
from tests.model_gateway.budget_helpers import write_budget


def recorded(tmp_path: Path, records: list[TurnRecord] | None = None) -> RunFiles:
    """A run directory holding the turns of the scenario, a case at a time as the runner writes."""
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    by_case: dict[str, list[TurnRecord]] = defaultdict(list)
    for turn in RECORDS if records is None else records:
        by_case[turn.case_id].append(turn)
    for turns in by_case.values():
        append_case(files, turns)
    return files


def test_the_report_the_json_and_the_review_page_are_written_into_the_run_directory(
    tmp_path: Path,
) -> None:
    files = recorded(tmp_path)
    budget = write_budget(tmp_path)
    judged = write_run_reports(files, manifest(), CASES, budget=budget)
    assert [v.case.id for v in judged] == [c.id for c in CASES]
    expected = verdicts({})
    assert (files.directory / "report.md").read_text(encoding="utf-8") == render_report(
        manifest(), expected, total_cases=7, budget=budget
    )
    assert (files.directory / "report.json").read_text(encoding="utf-8") == (
        report_json(manifest(), expected, budget=budget, total_cases=7) + "\n"
    )
    assert (files.directory / "review.html").read_text(encoding="utf-8") == render_review_page(
        manifest(), expected, {}, run_dir=str(files.directory)
    )


def test_nothing_else_is_written(tmp_path: Path) -> None:
    files = recorded(tmp_path)
    write_run_reports(files, manifest(), CASES, budget=None)
    assert sorted(path.name for path in files.directory.iterdir()) == [
        "report.json",
        "report.md",
        "review.html",
        "turns.jsonl",
    ]


def test_the_labels_of_the_run_directory_change_the_report_and_the_page(tmp_path: Path) -> None:
    files = recorded(tmp_path)
    write_run_reports(files, manifest(), CASES, budget=None)
    before = (files.directory / "report.md").read_text(encoding="utf-8")
    assert "| it | 6 | 2 | 2 | 1 | 0/3 | 1/2 | 0 | 3 | 1 |" in before  # three cases to review

    (files.directory / "labels.yaml").write_text(
        'refused:\n  refusal: wrong\nhours:\n  answer: wrong\n  invented_fact: true\n  note: "9"\n',
        encoding="utf-8",
    )
    judged = write_run_reports(files, manifest(), CASES, budget=None)
    after = (files.directory / "report.md").read_text(encoding="utf-8")
    assert "| it | 6 | 2 | 2 | 1 | 1/3 | 1/2 | 1 | 1 | 1 |" in after  # one invented; parking waits
    assert next(v for v in judged if v.case.id == "hours").invented_fact
    document = json.loads((files.directory / "report.json").read_text(encoding="utf-8"))
    assert document["languages"]["it"]["invented_facts"] == 1
    page = (files.directory / "review.html").read_text(encoding="utf-8")
    assert 'name="refusal-refused" value="wrong" checked' in page
    assert 'name="answer-hours" value="wrong" checked' in page
    assert ">9</textarea>" in page


def test_the_labels_that_were_kept_are_the_labels_applied(tmp_path: Path) -> None:
    files = recorded(tmp_path)
    (files.directory / "labels.yaml").write_text("refused:\n  refusal: wrong\n", encoding="utf-8")
    write_run_reports(files, manifest(), CASES, budget=None)
    expected = verdicts(LABELS)
    assert (files.directory / "report.md").read_text(encoding="utf-8") == render_report(
        manifest(), expected, total_cases=7, budget=None
    )


def test_a_case_with_a_turn_still_to_run_is_not_judged_yet(tmp_path: Path) -> None:
    two_turns = case("long", behaviour="answer").model_copy(
        update={"message": None, "messages": ("uno", "due")}
    )
    files = recorded(tmp_path, [*RECORDS, record("long", "ok", turn=1, turns=2)])
    judged = write_run_reports(files, manifest(), [*CASES, two_turns], budget=None)
    assert "long" not in [v.case.id for v in judged]
    text = (files.directory / "report.md").read_text(encoding="utf-8")
    assert "cases 7/8" in text
    assert "| Right answers refused (en) | at most 20% | 0% (0/1) | PENDING |" in text
    document = json.loads((files.directory / "report.json").read_text(encoding="utf-8"))
    status = {c["name"]: c["status"] for c in document["criteria"]}
    assert (
        status["Right answers refused (en)"] == "pending"
    )  # the json is as incomplete as the text


def test_a_run_with_no_turn_yet_passes_nothing(tmp_path: Path) -> None:
    files = RunFiles(tmp_path / "run-1")
    files.directory.mkdir()
    assert write_run_reports(files, manifest(), CASES, budget=None) == ()
    text = (files.directory / "report.md").read_text(encoding="utf-8")
    assert "cases 0/7" in text
    assert "PASS" not in text
    page = (files.directory / "review.html").read_text(encoding="utf-8")
    assert "0 cases to review" in page


def test_bad_labels_stop_the_report_before_anything_is_written(tmp_path: Path) -> None:
    files = recorded(tmp_path)
    (files.directory / "labels.yaml").write_text("nobody:\n  answer: right\n", encoding="utf-8")
    with pytest.raises(LabelsError, match="'nobody' is not a case of this run"):
        write_run_reports(files, manifest(), CASES, budget=None)
    assert not (files.directory / "report.md").exists()
    assert not (files.directory / "review.html").exists()


def test_a_damaged_turns_file_is_a_run_file_error(tmp_path: Path) -> None:
    files = recorded(tmp_path)
    with files.turns.open("ab") as turns:
        turns.write(b"not a record\n")
    with pytest.raises(RunFileError, match="is not a turn record"):
        write_run_reports(files, manifest(), CASES, budget=None)
    assert not (files.directory / "report.md").exists()


def test_writing_again_gives_the_same_files(tmp_path: Path) -> None:
    files = recorded(tmp_path)
    write_run_reports(files, manifest(), CASES, budget=None)
    first = {p.name: p.read_bytes() for p in files.directory.iterdir()}
    write_run_reports(files, manifest(), CASES, budget=None)
    assert {p.name: p.read_bytes() for p in files.directory.iterdir()} == first
