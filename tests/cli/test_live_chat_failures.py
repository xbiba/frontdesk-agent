"""What a person sees when something goes wrong in a live chat (M3 Task 6b, fix round 1): the line
that says why a model request failed, what a turn cost when a stop cut it short, and one line, not a
traceback, for a database or a ledger that cannot be reached or read.

No model and no network: a fake Gemini sits where `build_live_model` would build the real one, so
it takes the same `wrap_model` path, and fails on request.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx2
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from frontdesk.cli.live_chat import database_location
from tests.cli.live_chat_helpers import (
    PARKING,
    PRICE,
    QUESTION,
    Failure,
    Install,
    bootstrapped,
    env_file,
    ledger,
    live,
    refused,
)
from tests.model_gateway.budget_helpers import write_budget

MIDDLE_DOT = "\N{MIDDLE DOT}"
TURN_LOGGER = logging.getLogger("frontdesk.conversation.turn")


def _refused_by_google(status: int) -> Failure:
    """What the Gemini client raises for an error status that is not retried (a 403 or a 404)."""
    return lambda: ModelHTTPError(status, "gemini-3.1-flash-lite", body={"error": {"code": status}})


def _only_the_first_request(failure: Failure) -> Failure:
    calls: list[str] = []

    def first() -> BaseException | None:
        calls.append("request")
        return failure() if len(calls) == 1 else None

    return first


def _trace(output: str) -> str:
    return next(line for line in output.splitlines() if line.lstrip().startswith("[rule="))


@pytest.mark.db
def test_a_turn_that_ends_in_error_says_why_before_its_hand_off(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("frontdesk.model_gateway.metering.RETRY_WAITS", (0.0, 0.0, 0.0))
    # No network: the meter retries the request four times, then the turn falls back to the
    # hand-off with rule=error. The request never left, so it costs nothing.
    sent = gemini(PRICE, failing={"classifier": lambda: httpx2.ConnectError("no route to host")})
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)

    result = live(env, budget, input=f"{QUESTION}\nexit\n", trace=True)

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == ["model error: ConnectError"]
    lines = result.output.splitlines()
    reply = next(number for number, line in enumerate(lines) if line.startswith("bot> "))
    assert lines.index("model error: ConnectError") < reply  # before the hand-off it explains
    assert "rule=error" in _trace(result.output)
    assert f"cost: 0.0000 EUR {MIDDLE_DOT} session 0.0000 EUR" in result.stdout
    assert "Fisioterapia" not in result.stdout  # the hand-off, not an answer
    assert QUESTION not in result.stderr  # the type of the error, never the message
    assert sent == ["classifier"] * 4
    assert not budget.ledger_path.exists()
    assert (TURN_LOGGER.handlers, TURN_LOGGER.level) == ([], logging.NOTSET)  # given back


@pytest.mark.db
@pytest.mark.parametrize("status", [403, 404])
def test_google_refusing_the_request_says_so_although_the_rule_is_pass(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    status: int,
) -> None:
    # Vertex AI not enabled for the project (403), or a wrong project or model (404): the client
    # raises ModelHTTPError, which the classifier step catches, so the trace says rule=pass and
    # rejection=classifier_error and never names the exception.
    sent = gemini(PRICE, failing={"classifier": _refused_by_google(status)})
    env = bootstrapped(tmp_path, businesses_dir, database)

    result = live(env, write_budget(tmp_path), input=f"{QUESTION}\nexit\n", trace=True)

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == ["model error: ModelHTTPError"]
    trace = _trace(result.output)
    assert "rule=pass" in trace
    assert "rejection=classifier_error" in trace
    assert sent == ["classifier"]  # an error status is not retried


@pytest.mark.db
def test_a_responder_that_fails_after_a_paid_classifier_says_why_and_shows_the_cost(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    sent = gemini(PARKING, failing={"responder": _refused_by_google(403)})
    env = bootstrapped(tmp_path, businesses_dir, database)

    result = live(
        env, write_budget(tmp_path), input="Is there parking near the clinic?\nexit\n", trace=True
    )

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == ["model error: ModelHTTPError"]
    assert "rejection=responder_error" in _trace(result.output)
    # The classifier was answered and paid (0.0104 EUR); the refused responder cost nothing.
    assert f"cost: 0.0104 EUR {MIDDLE_DOT} session 0.0104 EUR {MIDDLE_DOT} left 9.99 EUR" in (
        result.stdout
    )
    assert sent == ["classifier", "responder"]


@pytest.mark.db
def test_each_failed_turn_has_its_line_and_a_good_turn_has_none(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    gemini(PRICE, failing={"classifier": _only_the_first_request(_refused_by_google(403))})
    env = bootstrapped(tmp_path, businesses_dir, database)

    result = live(env, write_budget(tmp_path), input=f"{QUESTION}\n{QUESTION}\nexit\n")

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == ["model error: ModelHTTPError"]  # the first turn only
    lines = result.output.splitlines()
    first_reply, second_reply = (n for n, line in enumerate(lines) if line.startswith("bot> "))
    assert lines.index("model error: ModelHTTPError") < first_reply < second_reply
    # Only the second turn answered: the first one fell back to the hand-off.
    assert result.stdout.count("30 EUR") == 1
    assert "30 EUR" in "\n".join(lines[second_reply:])


@pytest.mark.db
def test_a_turn_cut_short_by_the_stop_shows_what_it_had_cost_before_the_stop(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
) -> None:
    sent = gemini(PARKING)
    env = bootstrapped(tmp_path, businesses_dir, database)
    # The classifier reserves 0.038 USD and the responder 0.044. The classifier fits under 0.05 and
    # costs 0.013; then the responder no longer fits, and the turn is cut short.
    budget = write_budget(tmp_path, cap_eur=0.05, usd_per_eur=1.0)

    result = live(env, budget, input="Is there parking near the clinic?\nexit\n")

    assert result.exit_code == 1
    assert result.output.splitlines()[-2:] == [
        f"cost: 0.0130 EUR {MIDDLE_DOT} session 0.0130 EUR {MIDDLE_DOT} left 0.04 EUR",
        "stopped: budget",
    ]
    assert "bot>" not in result.output  # the hand-off the cut-short turn fell back to is not shown
    assert sent == ["classifier"]
    [entry] = ledger(budget)
    assert entry["usd"] == pytest.approx(0.013)


# --- A database or a ledger that cannot be reached or read -------------------------------------------


def test_a_database_that_is_down_is_one_line_without_the_password(
    tmp_path: Path, businesses_dir: Path
) -> None:
    down = "postgresql+psycopg://frontdesk_app:s3cret-pw@127.0.0.1:1/frontdesk_clinic"  # port 1
    env = env_file(tmp_path, businesses_dir / "demo-clinic", (down, down))
    budget = write_budget(tmp_path)

    result = live(env, budget, input="Ciao\n")

    assert refused(result) == (
        "error: cannot reach the database (127.0.0.1:1); "
        "start it with: docker compose up -d --wait postgres"
    )
    assert "s3cret-pw" not in result.output
    with budget.exclusive():  # the session never started, and gave the budget back
        pass


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        pytest.param(
            "postgresql+psycopg://frontdesk_app:s3cret-pw@127.0.0.1:5432/frontdesk",
            "127.0.0.1:5432",
            id="with-a-port",
        ),
        pytest.param(
            "postgresql+psycopg://frontdesk_app:s3cret-pw@db.example/frontdesk",
            "db.example",
            id="without-a-port",
        ),
        pytest.param(
            "postgresql+psycopg://frontdesk_app:s3cret-pw@/frontdesk",
            "localhost",
            id="without-a-host",
        ),
    ],
)
def test_only_the_host_and_the_port_of_the_database_are_shown(url: str, shown: str) -> None:
    assert database_location(url) == shown


def test_a_ledger_that_is_a_directory_is_one_error_line(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    budget = write_budget(tmp_path)
    budget.ledger_path.mkdir()

    line = refused(live(env, budget, input="Ciao\n"))

    assert line == f"error: {budget.ledger_path}: cannot be read (Is a directory)"
    with budget.exclusive():
        pass


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file whatever its mode")
def test_a_ledger_that_may_not_be_read_is_one_error_line(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = env_file(tmp_path, businesses_dir / "demo-clinic")
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text("", encoding="utf-8")
    budget.ledger_path.chmod(0o000)
    try:
        line = refused(live(env, budget, input="Ciao\n"))
    finally:
        budget.ledger_path.chmod(0o600)

    assert line == f"error: {budget.ledger_path}: cannot be read (Permission denied)"
    with budget.exclusive():
        pass
