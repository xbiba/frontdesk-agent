"""Ctrl-C at the prompt ends a live chat at once and gives the budget back (M3 Task 6b, fix round 1).

The chat holds the budget while it waits for a line, so a session that outlived its Ctrl-C would
keep every other live run out. The unit tests of the reader are in `test_live_costs.py`; these two
run the command: one with an interrupt raised where the line is read, one with a real SIGINT sent to
a real process.
"""

from __future__ import annotations

import logging
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

import pytest

from frontdesk.model_gateway.access import ENVIRONMENT_OVERRIDES
from tests.cli.live_chat_helpers import PRICE, Install, bootstrapped, live
from tests.model_gateway.budget_helpers import write_budget

TURN_LOGGER = logging.getLogger("frontdesk.conversation.turn")
EXIT_AFTER_CTRL_C = 130
# Python gives SIGINT its KeyboardInterrupt handler only if the shell that started pytest did not
# ignore it (a background job of a script does); the child asks for it, as an operator's has it.
# The free request that proves the credentials would go to Google: the child answers it itself.
ENTRY_POINT = (
    "import asyncio, signal; signal.signal(signal.SIGINT, signal.default_int_handler); "
    "import frontdesk.cli.live_common as common; "
    "common.check_route_access = lambda *args, **kwargs: asyncio.sleep(0); "
    "from frontdesk.cli.main import app; app()"
)


@pytest.mark.db
def test_an_interrupt_while_the_chat_waits_for_a_line_ends_it_with_130_and_frees_the_budget(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = gemini(PRICE)
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)

    def interrupted() -> str | None:
        raise KeyboardInterrupt  # what `asyncio.run` raises when Ctrl-C comes while a line is awaited

    monkeypatch.setattr("frontdesk.cli.main._read_line", interrupted)

    result = live(env, budget)

    assert result.exit_code == EXIT_AFTER_CTRL_C
    assert "Live: gemini-3.1-flash-lite" in result.stdout  # the session had started
    assert result.stderr == ""  # no traceback, no error, no stop
    assert sent == []
    with budget.exclusive():  # the budget is free for the next run
        pass
    assert (TURN_LOGGER.handlers, TURN_LOGGER.level) == ([], logging.NOTSET)


def _environment() -> dict[str, str]:
    """The environment of an operator's shell: no setting of ours, no Google variable, no proxy or
    certificate variable (a live chat refuses to start while one is set)."""
    kept = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("FRONTDESK_", "GOOGLE_", "CLOUDSDK_", "TYPER_", "_TYPER_"))
        and name.upper() not in ENVIRONMENT_OVERRIDES
    }
    return {**kept, "NO_COLOR": "1", "COLUMNS": "200", "PYTHONUNBUFFERED": "1"}


def _read_until(stream: IO[bytes], marker: bytes, timeout: float) -> bytes:
    seen = b""
    deadline = time.monotonic() + timeout
    while marker not in seen:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"{marker!r} not seen within {timeout} s; got {seen!r}"
        if select.select([stream], [], [], min(remaining, 1.0))[0]:
            chunk = os.read(stream.fileno(), 4096)
            assert chunk, f"the process ended before {marker!r}; got {seen!r}"
            seen += chunk
    return seen


@pytest.mark.db
def test_a_real_ctrl_c_at_the_prompt_ends_the_process_at_once_and_frees_the_budget(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    # The real entry point in a process of its own, as test_tracebacks.py runs it. The real Gemini
    # client is built and never asked anything: the person types nothing.
    env = bootstrapped(tmp_path, businesses_dir, database)
    budget = write_budget(tmp_path)
    errors = tmp_path / "stderr.txt"
    with errors.open("wb") as stderr:
        child = subprocess.Popen(  # noqa: S603 - this interpreter and a fixed entry point
            [
                sys.executable,
                "-c",
                ENTRY_POINT,
                *["chat", "--live", "--budget", str(budget.path), "--env-file", str(env)],
            ],
            stdin=subprocess.PIPE,  # open and silent: a person who has not typed yet
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=_environment(),
        )
        assert child.stdin is not None
        assert child.stdout is not None
        try:
            _read_until(child.stdout, b"type 'exit' to quit", timeout=120)
            time.sleep(0.5)  # the chat is now waiting for a line
            child.send_signal(signal.SIGINT)
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pytest.fail("the chat was still running 10 s after Ctrl-C")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdin.close()
            child.stdout.close()

    assert code == EXIT_AFTER_CTRL_C
    output = errors.read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in output
    assert "Fatal Python error" not in output  # a daemon thread left reading stdin is harmless
    with budget.exclusive():  # the lock went with the process
        pass
