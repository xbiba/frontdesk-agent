"""What the CLI prints when something unexpected fails: never the settings of the deployment.

`CliRunner` calls the command without Typer's exception hook, so it cannot show what an operator
sees. These tests run the real entry point in a process of its own.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

FAKE_PASSWORD = "hunter2-fake-3f9c"
# Nothing listens on port 1 of the loopback address: connecting fails at once, and the error is not
# one the CLI handles, so Typer prints a traceback of the frames that hold the settings.
UNREACHABLE_DATABASE = f"postgresql+psycopg://frontdesk_app:{FAKE_PASSWORD}@127.0.0.1:1/none"
ENTRY_POINT = "from frontdesk.cli.main import app; app()"


def _clean_environment() -> dict[str, str]:
    """The environment of an operator's shell, minus whatever would change what is printed.

    Settings in the environment beat the env file, and `TYPER_STANDARD_TRACEBACK` switches Typer's
    own traceback (the one that can show locals) off. A wide terminal keeps Rich from wrapping a
    long value across lines, where a search for it would miss it.
    """
    kept = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("FRONTDESK_", "TYPER_", "_TYPER_"))
    }
    return {**kept, "COLUMNS": "500", "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"}


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - this interpreter and a fixed script, no untrusted input
        [sys.executable, "-c", ENTRY_POINT, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_clean_environment(),
        timeout=120,
        check=False,
    )


def test_an_unexpected_error_prints_a_traceback_without_the_settings(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env_file = tmp_path / "clinic.env"
    env_file.write_text(
        f"FRONTDESK_BUSINESS_PACK={businesses_dir / 'demo-clinic'}\n"
        f"FRONTDESK_DATABASE_URL={UNREACHABLE_DATABASE}\n"
        f"FRONTDESK_MIGRATION_DATABASE_URL={UNREACHABLE_DATABASE}\n"
        "FRONTDESK_MODEL_ROUTE=offline-demo\n"
        "FRONTDESK_EMBEDDING_BACKEND=hash\n",
        encoding="utf-8",
    )
    completed = _run_cli("knowledge", "draft", "--env-file", str(env_file))
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1
    # A real traceback of the failed connection, so that the checks below are not vacuous: the
    # frames of `knowledge draft` hold the Deployment, which holds the Settings.
    assert "Traceback (most recent call last)" in output
    assert "OperationalError" in output
    assert "knowledge_draft" in output
    assert FAKE_PASSWORD not in output
    assert UNREACHABLE_DATABASE not in output
