"""A developer's exported FRONTDESK_* variables must not reach the tests.

Settings let the environment override an env file, so an exported database URL would receive the
writes of every CLI test that names its own env file. `conftest.py` clears the variables for each
test; this runs a probe test in a process that has them exported, as an operator's shell would.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "tests" / "_probe_environment.py"
EXPORTED = {
    "FRONTDESK_DATABASE_URL": "postgresql+psycopg://operator:not-a-real-secret@db.example/prod",
    "FRONTDESK_MIGRATION_DATABASE_URL": "postgresql+psycopg://owner:not-a-real-secret@db.example/p",
    "FRONTDESK_MODEL_ROUTE": "vertex-eu-opus",
    "FRONTDESK_BUSINESS_PACK": "businesses/demo-shop",
}


def test_exported_frontdesk_variables_never_reach_a_test() -> None:
    result = subprocess.run(  # noqa: S603 - this interpreter and a fixed probe file
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(PROBE)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **EXPORTED},
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
