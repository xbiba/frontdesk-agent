"""No function of the live evaluation is long (the project guideline: under 50 lines).

`criteria` and `render_report` were 68 and 54 lines (Task 7 review, M10); the long ones are split
into helpers that each do one thing, so a reader sees a table's rows apart from its layout.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

LIVE_EVAL = Path(__file__).resolve().parents[2] / "src" / "frontdesk" / "live_eval"
LIMIT = 50


def lengths(path: Path) -> Iterator[tuple[str, int]]:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node.name, (node.end_lineno or node.lineno) - node.lineno + 1


@pytest.mark.parametrize("path", sorted(LIVE_EVAL.glob("*.py")), ids=lambda path: path.name)
def test_no_function_is_50_lines_or_longer(path: Path) -> None:
    assert [(name, size) for name, size in lengths(path) if size >= LIMIT] == []
