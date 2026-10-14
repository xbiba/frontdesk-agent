"""Real conversations never reach a route approved for evaluation data only (M3 design §4.2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typer.testing import CliRunner

import frontdesk
from frontdesk.cli.main import app

runner = CliRunner()
SOURCE = Path(frontdesk.__file__).parent
NOWHERE = "postgresql+psycopg://nobody:x@127.0.0.1:1/none"
DATA_CLASS_IMPORT = "from frontdesk.model_gateway.routes import DataClass"
NAMES_EVALUATION_DATA = [
    pytest.param(f"{DATA_CLASS_IMPORT}\nx = DataClass.EVALUATION\n", id="the member"),
    pytest.param(f"{DATA_CLASS_IMPORT} as D\nx = D.EVALUATION\n", id="an alias of the class"),
    pytest.param(
        "from frontdesk.model_gateway import routes\nx = routes.DataClass.EVALUATION\n",
        id="through the module",
    ),
    pytest.param(f'{DATA_CLASS_IMPORT}\nx = DataClass("evaluation")\n', id="its value"),
    # A StrEnum member equals its value, so require_route accepts the plain string too.
    pytest.param('require_route("vertex-eu-opus", "evaluation")\n', id="the bare string"),
]
SAYS_NOTHING_OF_IT = [
    pytest.param(f"{DATA_CLASS_IMPORT}\nx = DataClass.PATIENT_DERIVED\n", id="patient data"),
    pytest.param(
        '"""Evaluation data is synthetic."""\n# DataClass.EVALUATION is for runs\nevaluation = 1\n',
        id="words about it",
    ),
]


def declares_evaluation_data(source: str) -> bool:
    """Whether a module names the evaluation data class, however it spells it: the attribute
    `EVALUATION` of anything (the class, an alias of it, a module path to it) or the string
    "evaluation" (the member's value; require_route takes it as it is). Words in comments and
    docstrings do not count."""
    return any(
        (isinstance(node, ast.Attribute) and node.attr == "EVALUATION")
        or (isinstance(node, ast.Constant) and node.value == "evaluation")
        for node in ast.walk(ast.parse(source))
    )


def declaring_modules(root: Path) -> list[str]:
    """The modules under `root`, as paths relative to it, that declare evaluation data."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if declares_evaluation_data(path.read_text(encoding="utf-8"))
    )


def test_a_conversation_never_reaches_a_route_approved_only_for_evaluation(
    tmp_path: Path, businesses_dir: Path
) -> None:
    env = tmp_path / "gemini.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={businesses_dir / 'demo-clinic'}\n"
        f"FRONTDESK_DATABASE_URL={NOWHERE}\n"
        f"FRONTDESK_MIGRATION_DATABASE_URL={NOWHERE}\n"
        "FRONTDESK_MODEL_ROUTE=vertex-eu-gemini-flash-lite\n"
        "FRONTDESK_GCP_PROJECT=demo-project\n"
        "FRONTDESK_EMBEDDING_BACKEND=hash\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["chat", "--env-file", str(env)], input="Ciao\n")
    assert result.exit_code == 1
    # Refused before the database is opened: its URL leads nowhere.
    assert "not approved for patient_derived" in result.stderr


def test_only_the_route_table_declares_evaluation_data() -> None:
    """Besides the route table, only a command that sends synthetic text may call it evaluation
    data. A module that starts to do so widens this list in its own task, on purpose. The live
    chat does: a person types test messages into it, and it says so before the first prompt. So
    does the live evaluation, which runs the business's labelled test messages, and so does the
    embedder experiment, which sends the pack's approved public text and the labelled retrieval
    questions to the embedding models it compares. And so does the Instagram channel in testers
    mode (M4 design D4): only the people who sent the pairing code reach the model, and they write
    test messages; in live mode the channel's messages are patient data."""
    declaring = declaring_modules(SOURCE)
    assert declaring == [
        "channels/wiring.py",
        "cli/experiment.py",
        "cli/live.py",
        "cli/live_chat.py",
        "model_gateway/routes.py",
    ]


@pytest.mark.parametrize("source", NAMES_EVALUATION_DATA)
def test_the_scan_finds_every_spelling_of_evaluation_data(source: str) -> None:
    assert declares_evaluation_data(source)


@pytest.mark.parametrize("source", SAYS_NOTHING_OF_IT)
def test_the_scan_ignores_words_about_evaluation_data(source: str) -> None:
    assert not declares_evaluation_data(source)


def test_a_module_that_aliases_the_data_class_is_reported(tmp_path: Path) -> None:
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "aliased.py").write_text(
        f"{DATA_CLASS_IMPORT} as D\nx = D.EVALUATION\n", encoding="utf-8"
    )
    (tmp_path / "cli" / "plain.py").write_text("x = 1\n", encoding="utf-8")
    assert declaring_modules(tmp_path) == ["cli/aliased.py"]
