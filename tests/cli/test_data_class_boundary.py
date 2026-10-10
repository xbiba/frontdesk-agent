"""Real conversations never reach a route approved for evaluation data only (M3 design §4.2)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import frontdesk
from frontdesk.cli.main import app

runner = CliRunner()
SOURCE = Path(frontdesk.__file__).parent
NOWHERE = "postgresql+psycopg://nobody:x@127.0.0.1:1/none"


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
    data. A module that starts to do so widens this list in its own task, on purpose."""
    declaring = sorted(
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*.py")
        if "DataClass.EVALUATION" in path.read_text(encoding="utf-8")
    )
    assert declaring == ["model_gateway/routes.py"]
