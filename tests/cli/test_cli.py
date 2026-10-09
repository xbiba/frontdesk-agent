import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, text
from typer.testing import CliRunner

from frontdesk.cli.main import app
from frontdesk.composition import open_deployment
from frontdesk.knowledge.embedders import HashEmbedder

pytestmark = pytest.mark.db
runner = CliRunner()


class OtherEmbedder(HashEmbedder):
    identity = "other-v1"


def _env(tmp_path: Path, name: str, pack_dir: Path, database: tuple[str, str]) -> Path:
    owner_url, app_url = database
    env = tmp_path / f"{name}.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={pack_dir}\n"
        f"FRONTDESK_DATABASE_URL={app_url}\n"
        f"FRONTDESK_MIGRATION_DATABASE_URL={owner_url}\n"
        "FRONTDESK_MODEL_ROUTE=offline-demo\n"
        "FRONTDESK_EMBEDDING_BACKEND=hash\n",
        encoding="utf-8",
    )
    return env


def _run(*args: str, input: str | None = None) -> str:
    result = runner.invoke(app, list(args), input=input)
    assert result.exit_code == 0, result.output
    return result.output


def test_knowledge_lifecycle_through_the_cli(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", database))
    drafted = _run("knowledge", "draft", "--env-file", env)
    version = re.search(r"draft: ([0-9a-f-]{36})", drafted).group(1)  # type: ignore[union-attr]
    assert '- added: price:physiotherapy\n  - amount: "30"' in drafted
    _run(
        "knowledge",
        "approve",
        "--env-file",
        env,
        "--version",
        version,
        "--by",
        "Owner",
        "--evidence",
        "signed sheet",
    )
    _run("knowledge", "activate", "--env-file", env, "--version", version)
    chat = _run("chat", "--env-file", env, input="Quanto costa la fisioterapia?\nexit\n")
    assert "30 EUR" in chat
    assert "intent=price" in chat
    assert "sources=price:physiotherapy" in chat


def test_chat_without_knowledge_explains_what_to_run(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", database))
    result = runner.invoke(app, ["chat", "--env-file", env], input="ciao\n")
    assert result.exit_code == 1
    assert "bootstrap-demo" in result.output


def test_bootstrap_refuses_real_businesses(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    pack_dir = tmp_path / "real-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", pack_dir)
    business = yaml.safe_load((pack_dir / "business.yaml").read_text(encoding="utf-8"))
    business["id"] = "real-clinic"
    (pack_dir / "business.yaml").write_text(
        yaml.safe_dump(business, allow_unicode=True), encoding="utf-8"
    )
    env = str(_env(tmp_path, "real", pack_dir, database))
    result = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env])
    assert result.exit_code == 1
    assert "only for demo businesses" in result.output


def test_bootstrap_finishes_a_pending_draft(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", database))
    drafted = _run("knowledge", "draft", "--env-file", env)
    version = re.search(r"draft: ([0-9a-f-]{36})", drafted).group(1)  # type: ignore[union-attr]
    assert f"active: {version}" in _run("knowledge", "bootstrap-demo", "--env-file", env)
    assert "30 EUR" in _run(
        "chat", "--env-file", env, input="Quanto costa la fisioterapia?\nexit\n"
    )


def test_two_deployments_stay_separate(
    tmp_path: Path, businesses_dir: Path, new_database: Callable[[], tuple[str, str]]
) -> None:
    clinic = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", new_database()))
    shop = str(_env(tmp_path, "shop", businesses_dir / "demo-shop", new_database()))
    _run("knowledge", "bootstrap-demo", "--env-file", clinic)
    _run("knowledge", "bootstrap-demo", "--env-file", shop)
    order = "Dov'è il mio ordine ORD-10427? CAP 20121\nexit\n"
    assert "spedito" in _run("chat", "--env-file", shop, input=order)
    assert "spedito" not in _run("chat", "--env-file", clinic, input=order)
    assert "30 EUR" not in _run(
        "chat", "--env-file", shop, input="Quanto costa la fisioterapia?\nexit\n"
    )


def test_a_changed_embedder_is_refused_until_reembed(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    monkeypatch.setattr("frontdesk.composition.build_embedder", lambda backend: OtherEmbedder())
    refused = runner.invoke(app, ["chat", "--env-file", env], input="ciao\n")
    assert refused.exit_code == 1
    assert "reembed" in refused.output
    assert "other-v1" in _run("knowledge", "reembed", "--env-file", env)
    assert "30 EUR" in _run(
        "chat", "--env-file", env, input="Quanto costa la fisioterapia?\nexit\n"
    )


def test_retrieval_evaluation_command(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    report = tmp_path / "report.json"
    output = _run("eval", "retrieval", "--env-file", env, "--output", str(report))
    assert "| case | language | recall@5 | RR |" in output
    assert report.exists()


# --- Controller rulings for Task 23 ---------------------------------------------------------------

# No connection is ever opened with these: the tests that use them stop before the database.
NO_DATABASE = (
    "postgresql+psycopg://nobody:nobody@127.0.0.1:1/none",
    "postgresql+psycopg://nobody:nobody@127.0.0.1:1/none",
)


def _copy_pack(tmp_path: Path, businesses_dir: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(businesses_dir / name, target)
    return target


def _fixed_clock(monkeypatch: pytest.MonkeyPatch, instant: datetime) -> None:
    """Every deployment the CLI opens from now on reads `instant` as the current time."""
    real = open_deployment
    monkeypatch.setattr(
        "frontdesk.cli.main.open_deployment",
        lambda settings: replace(real(settings), now=lambda: instant),
    )


def test_retrieval_evaluation_dates_documents_by_the_business_day(
    tmp_path: Path,
    businesses_dir: Path,
    database: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack_dir = _copy_pack(tmp_path, businesses_dir, "demo-clinic")
    parking = pack_dir / "knowledge" / "parking-en.md"
    parking.write_text(
        parking.read_text(encoding="utf-8").replace(
            "language: en\n", "language: en\nvalid_from: 2031-03-15\n", 1
        ),
        encoding="utf-8",
    )
    env = str(_env(tmp_path, "clinic", pack_dir, database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)

    def parking_recall(instant: datetime) -> float:
        _fixed_clock(monkeypatch, instant)
        report = tmp_path / "report.json"
        _run("eval", "retrieval", "--env-file", env, "--output", str(report))
        results = json.loads(report.read_text(encoding="utf-8"))["results"]
        return float(next(r["recall"] for r in results if r["case"]["id"] == "parking-en"))

    # Both instants are March 14 in UTC. In Tirane the first is still March 14 (the document is not
    # valid yet) and the second is already March 15.
    assert parking_recall(datetime(2031, 3, 14, 22, 30, tzinfo=UTC)) == 0.0
    assert parking_recall(datetime(2031, 3, 14, 23, 30, tzinfo=UTC)) == 1.0


def test_a_broken_retrieval_dataset_is_reported_on_stderr_with_exit_code_1(
    tmp_path: Path, businesses_dir: Path
) -> None:
    pack_dir = _copy_pack(tmp_path, businesses_dir, "demo-clinic")
    (pack_dir / "evals" / "retrieval.yaml").write_text(
        '- id: broken-case\n  question: "Is there parking?"\n  language: en\n', encoding="utf-8"
    )
    env = str(_env(tmp_path, "clinic", pack_dir, NO_DATABASE))
    result = runner.invoke(app, ["eval", "retrieval", "--env-file", env])
    assert result.exit_code == 1
    assert "case 'broken-case'" in result.stderr
    assert "relevant" in result.stderr
    assert str(pack_dir / "evals" / "retrieval.yaml") in result.stderr
    assert "case 'broken-case'" not in result.stdout
    assert "Traceback" not in result.output


def test_the_cli_keeps_the_pydantic_ai_banner_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "set-and-removed")  # makes monkeypatch restore it
    monkeypatch.delenv("PYDANTIC_AI_NO_BANNER")
    assert runner.invoke(app, ["db", "--help"]).exit_code == 0
    assert os.environ["PYDANTIC_AI_NO_BANNER"] == "1"


def test_the_cli_leaves_an_existing_banner_setting_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "kept")
    assert runner.invoke(app, ["db", "--help"]).exit_code == 0
    assert os.environ["PYDANTIC_AI_NO_BANNER"] == "kept"


def test_the_chat_trace_never_shows_an_order_reference(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "shop", businesses_dir / "demo-shop", database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    output = _run(
        "chat", "--env-file", env, input="Dov'è il mio ordine ORD-10427? CAP 20121\nexit\n"
    )
    assert "Ordine ORD-10427: spedito" in output  # the reply to the person names the order
    traces = [line for line in output.splitlines() if line.lstrip().startswith("[rule=")]
    assert len(traces) == 1
    assert "intent=order_status" in traces[0]
    assert "sources=order " in traces[0]
    assert "ORD-" not in traces[0]
    assert "10427" not in traces[0]


# --- The demo answers documents with quotes that pass the claim checks ------------------------------


def test_the_clinic_answers_a_document_question_with_a_checked_quote(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "clinic", businesses_dir / "demo-clinic", database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    output = _run("chat", "--env-file", env, input="Is there parking near the clinic?\nexit\n")
    assert "Klinika Demo has six free parking spaces behind the building." in output
    assert "sources=parking-en" in output
    assert "outcome=replied" in output


def test_the_shop_answers_a_document_question_with_a_checked_quote(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "shop", businesses_dir / "demo-shop", database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    output = _run("chat", "--env-file", env, input="Posso fare un reso?\nexit\n")
    assert "Puoi restituire un articolo entro quattordici giorni dalla consegna" in output
    assert "sources=returns-it" in output
    assert "outcome=replied" in output


# --- bootstrap-demo approves by itself only synthetic data: "demo-" id and isolation level T1 ------


def _edit_business(pack_dir: Path, **changes: str) -> None:
    path = pack_dir / "business.yaml"
    business = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump({**business, **changes}, allow_unicode=True), encoding="utf-8")


def _knowledge_versions(owner_url: str) -> int:
    """Every knowledge version in the database, of any business and status (the owner sees all)."""
    engine = create_engine(owner_url)
    try:
        with engine.connect() as connection:
            count = connection.execute(text("SELECT count(*) FROM knowledge_versions")).scalar_one()
    finally:
        engine.dispose()
    return int(count)


@pytest.mark.parametrize("level", ["T2", "T3"])
def test_bootstrap_refuses_a_demo_named_pack_that_declares_more_than_demo_data(
    level: str, tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    pack_dir = _copy_pack(tmp_path, businesses_dir, "demo-clinic")
    _edit_business(pack_dir, id="demo-clinic-live", isolation_level=level)
    env = str(_env(tmp_path, "live", pack_dir, database))
    result = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env])
    assert result.exit_code == 1
    assert result.stdout == ""  # no "active: ..." line
    assert "only for demo businesses" in result.stderr
    assert "demo-clinic-live" in result.stderr
    assert f"isolation level {level}" in result.stderr
    assert "T1" in result.stderr
    assert "recorded approval" in result.stderr
    assert _knowledge_versions(database[0]) == 0  # nothing was drafted, approved or activated


def test_bootstrap_still_serves_a_demo_pack_that_declares_t1(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    pack_dir = _copy_pack(tmp_path, businesses_dir, "demo-clinic")
    _edit_business(pack_dir, id="demo-clinic-copy")  # still "demo-" and still T1
    env = str(_env(tmp_path, "copy", pack_dir, database))
    assert "active: " in _run("knowledge", "bootstrap-demo", "--env-file", env)
    assert _knowledge_versions(database[0]) == 1


def test_bootstrap_names_what_is_wrong_with_a_pack_that_is_not_a_demo(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    pack_dir = _copy_pack(tmp_path, businesses_dir, "demo-clinic")
    _edit_business(pack_dir, id="real-clinic")  # T1, but not named like a demo
    env = str(_env(tmp_path, "real", pack_dir, database))
    result = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", env])
    assert result.exit_code == 1
    assert "only for demo businesses" in result.stderr
    assert "real-clinic" in result.stderr
    assert "demo-" in result.stderr
    assert _knowledge_versions(database[0]) == 0


# --- eval turns: labelled conversation turns through the whole pipeline -------------------------------


def test_turn_evaluation_command(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "shop", businesses_dir / "demo-shop", database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    report = tmp_path / "turns.json"
    output = _run("eval", "turns", "--env-file", env, "--output", str(report))
    assert "| case | language | outcome | result |" in output
    assert "FAIL" not in output
    assert report.exists()


def test_a_failing_case_fails_the_turn_evaluation_but_the_report_is_still_written(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    pack_dir = _copy_pack(tmp_path, businesses_dir, "demo-shop")
    with (pack_dir / "evals" / "turns.yaml").open("a", encoding="utf-8") as dataset:
        dataset.write(
            '- id: wrong-on-purpose\n  message: "Posso fare un reso?"\n  expect: {outcome: handoff}\n'
        )
    env = str(_env(tmp_path, "shop", pack_dir, database))
    _run("knowledge", "bootstrap-demo", "--env-file", env)
    report = tmp_path / "turns.json"
    result = runner.invoke(app, ["eval", "turns", "--env-file", env, "--output", str(report)])
    assert result.exit_code == 1
    assert "| wrong-on-purpose | it | replied | FAIL: outcome: replied |" in result.stdout
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["cases"] == written["passed"] + 1
    failed = [entry["case_id"] for entry in written["results"] if not entry["passed"]]
    assert failed == ["wrong-on-purpose"]


def test_turn_evaluation_without_knowledge_explains_what_to_run(
    tmp_path: Path, businesses_dir: Path, database: tuple[str, str]
) -> None:
    env = str(_env(tmp_path, "shop", businesses_dir / "demo-shop", database))
    report = tmp_path / "turns.json"
    result = runner.invoke(app, ["eval", "turns", "--env-file", env, "--output", str(report)])
    assert result.exit_code == 1
    assert "bootstrap-demo" in result.stderr
    assert not report.exists()
