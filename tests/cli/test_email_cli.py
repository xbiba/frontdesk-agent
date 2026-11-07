"""`frontdesk email run`: a pass on a file mailbox with the deployment's database, dry or with a
scripted model in place of the live one; what it refuses to start without."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from frontdesk.cli import email as email_cli
from frontdesk.cli.main import app
from tests.email.helpers import Script, reply

PACK_EMAIL = "\nemail:\n  mailbox: info@example.com\n  testers_only: false\n"
BOX = {
    "threads": [
        {
            "id": "t1",
            "messages": [
                {"from": "Anna <anna@example.com>", "subject": "Fisio", "body": "Quanto costa?"}
            ],
        },
        {"id": "t2", "messages": [{"from": "noreply@example.com", "subject": "News", "body": "x"}]},
    ]
}


@pytest.fixture
def env_file(tmp_path: Path, businesses_dir: Path, app_url: str) -> Path:
    import shutil

    pack = tmp_path / "pack"
    shutil.copytree(businesses_dir / "demo-clinic", pack)
    with (pack / "business.yaml").open("a", encoding="utf-8") as file:
        file.write(PACK_EMAIL)
    box = tmp_path / "box.yaml"
    box.write_text(yaml.safe_dump(BOX), encoding="utf-8")
    env = tmp_path / "email.env"
    env.write_text(
        f"FRONTDESK_BUSINESS_PACK={pack}\nFRONTDESK_DATABASE_URL={app_url}\n"
        f"FRONTDESK_EMAIL_BACKEND=file\nFRONTDESK_EMAIL_FILE={box}\n"
        "FRONTDESK_EMAIL_SENDER_KEY=0123456789abcdef0123456789abcdef\n"
        "FRONTDESK_MODEL_ROUTE=anthropic-haiku-5-5\nFRONTDESK_ANTHROPIC_API_KEY=test-key\n",
        encoding="utf-8",
    )
    return env


@pytest.mark.db
def test_a_dry_run_says_what_it_would_do(env_file: Path) -> None:
    result = CliRunner().invoke(app, ["email", "run", "--env-file", str(env_file), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "t1 would_draft" in result.output
    assert "t2 ignored" in result.output
    assert not env_file.with_name("box.out.yaml").exists()


@pytest.mark.db
def test_a_pass_drafts_and_labels(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script(reply("Gentile Anna, [FIRMA]"))
    monkeypatch.setattr(email_cli, "build_live_model", lambda *a, **k: script.model())
    runner = CliRunner()
    result = runner.invoke(app, ["email", "run", "--env-file", str(env_file)])
    assert result.exit_code == 0, result.output
    assert "t1 drafted" in result.output
    out = yaml.safe_load(env_file.with_name("box.out.yaml").read_text(encoding="utf-8"))
    assert out["labels"] == {"t1": ["Bozza AI"], "t2": ["AI ignorata"]}
    again = runner.invoke(app, ["email", "run", "--env-file", str(env_file)])
    assert "t1 seen" in again.output  # the ledger in Postgres remembers it


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"FRONTDESK_EMAIL_SENDER_KEY": "short"}, "frontdesk-email-sender-key"),
        ({"FRONTDESK_EMAIL_BACKEND": "none"}, "FRONTDESK_EMAIL_BACKEND"),
        ({"FRONTDESK_EMAIL_BACKEND": "gmail"}, "FRONTDESK_EMAIL_DELEGATION_ACCOUNT"),
        ({"FRONTDESK_MODEL_ROUTE": "vertex-eu-gemini-flash"}, "patient_derived"),
    ],
)
@pytest.mark.db
def test_what_the_pass_refuses_to_start_without(
    env_file: Path, change: dict[str, str], named: str
) -> None:
    lines = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    lines.update(change)
    env_file.write_text("".join(f"{k}={v}\n" for k, v in lines.items()), encoding="utf-8")
    result = CliRunner().invoke(app, ["email", "run", "--env-file", str(env_file)])
    assert result.exit_code == 1
    assert named in result.output


def test_the_switch_turns_the_pass_off(tmp_path: Path) -> None:
    env = tmp_path / "off.env"
    env.write_text("FRONTDESK_EMAIL_OFF=true\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["email", "run", "--env-file", str(env)])
    assert result.exit_code == 0
    assert "off" in result.output


def test_the_email_commands_are_listed() -> None:
    result: Any = CliRunner().invoke(app, ["email", "--help"])
    assert "run" in result.output
    assert "login" in result.output


def test_gmail_is_reached_only_as_the_packs_mailbox(monkeypatch: pytest.MonkeyPatch) -> None:
    from frontdesk.settings import Settings
    from frontdesk.tenancy.pack import EmailChannel

    seen: dict[str, Any] = {}

    def delegated(scopes: Any, **options: Any) -> Any:
        seen.update(options)
        return lambda: "tok"

    monkeypatch.setattr(email_cli, "delegated_token_source", delegated)
    settings = Settings(
        email_backend="gmail", email_delegation_account="sa@demo.iam.gserviceaccount.com"
    )
    email_cli.build_mailbox(settings, EmailChannel(mailbox="info@example.com"))
    assert seen["subject"] == "info@example.com"
    assert seen["allowed_subjects"] == frozenset({"info@example.com"})
