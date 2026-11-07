"""The site's chat end to end on this computer (`frontdesk web serve`, design of 2026-11-07): the
widget's POST, the worker's run in the same process on the offline model, the reply read back by
GET; the test page embeds the widget; a live route is refused without --live, and served, metered, with it."""

import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from fastapi import FastAPI
from fastapi.testclient import TestClient

from frontdesk.channels.inbox import BURST
from frontdesk.cli.main import app
from frontdesk.cli.web import local_app
from frontdesk.model_gateway.routes import UnapprovedRouteError
from frontdesk.settings import Settings
from frontdesk.tenancy.loader import load_pack
from tests.cli.live_chat_helpers import (
    BANNER,
    NOWHERE,
    PRICE,
    QUESTION,
    Install,
    env_file,
    ledger,
    refused,
    runner,
)
from tests.model_gateway.budget_helpers import write_budget
from tests.tenancy.helpers import CLINIC, copy_pack, edit_yaml, web_section

ORIGIN = "http://localhost:8000"
SESSION = "0d9c1f7e-5a43-4b8e-a2f1-9e8d7c6b5a40"


@pytest.fixture
def web_dir(tmp_path: Path, businesses_dir: Path) -> Path:
    target = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(target / "business.yaml", lambda d: d.update(channels={"web": web_section()}))
    return target


def _settings(pack: Path, **fields: Any) -> Settings:
    return Settings(**{"business_pack": pack, "model_route": "offline-demo", **fields})


@pytest.mark.db
def test_a_message_posted_is_answered_and_read_back(
    web_dir: Path, app_url: str, activate_pack: Any
) -> None:
    activate_pack(load_pack(web_dir))
    served = local_app(_settings(web_dir, database_url=app_url))
    with TestClient(served) as client:
        page = client.get("/?lang=it")
        assert '<script src="/web/widget.js" data-lang="it" defer>' in page.text
        body = {"session": SESSION, "id": "m-1", "text": "Dove si trova la clinica?"}
        posted = client.post("/web/messages", json=body, headers={"origin": ORIGIN})
        assert posted.status_code == 202
        replies: list[dict[str, str]] = []
        deadline = time.monotonic() + BURST.total_seconds() + 20
        while time.monotonic() < deadline:
            found = client.get("/web/replies", params={"session": SESSION, "after": "0"}).json()
            replies = found["replies"]
            if replies and not found["waiting"]:
                break
            time.sleep(0.5)
    assert replies, "no reply within the burst and 20 seconds"
    pack = load_pack(web_dir)
    disclosure = pack.fixed_texts["it"]["disclosure"]  # type: ignore[index]
    assert any(disclosure in reply["text"] for reply in replies)


def test_a_live_route_is_refused(web_dir: Path) -> None:
    with pytest.raises(UnapprovedRouteError, match="offline model unless --live"):
        local_app(_settings(web_dir, model_route="anthropic-haiku-5-5"))


# --- `web serve --live`: the site's chat on the real model, paid from the budget ------------------


def _live_env(tmp_path: Path, pack: Path, urls: tuple[str, str] = (NOWHERE, NOWHERE)) -> Path:
    return env_file(tmp_path, pack, urls, name="web-live")


def _serve(env: Path, *flags: str) -> Result:
    return runner.invoke(app, ["web", "serve", "--env-file", str(env), *flags])


def test_live_needs_a_budget(tmp_path: Path, web_dir: Path) -> None:
    line = refused(_serve(_live_env(tmp_path, web_dir), "--live"))
    assert line.startswith("error: --live needs --budget")


def test_a_budget_goes_with_live_only(tmp_path: Path, web_dir: Path) -> None:
    budget = write_budget(tmp_path)
    line = refused(_serve(_live_env(tmp_path, web_dir), "--budget", str(budget.path)))
    assert line == "error: --budget only goes with --live"


def test_live_needs_a_live_route(tmp_path: Path, web_dir: Path) -> None:
    env = env_file(tmp_path, web_dir, route="offline-demo", project=None)
    line = refused(_serve(env, "--live", "--budget", str(write_budget(tmp_path).path)))
    assert line == "error: --live needs a live model route; FRONTDESK_MODEL_ROUTE is offline-demo"


@pytest.mark.db
def test_live_answers_the_widget_on_the_live_route_and_meters_each_request(
    tmp_path: Path,
    web_dir: Path,
    database: tuple[str, str],
    gemini: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = gemini(PRICE)
    env = _live_env(tmp_path, web_dir, database)
    done = runner.invoke(app, ["knowledge", "bootstrap-demo", "--env-file", str(env)])
    assert done.exit_code == 0, done.output
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)
    replies: list[dict[str, str]] = []

    def run(served: FastAPI, **_: Any) -> None:
        """Stands for uvicorn: one message through the widget's routes while the budget is held."""
        with TestClient(served) as client:
            body = {"session": SESSION, "id": "m-1", "text": QUESTION}
            assert client.post("/web/messages", json=body, headers={"origin": ORIGIN}).is_success
            deadline = time.monotonic() + BURST.total_seconds() + 20
            while time.monotonic() < deadline and not replies:
                found = client.get("/web/replies", params={"session": SESSION, "after": "0"})
                replies.extend(found.json()["replies"])
                time.sleep(0.5)

    monkeypatch.setattr("frontdesk.cli.web.uvicorn.run", run)
    result = _serve(env, "--live", "--budget", str(budget.path))

    assert result.exit_code == 0, result.output
    assert BANNER.format(path=budget.path, left="10.00", cap="10.00") in result.stdout
    assert "Fisioterapia: 30 EUR" in replies[0]["text"]
    assert sent == ["classifier"]
    proof, paid = ledger(budget)
    assert (proof["kind"], paid["kind"]) == ("preflight", "chat")
    assert paid["run"].startswith("web-")
    assert "cost: 0.0104 EUR" in result.stdout
