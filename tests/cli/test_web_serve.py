"""The site's chat end to end on this computer (`frontdesk web serve`, design of 2026-11-07): the
widget's POST, the worker's run in the same process on the offline model, the reply read back by
GET; the test page embeds the widget; a live route is refused."""

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from frontdesk.channels.inbox import BURST
from frontdesk.cli.web import local_app
from frontdesk.model_gateway.routes import UnapprovedRouteError
from frontdesk.settings import Settings
from frontdesk.tenancy.loader import load_pack
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
    app = local_app(_settings(web_dir, database_url=app_url))
    with TestClient(app) as client:
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
    with pytest.raises(UnapprovedRouteError, match="offline model only"):
        local_app(_settings(web_dir, model_route="anthropic-haiku-5-5"))
