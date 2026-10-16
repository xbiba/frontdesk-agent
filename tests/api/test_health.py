import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from frontdesk.api.app import create_app, create_webhook_app
from frontdesk.channels.inbox import Inbox
from frontdesk.channels.instagram import InstagramWebhook
from frontdesk.channels.wiring import Ingress
from frontdesk.settings import Settings
from tests.channels.fakes import NOW, RecordingDispatcher


@pytest.mark.db
def test_health_is_ok_when_database_answers(app_url: str) -> None:
    client = TestClient(create_app(Settings(), engine=create_engine(app_url)))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_unavailable_without_database() -> None:
    engine = create_engine(
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none",
        connect_args={"connect_timeout": 1},
    )
    client = TestClient(create_app(Settings(), engine=engine))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_the_public_webhook_s_health_never_opens_a_database_connection() -> None:
    # Final review of M4, M10: anyone may call it, and each call took a connection of the pool the
    # worker's turns share.
    engine = create_engine(
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none",
        connect_args={"connect_timeout": 1},
    )
    inbox = Inbox(
        sessionmaker(),
        business_id="demo-clinic",
        channel="instagram",
        account_id="1",
        testers_only=True,
        pairing_code="code",
        now=lambda: NOW,
    )
    ingress = Ingress(
        webhook=InstagramWebhook("secret"),
        inbox=inbox,
        dispatcher=RecordingDispatcher(),
        verify_token="verify",
        off=False,
    )
    app = create_webhook_app(Settings(), ingress=ingress, engine=engine)
    response = TestClient(app).get("/health")
    assert (response.status_code, response.json()) == (200, {"status": "ok"})
