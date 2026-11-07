"""The site's chat built from the settings (design of 2026-11-07): the webhook's part, the worker's
part with the web policy and the limits, tester mode with Instagram's pairing code."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from frontdesk.channels.web import WEB
from frontdesk.channels.wiring import build_web_ingress, build_worker
from frontdesk.composition import open_deployment
from frontdesk.model_gateway.routes import UnapprovedRouteError
from frontdesk.settings import InstagramMode, Settings
from frontdesk.tenancy.loader import load_pack
from tests.channels.fakes import RecordingDispatcher
from tests.tenancy.helpers import CLINIC, copy_pack, edit_yaml, web_section

QUEUE = "projects/demo-clinic/locations/europe-west1/queues/frontdesk-demo"
WORKER = "https://frontdesk-demo-worker-123456789.europe-west1.run.app"
INVOKER = "frontdesk-demo-invoker@demo-clinic.iam.gserviceaccount.com"
WEB_CODE = "a-long-web-pairing-code-2026"


@pytest.fixture
def web_dir(tmp_path: Path, businesses_dir: Path) -> Path:
    target = copy_pack(businesses_dir, CLINIC, tmp_path)
    edit_yaml(target / "business.yaml", lambda d: d.update(channels={"web": web_section()}))
    return target


def _settings(pack: Path, **fields: Any) -> Settings:
    values: dict[str, Any] = {
        "business_pack": pack,
        "web_pairing_code": SecretStr(WEB_CODE),
        "email_sender_key": SecretStr("a-server-side-secret-of-32-bytes"),
        "tasks_queue": QUEUE,
        "worker_url": WORKER,
        "tasks_invoker": INVOKER,
        **fields,
    }
    return Settings(**values)


def test_the_site_s_ingress_is_built_from_the_settings(web_dir: Path) -> None:
    web = build_web_ingress(_settings(web_dir))
    assert web is not None
    assert web.channel.allows("https://clinic.example")
    assert web.off is False


def test_a_pack_without_the_site_has_no_ingress(businesses_dir: Path) -> None:
    assert build_web_ingress(_settings(businesses_dir / CLINIC)) is None


def test_tester_mode_needs_a_pairing_code_of_its_own(web_dir: Path) -> None:
    # Instagram's code never opens the site: the site's endpoint is public (security review).
    settings = _settings(
        web_dir, web_pairing_code=None, instagram_pairing_code=SecretStr("demo-prova-1234")
    )
    with pytest.raises(ValueError, match="frontdesk-web-pairing-code"):
        build_web_ingress(settings)


def test_a_short_pairing_code_is_refused(web_dir: Path) -> None:
    with pytest.raises(ValueError, match="at least 20 characters"):
        build_web_ingress(_settings(web_dir, web_pairing_code=SecretStr("1234")))


def test_everyone_mode_needs_no_code_and_a_route_for_patient_data(web_dir: Path) -> None:
    dispatcher = RecordingDispatcher()
    web = build_web_ingress(
        _settings(web_dir, web_mode=InstagramMode.LIVE, web_pairing_code=None),
        dispatcher=dispatcher,
    )
    assert web is not None
    assert web.dispatcher is dispatcher
    with pytest.raises(UnapprovedRouteError, match="patient_derived"):
        build_web_ingress(
            _settings(web_dir, web_mode=InstagramMode.LIVE, model_route="vertex-eu-gemini-flash")
        )


@pytest.mark.db
def test_the_worker_answers_the_site_without_booking(
    web_dir: Path, app_url: str, activate_pack: Any
) -> None:
    activate_pack(load_pack(web_dir))
    settings = _settings(web_dir, database_url=app_url)
    worker = build_worker(settings)
    assert set(worker.runners) == {WEB}
    assert set(worker.inbox) == {WEB}
    deployment = open_deployment(settings)
    assert deployment.pack.channels.web is not None


@pytest.mark.db
def test_the_worker_needs_the_key_of_the_digests(
    web_dir: Path, app_url: str, activate_pack: Any
) -> None:
    activate_pack(load_pack(web_dir))
    with pytest.raises(ValueError, match="frontdesk-email-sender-key"):
        build_worker(_settings(web_dir, database_url=app_url, email_sender_key=None))
