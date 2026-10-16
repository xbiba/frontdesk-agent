"""The Instagram channel built from the settings (M4 design D4, D7): what it refuses to start
without, and live mode refused on a route not approved for patient data."""

import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from frontdesk.channels.wiring import build_ingress, build_worker
from frontdesk.model_gateway.routes import UnapprovedRouteError
from frontdesk.settings import InstagramMode, Settings
from frontdesk.tenancy.loader import load_pack

QUEUE = "projects/clinica-mcu/locations/europe-west1/queues/frontdesk-mcu"
WORKER = "https://frontdesk-mcu-worker-123456789.europe-west1.run.app"
INVOKER = "frontdesk-mcu-invoker@clinica-mcu.iam.gserviceaccount.com"
ACCOUNT = "17841400000000001"


@pytest.fixture
def pack_dir(tmp_path: Path, businesses_dir: Path) -> Path:
    target = tmp_path / "demo-clinic"
    shutil.copytree(businesses_dir / "demo-clinic", target)
    with (target / "business.yaml").open("a", encoding="utf-8") as file:
        file.write("\nchannels:\n  instagram: {}\n")
    return target


def _settings(pack: Path, **fields: Any) -> Settings:
    values: dict[str, Any] = {
        "business_pack": pack,
        "instagram_account_id": ACCOUNT,
        "instagram_app_secret": SecretStr("app-secret"),
        "instagram_verify_token": SecretStr("verify-me"),
        "instagram_pairing_code": SecretStr("mcu-prova-1234"),
        "instagram_token": SecretStr("t0ken"),
        "tasks_queue": QUEUE,
        "worker_url": WORKER,
        "tasks_invoker": INVOKER,
        **fields,
    }
    return Settings(**values)


def test_the_webhook_side_is_built_from_the_settings(pack_dir: Path) -> None:
    ingress = build_ingress(_settings(pack_dir))
    assert ingress.verify_token == "verify-me"
    assert ingress.off is False
    assert ingress.webhook.name == "instagram"


def test_a_pack_without_instagram_is_refused(businesses_dir: Path) -> None:
    with pytest.raises(ValueError, match="does not enable Instagram"):
        build_ingress(_settings(businesses_dir / "demo-clinic"))


def test_testers_mode_needs_the_pairing_code(pack_dir: Path) -> None:
    with pytest.raises(ValueError, match="frontdesk-instagram-pairing-code"):
        build_ingress(_settings(pack_dir, instagram_pairing_code=None))


@pytest.mark.parametrize(
    ("field", "named"),
    [
        ("instagram_account_id", "FRONTDESK_INSTAGRAM_ACCOUNT_ID"),
        ("tasks_queue", "FRONTDESK_TASKS_QUEUE"),
    ],
)
def test_the_channel_needs_its_account_and_its_queue(
    pack_dir: Path, field: str, named: str
) -> None:
    with pytest.raises(ValueError, match=named):
        build_ingress(_settings(pack_dir, **{field: None}))


def test_live_mode_is_refused_on_a_route_not_approved_for_patient_data(pack_dir: Path) -> None:
    settings = _settings(
        pack_dir,
        instagram_mode=InstagramMode.LIVE,
        model_route="vertex-eu-gemini-flash",
        gcp_project="clinica-mcu",
    )
    with pytest.raises(UnapprovedRouteError, match="patient_derived"):
        build_worker(settings)
    # Final review of M4, M6: the webhook refuses it too, or it would store patients' messages that
    # nobody answers until the purge.
    with pytest.raises(UnapprovedRouteError, match="patient_derived"):
        build_ingress(settings)


@pytest.mark.db
def test_the_worker_side_is_built_for_testers(
    pack_dir: Path, app_url: str, activate_pack: Any
) -> None:
    activate_pack(load_pack(pack_dir))
    worker = build_worker(_settings(pack_dir, database_url=app_url))
    assert set(worker.runners) == {"instagram"}
    assert worker.requests is None
