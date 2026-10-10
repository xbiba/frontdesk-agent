"""The composition root: what a deployment wires for its business."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models import Model
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.prompts import prompt_version, read_prompt
from frontdesk.composition import Deployment, build_turn_runtime, open_deployment
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, handle_message
from frontdesk.knowledge.embedders import HashEmbedder
from frontdesk.model_gateway.routes import UnapprovedRouteError
from frontdesk.settings import EmbeddingBackend, Settings
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.pack import BusinessPack, Profile

Activate = Callable[..., sessionmaker[Session]]
NO_DATABASE = "postgresql+psycopg://nobody:nobody@127.0.0.1:1/none"  # never connected to


def _deployment(
    pack: BusinessPack,
    factory: sessionmaker[Session] | None = None,
    *,
    now: Callable[[], datetime] | None = None,
    app_url: str = NO_DATABASE,
    model_route: str = "offline-demo",
    **settings: Any,
) -> Deployment:
    configured = Settings(
        business_pack=Path("businesses") / pack.id,
        database_url=app_url,
        model_route=model_route,
        embedding_backend=EmbeddingBackend.HASH,
        **settings,
    )
    deployment = Deployment(configured, pack, factory or sessionmaker(), HashEmbedder())
    return deployment if now is None else replace(deployment, now=now)


# --- The business day -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("timezone", "instant", "local_day"),
    [
        pytest.param(
            "Europe/Tirane",
            datetime(2031, 3, 14, 23, 30, tzinfo=UTC),
            date(2031, 3, 15),
            id="ahead-of-utc",
        ),
        pytest.param(
            "Europe/Tirane",
            datetime(2031, 7, 14, 21, 30, tzinfo=UTC),
            date(2031, 7, 14),
            id="summer-time-before-local-midnight",
        ),
        pytest.param(
            "Europe/Tirane",
            datetime(2031, 7, 14, 22, 30, tzinfo=UTC),
            date(2031, 7, 15),
            id="summer-time-after-local-midnight",
        ),
        pytest.param(
            "Europe/Tirane",
            datetime(2031, 10, 26, 22, 30, tzinfo=UTC),
            date(2031, 10, 26),
            id="the-day-summer-time-ends-before-local-midnight",
        ),
        pytest.param(
            "Europe/Tirane",
            datetime(2031, 10, 26, 23, 30, tzinfo=UTC),
            date(2031, 10, 27),
            id="the-day-summer-time-ends-after-local-midnight",
        ),
        pytest.param(
            "America/Los_Angeles",
            datetime(2031, 7, 15, 2, 0, tzinfo=UTC),
            date(2031, 7, 14),
            id="behind-utc",
        ),
        pytest.param(
            "Europe/Rome",
            datetime(2031, 3, 14, 12, 0, tzinfo=UTC),
            date(2031, 3, 14),
            id="same-day",
        ),
    ],
)
def test_today_is_the_calendar_date_in_the_zone_of_the_business(
    clinic_pack: BusinessPack, timezone: str, instant: datetime, local_day: date
) -> None:
    pack = clinic_pack.model_copy(update={"timezone": timezone})
    assert _deployment(pack, now=lambda: instant).today() == local_day


def test_the_server_day_and_the_business_day_can_differ(clinic_pack: BusinessPack) -> None:
    instant = datetime(2031, 3, 14, 23, 30, tzinfo=UTC)
    assert instant.date() == date(2031, 3, 14)  # the day on a server that keeps UTC
    assert _deployment(clinic_pack, now=lambda: instant).today() == date(2031, 3, 15)


def test_a_deployment_reads_the_real_clock_unless_told_otherwise(clinic_pack: BusinessPack) -> None:
    deployment = _deployment(clinic_pack)
    assert deployment.now().utcoffset() is not None  # an aware time
    assert abs((datetime.now(UTC) - deployment.now()).total_seconds()) < 5


def _with_price_valid_from(pack: BusinessPack, service_id: str, day: date) -> BusinessPack:
    prices = [
        price.model_copy(update={"valid_from": day}) if price.service_id == service_id else price
        for price in pack.prices
    ]
    return pack.model_copy(update={"prices": prices})


@pytest.mark.db
async def test_prices_begin_at_midnight_in_the_business_zone_not_on_the_server(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    pack = _with_price_valid_from(clinic_pack, "physiotherapy", date(2031, 3, 15))
    factory = activate_pack(pack)

    async def physio_price(instant: datetime) -> str:
        runtime = build_turn_runtime(
            _deployment(pack, factory, now=lambda: instant, app_url=app_url)
        )
        result, _ = await handle_message(
            "Quanto costa la fisioterapia?", ConversationState(turns=1), runtime
        )
        return result.text

    # Both instants are March 14 in UTC. In Tirane the first is March 14, 23:30 and the second
    # March 15, 00:30, the first minutes of the day on which the price takes effect.
    before = await physio_price(datetime(2031, 3, 14, 22, 30, tzinfo=UTC))
    after = await physio_price(datetime(2031, 3, 14, 23, 30, tzinfo=UTC))
    assert "30 EUR" not in before
    assert after.startswith("Fisioterapia: 30 EUR (a seduta)")


@pytest.mark.db
def test_the_runtime_dates_by_the_business_day_unless_a_date_is_given(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    instant = datetime(2031, 3, 14, 23, 30, tzinfo=UTC)
    deployment = _deployment(clinic_pack, factory, now=lambda: instant, app_url=app_url)
    runtime = build_turn_runtime(deployment)
    assert runtime.today() == date(2031, 3, 15)
    assert runtime.now() == instant  # one clock for the business day and for the order lookups
    explicit = build_turn_runtime(deployment, today=lambda: date(2030, 1, 1))
    assert explicit.today() == date(2030, 1, 1)


# --- The prompts are read once, when the runtime is built ----------------------------------------


@pytest.mark.db
def test_the_runtime_keeps_the_versions_of_the_prompts_it_was_built_with(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack, agents_copy: Path
) -> None:
    factory = activate_pack(clinic_pack)
    versions = {
        name: read_prompt(Profile.RECEPTIONIST, name).version
        for name in ("classifier", "responder")
    }
    runtime = build_turn_runtime(_deployment(clinic_pack, factory, app_url=app_url))
    prompts = agents_copy / Profile.RECEPTIONIST.value / "prompts"
    (prompts / "classifier.md").write_text("Edited after the build.\n", encoding="utf-8")
    (prompts / "responder.md").unlink()
    assert runtime.classifier_prompt_version == versions["classifier"]
    assert runtime.responder_prompt_version == versions["responder"]
    assert prompt_version(Profile.RECEPTIONIST, "classifier") != versions["classifier"]


@pytest.mark.db
def test_each_profile_gets_the_versions_of_its_own_prompts(
    activate_pack: Activate, app_url: str, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    runtime = build_turn_runtime(_deployment(shop_pack, factory, app_url=app_url))
    assert runtime.classifier_prompt_version == prompt_version(Profile.SUPPORT_SALES, "classifier")
    assert runtime.responder_prompt_version == prompt_version(Profile.SUPPORT_SALES, "responder")


@pytest.mark.parametrize("name", ["classifier", "responder"])
def test_a_blank_prompt_stops_the_build_before_the_database_is_touched(
    clinic_pack: BusinessPack, agents_copy: Path, name: str
) -> None:
    (agents_copy / Profile.RECEPTIONIST.value / "prompts" / f"{name}.md").write_text(
        " \n\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=rf"receptionist/prompts/{name}\.md"):
        build_turn_runtime(_deployment(clinic_pack))  # the database URL leads nowhere


# --- What is wired ---------------------------------------------------------------------------------


def test_open_deployment_loads_the_pack_and_the_embedder(businesses_dir: Path) -> None:
    settings = Settings(
        business_pack=businesses_dir / "demo-clinic",
        database_url=NO_DATABASE,
        embedding_backend=EmbeddingBackend.HASH,
    )
    deployment = open_deployment(settings)  # opens no connection
    assert deployment.pack.id == "demo-clinic"
    assert deployment.embedder.identity == "hash-v1"
    assert deployment.settings is settings


@pytest.mark.db
def test_a_clinic_never_builds_the_order_store(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    runtime = build_turn_runtime(_deployment(clinic_pack, factory, app_url=app_url))
    assert runtime.orders is None


@pytest.mark.db
def test_a_shop_gets_the_orders_of_its_pack(
    activate_pack: Activate, app_url: str, shop_pack: BusinessPack
) -> None:
    factory = activate_pack(shop_pack)
    runtime = build_turn_runtime(_deployment(shop_pack, factory, app_url=app_url))
    assert runtime.orders == OrderStore(tuple(shop_pack.orders))


@pytest.mark.db
def test_the_settings_reach_the_runtime(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    strict = build_turn_runtime(
        _deployment(
            clinic_pack,
            factory,
            app_url=app_url,
            min_support_similarity=0.7,
            telemetry_capture_content=True,
            min_retrieval_similarity=0.35,
        )
    )
    assert (strict.min_support_similarity, strict.capture_content) == (0.7, True)
    assert strict.min_retrieval_similarity == 0.35
    loose = build_turn_runtime(
        _deployment(
            clinic_pack,
            factory,
            app_url=app_url,
            min_support_similarity=0.4,
            telemetry_capture_content=False,
        )
    )
    assert (loose.min_support_similarity, loose.capture_content) == (0.4, False)


@pytest.mark.db
def test_the_vertex_route_builds_live_models_without_a_request(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    runtime = build_turn_runtime(
        _deployment(
            clinic_pack,
            factory,
            app_url=app_url,
            model_route="vertex-eu-opus",
            gcp_project="demo-project",
        )
    )
    classifier_settings = runtime.classifier.model_settings
    responder_settings = runtime.responder.model_settings
    for agent in (runtime.classifier, runtime.responder):
        assert isinstance(agent.model, Model)
        assert agent.model.model_name == "claude-opus-5-5"
    assert isinstance(classifier_settings, dict)
    assert isinstance(responder_settings, dict)
    assert classifier_settings["max_tokens"] == 600
    assert responder_settings["max_tokens"] == 1200


@pytest.mark.db
def test_an_unregistered_route_is_refused(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    with pytest.raises(UnapprovedRouteError, match="not registered"):
        build_turn_runtime(
            _deployment(clinic_pack, factory, app_url=app_url, model_route="openai-us")
        )


@pytest.mark.db
async def test_the_offline_runtime_answers_without_a_provider(
    activate_pack: Activate, app_url: str, clinic_pack: BusinessPack
) -> None:
    factory = activate_pack(clinic_pack)
    runtime = build_turn_runtime(_deployment(clinic_pack, factory, app_url=app_url))
    result, _ = await handle_message(
        "Quanto costa la fisioterapia?", ConversationState(turns=1), runtime
    )
    assert result.outcome is TurnOutcome.REPLIED
    assert result.text.startswith("Fisioterapia: 30 EUR (a seduta)")
