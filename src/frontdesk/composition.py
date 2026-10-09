"""Composition root: one deployment, one business (ADR-0008).

Integrations are wired only when the business pack enables them: a clinic never
builds the order store, and nothing here initialises Calendar. A deployment
refuses to start when its active knowledge was embedded by another model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classifier import build_classifier
from frontdesk.agent.demo_models import demo_classifier_model, demo_responder_model
from frontdesk.agent.prompts import read_prompt
from frontdesk.agent.responder import build_responder
from frontdesk.agent.tools import ToolServices, build_toolset
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.db.engine import business_session, make_engine, make_session_factory
from frontdesk.knowledge.embedders import Embedder, build_embedder
from frontdesk.knowledge.facts import load_catalog
from frontdesk.knowledge.versions import check_embedder
from frontdesk.model_gateway.factory import build_live_model, live_model_settings
from frontdesk.model_gateway.routes import DataClass, require_route
from frontdesk.settings import Settings
from frontdesk.shop.orders import OrderStore
from frontdesk.tenancy.loader import load_pack
from frontdesk.tenancy.pack import BusinessPack


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Deployment:
    """What one process serves: a business, its database and its embedder.

    `now` is the deployment's clock (an aware time); tests replace it with a fixed one.
    """

    settings: Settings
    pack: BusinessPack
    session_factory: sessionmaker[Session]
    embedder: Embedder
    now: Callable[[], datetime] = _utc_now

    def today(self) -> date:
        """The calendar date in the business's own time zone, not the server's.

        Prices and documents are valid for business days: at 00:30 in Tirane it is already the
        next day there, whichever date the server shows.
        """
        return self.now().astimezone(ZoneInfo(self.pack.timezone)).date()


class NoActiveKnowledgeError(RuntimeError):
    """The business has no approved, active knowledge version yet."""

    def __init__(self, business_id: str) -> None:
        super().__init__(f"business '{business_id}' has no active knowledge version")


def open_deployment(settings: Settings) -> Deployment:
    pack = load_pack(settings.require_business_pack())
    factory = make_session_factory(make_engine(settings.database_url))
    return Deployment(settings, pack, factory, build_embedder(settings.embedding_backend))


def build_turn_runtime(
    deployment: Deployment, *, today: Callable[[], date] | None = None
) -> TurnRuntime:
    """The pipeline for this deployment's business.

    `today` defaults to the business-local date (`Deployment.today`); a caller may fix it. Each
    prompt is read once, here: the agents run that text and the runtime records its version, so a
    turn never reads a prompt file. A missing or blank prompt stops the build.
    """
    pack = deployment.pack
    business_today = today or deployment.today
    classifier_prompt = read_prompt(pack.profile, "classifier")
    responder_prompt = read_prompt(pack.profile, "responder")
    with business_session(deployment.session_factory, pack.id) as session:
        catalog = load_catalog(session, pack.id)
        check_embedder(session, pack.id, deployment.embedder)
    if catalog is None:
        raise NoActiveKnowledgeError(pack.id)
    route = require_route(deployment.settings.model_route, DataClass.PATIENT_DERIVED)
    services = ToolServices(
        session_factory=deployment.session_factory,
        embedder=deployment.embedder,
        today=business_today,
    )
    toolset = build_toolset(pack.profile, services)
    if route.provider == "offline":
        classifier = build_classifier(
            demo_classifier_model(catalog), pack.profile, instructions=classifier_prompt.text
        )
        responder = build_responder(
            demo_responder_model(), pack.profile, toolset, instructions=responder_prompt.text
        )
    else:
        model = build_live_model(route, gcp_project=deployment.settings.gcp_project)
        classifier = build_classifier(
            model,
            pack.profile,
            model_settings=live_model_settings(max_tokens=600),
            instructions=classifier_prompt.text,
        )
        responder = build_responder(
            model,
            pack.profile,
            toolset,
            model_settings=live_model_settings(max_tokens=1200),
            instructions=responder_prompt.text,
        )
    return TurnRuntime(
        pack=pack,
        classifier=classifier,
        responder=responder,
        classifier_prompt_version=classifier_prompt.version,
        responder_prompt_version=responder_prompt.version,
        session_factory=deployment.session_factory,
        embedder=deployment.embedder,
        today=business_today,
        min_support_similarity=deployment.settings.min_support_similarity,
        capture_content=deployment.settings.telemetry_capture_content,
        orders=OrderStore(tuple(pack.orders)) if pack.features.order_status else None,
        now=deployment.now,
    )
