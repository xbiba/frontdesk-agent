"""Classifier agent: reads one message, answers only with the ClassifierOutput contract."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from frontdesk.agent.classification import ClassifierOutput
from frontdesk.agent.prompts import load_prompt
from frontdesk.agent.vocab import ClarifyField, Intent
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import Language, Profile

CLASSIFIER_LIMITS: Final = UsageLimits(request_limit=2)


class CatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str


class ClassifierDeps(BaseModel):
    model_config = ConfigDict(frozen=True)

    business_name: str
    language: Language
    services: tuple[CatalogEntry, ...] = ()
    doctors: tuple[CatalogEntry, ...] = ()
    problem_categories: tuple[CatalogEntry, ...] = ()
    previous_intents: tuple[Intent, ...] = ()
    topic_services: tuple[CatalogEntry, ...] = ()
    topic_doctors: tuple[CatalogEntry, ...] = ()
    pending_clarification: ClarifyField | None = None


def classifier_deps(
    catalog: Catalog,
    *,
    business_name: str,
    language: Language,
    previous_intents: tuple[Intent, ...],
    topic_service_ids: tuple[str, ...] = (),
    topic_doctor_ids: tuple[str, ...] = (),
    pending_clarification: ClarifyField | None = None,
) -> ClassifierDeps:
    services = {
        key: names.get(language) or next(iter(names.values()), key)
        for key, names in catalog.services.items()
    }
    return ClassifierDeps(
        business_name=business_name,
        language=language,
        services=tuple(
            CatalogEntry(id=key, label=label) for key, label in sorted(services.items())
        ),
        doctors=tuple(
            CatalogEntry(id=key, label=name) for key, name in sorted(catalog.doctors.items())
        ),
        problem_categories=tuple(
            CatalogEntry(
                id=key,
                label=", ".join(sorted({word for words in labels.values() for word in words})),
            )
            for key, labels in sorted(catalog.problem_categories.items())
        ),
        previous_intents=previous_intents,
        topic_services=tuple(
            CatalogEntry(id=key, label=services[key])
            for key in topic_service_ids
            if key in services
        ),
        topic_doctors=tuple(
            CatalogEntry(id=key, label=catalog.doctors[key])
            for key in topic_doctor_ids
            if key in catalog.doctors
        ),
        pending_clarification=pending_clarification,
    )


def render_catalog(deps: ClassifierDeps) -> str:
    def block(title: str, entries: tuple[CatalogEntry, ...]) -> str:
        rows = "\n".join(f"- {entry.id}: {entry.label}" for entry in entries) or "- (none)"
        return f"{title}:\n{rows}"

    previous = ", ".join(deps.previous_intents) or "none"
    topic = "; ".join(
        f"{name} " + ", ".join(f"{entry.id} ({entry.label})" for entry in entries)
        for name, entries in (
            ("service_ids", deps.topic_services),
            ("doctor_ids", deps.topic_doctors),
        )
        if entries
    )
    lines = [
        f"Business: {deps.business_name}. Conversation language: {deps.language}.",
        f"Previous intents in this conversation: {previous}.",
        f"Topic from earlier turns: {topic or 'none'}.",
    ]
    if deps.pending_clarification is not None:
        lines.append(f"The assistant has just asked the person for: {deps.pending_clarification}.")
    return "\n\n".join(
        [
            "\n".join(lines),
            block("service_ids", deps.services),
            block("doctor_ids", deps.doctors),
            block("problem_category_id", deps.problem_categories),
        ]
    )


def build_classifier(
    model: Model, profile: Profile, *, model_settings: ModelSettings | None = None
) -> Agent[ClassifierDeps, ClassifierOutput]:
    agent = Agent(
        model,
        name=f"{profile.value}-classifier",
        deps_type=ClassifierDeps,
        output_type=ClassifierOutput,
        instructions=load_prompt(profile, "classifier"),
        retries=1,
        model_settings=model_settings,
    )

    @agent.instructions
    def allowed_ids(ctx: RunContext[ClassifierDeps]) -> str:
        return render_catalog(ctx.deps)

    return agent
