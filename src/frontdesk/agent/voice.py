"""The voice (Albi's live chat, 2026-10-12: "it reads like a stupid chatbot"): the model rewrites a
reply that code wrote, so that it answers this message in this conversation.

The model chooses the words; code keeps the facts. The rewrite is checked
(frontdesk.agent.voice_check) and the code's text is sent when it fails. Emergency, clinical and
personal-data texts, the disclosure, the answers from documents and the booking steps never reach
the voice.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection
from typing import Final

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from frontdesk.agent.answer_check import to_plain_text
from frontdesk.agent.prompts import instructions_for
from frontdesk.agent.voice_check import voice_problems
from frontdesk.observability.logs import log_event
from frontdesk.tenancy.pack import Language, Profile

logger = logging.getLogger(__name__)
VOICE_LIMITS: Final = UsageLimits(request_limit=1)
VOICE_TIMEOUT_SECONDS: Final = 10.0


class VoiceDeps(BaseModel):
    model_config = ConfigDict(frozen=True)

    business_name: str
    language: Language
    style: str  # one of the business's own texts, in its register ("Lei" or "tu")
    draft: str
    previous_reply: str | None = None


def render_context(deps: VoiceDeps) -> str:
    lines = [
        f"Business: {deps.business_name}. Reply language: {deps.language}.",
        f"Style sample (the business's register):\n<<<\n{deps.style}\n>>>",
    ]
    if deps.previous_reply:
        lines.append(f"The assistant's previous message:\n<<<\n{deps.previous_reply}\n>>>")
    lines.append(f"The draft:\n<<<\n{deps.draft}\n>>>")
    return "\n\n".join(lines)


def build_voice(
    model: Model,
    profile: Profile,
    *,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
) -> Agent[VoiceDeps, str]:
    """`instructions` is the prompt text a caller has already read (`read_prompt`)."""
    if not isinstance(model, Model):
        raise TypeError(
            f"the voice needs a Model object, not {type(model).__name__}: build live models in "
            "the model gateway, because a model-id string would skip its EU pin"
        )
    agent = Agent(
        model,
        name=f"{profile.value}-voice",
        deps_type=VoiceDeps,
        output_type=str,
        instructions=instructions_for(profile, "voice", instructions),
        retries=0,
        model_settings=model_settings,
    )

    @agent.instructions
    def context(ctx: RunContext[VoiceDeps]) -> str:
        return render_context(ctx.deps)

    return agent


async def rewrite(
    voice: Agent[VoiceDeps, str],
    message: str,
    deps: VoiceDeps,
    *,
    names: Collection[str],
    facts: bool,
) -> tuple[str | None, str]:
    """The voice's rewrite of the draft, or None, and what happened: "rewritten", "failed", or the
    problems that kept the draft (frontdesk.agent.voice_check). Only the error's type is logged."""
    try:
        async with asyncio.timeout(VOICE_TIMEOUT_SECONDS):
            run = await voice.run(message, deps=deps, usage_limits=VOICE_LIMITS)
    except Exception as error:  # the draft is a complete reply: any failure keeps it
        log_event(logger, "voice.failed", reason=type(error).__name__)
        return None, "failed"
    problems = voice_problems(
        deps.draft, run.output, language=deps.language, names=names, facts=facts
    )
    if problems:
        return None, ",".join(problems)
    return to_plain_text(run.output), "rewritten"
