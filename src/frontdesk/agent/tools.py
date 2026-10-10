"""The responder's one tool, built once per deployment over its services (ADR-0002).

Deps stay serializable; database sessions and embedders come from the closure,
so the same toolset works in-process now and as Temporal activities if Temporal stays
(ADR-0001, reconsidered in M4).
It reads approved documents only and returns typed evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from pydantic_ai import FunctionToolset, RunContext
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.agent.classification import check_free_text
from frontdesk.agent.deps import ResponderDeps, ToolResult
from frontdesk.db.engine import business_session
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.evidence import neutralized
from frontdesk.knowledge.retrieval import chunk_evidence, hybrid_search
from frontdesk.tenancy.pack import Profile


@dataclass(frozen=True)
class ToolServices:
    session_factory: sessionmaker[Session]
    embedder: Embedder
    today: Callable[[], date]
    min_similarity: float = 0.0  # the vector side's relevance floor (Settings)


def build_toolset(profile: Profile, services: ToolServices) -> FunctionToolset[ResponderDeps]:
    toolset: FunctionToolset[ResponderDeps] = FunctionToolset(
        id=f"{profile.value}-tools", strict=True
    )

    @toolset.tool
    def search_information(ctx: RunContext[ResponderDeps], question: str) -> ToolResult:
        """Search the business's approved documents.

        Args:
            question: A short practical question without names, numbers, contact details or health details.
        """
        reason = check_free_text(question)
        if reason is not None:
            return ToolResult(note=reason)
        vector = services.embedder.embed([question])[0]
        with business_session(services.session_factory, ctx.deps.business_id) as session:
            chunks = hybrid_search(
                session,
                business_id=ctx.deps.business_id,
                query=question,
                query_vector=vector,
                today=services.today(),
                min_similarity=services.min_similarity,
            )
        return ToolResult(evidence=[neutralized(chunk_evidence(chunk)) for chunk in chunks])

    return toolset
