"""The knowledge in the prompt (Albi, 2026-10-16): a pack whose approved documents are small gives
them whole to the reception agent, with no search tool and no embedder; a larger one is searched,
as before (frontdesk.conversation.reception).

Clinica MCU's documents are 12 texts in three languages, about 3,000 characters: the agent searched
them in one turn of seven, one request more each time. Up to IN_PROMPT_CHARS, the documents valid
today ride with every request, in the conversation's language, as a search returns them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from frontdesk.db.models import Chunk
from frontdesk.knowledge.evidence import neutralized
from frontdesk.knowledge.retrieval import chunk_evidence, valid_chunks
from frontdesk.knowledge.versions import active_version_id
from frontdesk.tenancy.pack import Language

# The most characters of approved documents, every language counted, that ride in the prompt: about
# 3,500 tokens at 4 characters a token (Albi: "sotto 3-4 mila token").
IN_PROMPT_CHARS: Final = 14_000


def knowledge_chars(session: Session, business_id: str) -> int:
    """The characters of the active version's documents, in every language and period."""
    version_id = active_version_id(session, business_id)
    if version_id is None:
        return 0
    size = func.sum(func.length(Chunk.heading) + func.length(Chunk.content))
    total = session.scalar(
        select(func.coalesce(size, 0)).where(
            Chunk.business_id == business_id, Chunk.knowledge_version_id == version_id
        )
    )
    return int(total or 0)


@dataclass(frozen=True)
class PromptDocuments:
    text: str  # each passage as "[heading]\ncontent", as search_information returns it
    texts: tuple[tuple[str, str], ...] = ()  # each document's key and its passages

    @property
    def sources(self) -> tuple[str, ...]:
        """The keys of the documents it holds."""
        return tuple(key for key, _ in self.texts)


def documents_for(
    session: Session, business_id: str, today: date, language: Language
) -> PromptDocuments:
    """The active documents valid today, in `language` when it has some, else in every language."""
    chunks = valid_chunks(session, business_id=business_id, today=today)
    chosen = [chunk for chunk in chunks if chunk.language == language] or chunks
    blocks: dict[str, list[str]] = {}
    for chunk in chosen:
        item = neutralized(chunk_evidence(chunk))
        blocks.setdefault(chunk.document_key, []).append(f"[{item.title}]\n{item.content}")
    texts = tuple((key, "\n\n".join(parts)) for key, parts in blocks.items())
    return PromptDocuments("\n\n".join(text for _, text in texts), texts)
