"""The knowledge in the prompt (Albi, 2026-10-16): a pack whose approved documents are small gives
them whole to the reception agent, with no search tool and no embedder; a larger one is searched,
as before (frontdesk.conversation.reception).

The first clinic's documents are 12 texts in three languages, about 3,000 characters: the agent searched
them in one turn of seven, one request more each time. Up to IN_PROMPT_CHARS, the documents valid
today ride with every request, in the conversation's language, as a search returns them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

from sqlalchemy.orm import Session

from frontdesk.knowledge.evidence import neutralized
from frontdesk.knowledge.retrieval import chunk_evidence, valid_chunks
from frontdesk.knowledge.versions import active_version_id, version_chars
from frontdesk.tenancy.pack import Language

NONE_VALID: Final = "No document is valid today."  # never an empty section (final review, M3)


def knowledge_chars(session: Session, business_id: str) -> int:
    """The characters of the active version's documents, in every language and period."""
    version_id = active_version_id(session, business_id)
    return 0 if version_id is None else version_chars(session, version_id)


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
    """The active documents valid today, in `language` when it has some, else in every language.
    Each is fenced as data, its angle brackets turned to quotes (final review of #48, M5)."""
    chunks = valid_chunks(session, business_id=business_id, today=today)
    chosen = [chunk for chunk in chunks if chunk.language == language] or chunks
    blocks: dict[str, list[str]] = {}
    for chunk in chosen:
        item = neutralized(chunk_evidence(chunk))
        blocks.setdefault(chunk.document_key, []).append(f"[{item.title}]\n{item.content}")
    texts = tuple((key, "\n\n".join(parts)) for key, parts in blocks.items())
    if not texts:
        return PromptDocuments(NONE_VALID)
    fenced = "\n".join(f'<document key="{key}">\n{text}\n</document>' for key, text in texts)
    return PromptDocuments(fenced, texts)
