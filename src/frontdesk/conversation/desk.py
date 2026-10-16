"""What the reception agent works with during one turn (frontdesk.conversation.reception): the
clinic, its calendar, the patient's message, the facts, and what the tools did. The tools update
`booking`, `seen`, `sources`, `booked` and the summary; the turn reads them afterwards."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session, sessionmaker

from frontdesk.booking.scheduler import Scheduler
from frontdesk.conversation.booking_turn import BookingContext
from frontdesk.conversation.state import BookingStep
from frontdesk.knowledge.embedders import Embedder
from frontdesk.knowledge.facts import Catalog
from frontdesk.tenancy.pack import BusinessPack, Language


@dataclass
class Desk:
    pack: BusinessPack
    catalog: Catalog
    session_factory: sessionmaker[Session]
    scheduler: Scheduler | None
    embedder: Embedder
    min_similarity: float
    conversation_id: str
    language: Language
    now: datetime
    message: str
    first_turn: bool
    turn: int  # the turns before this one: a summary shown now may be confirmed at turn + 1
    facts: str
    conversation: str  # the conversation so far, as text: a number in it may be repeated
    booking: BookingStep | None
    documents: str = ""  # the clinic's documents, when they ride in the prompt
    # Each document's key and text: the sources of a reply (frontdesk.conversation.reception).
    document_texts: tuple[tuple[str, str], ...] = ()
    previous: str = ""  # the agent's last message, which a reply may not repeat word for word
    seen: list[str] = field(default_factory=list)  # what the tools returned this turn
    sources: list[str] = field(default_factory=list)
    booked: bool = False
    released: bool = False  # cancel_booking let go of a held time
    summary: str | None = None  # the summary prepare_booking wrote this turn, shown as written
    booked_text: str | None = None  # what confirm_booking said when it booked
    rejected: list[str] = field(default_factory=list)  # the codes of the replies sent back

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.pack.timezone)

    @property
    def today(self) -> date:
        return self.now.astimezone(self.zone).date()

    def context(self) -> BookingContext:
        if self.scheduler is None:
            raise ValueError("a booking needs the runtime's calendar")
        return BookingContext(
            pack=self.pack,
            catalog=self.catalog,
            scheduler=self.scheduler,
            session_factory=self.session_factory,
            conversation_id=self.conversation_id,
            language=self.language,
            now=self.now,
        )

    def said(self, text: str) -> str:
        self.seen.append(text)
        return text
