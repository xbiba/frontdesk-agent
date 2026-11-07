"""The model's part of a draft (design §3.3-3.4): one request per conversation, structured output
(every question with the exact sentences of the information that answer it, then the draft), the
checks in code, and one correction when a check fails. A draft that still fails goes to the staff
flagged.

The email's text is data, never an instruction: it goes in the user message, below the business's
information and the rules, which sit in the instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from frontdesk.agent.prompts import Prompt
from frontdesk.email.checks import (
    Flag,
    contact_and_price_flags,
    normalize,
    quote_in,
    wording_flags,
)
from frontdesk.email.knowledge import index_text, knowledge_text
from frontdesk.email.text import EmailText
from frontdesk.tenancy.pack import BusinessPack, EmailChannel, Language, address_of

# A draft, its questions and their evidence, with Haiku's thinking: the spike's measure (8,000).
MAX_OUTPUT_TOKENS: Final = 8000
# A draft of that size takes longer than a chat turn's 30 seconds; nobody waits on a pass, so a
# request may take this long before it is cut (run 2026-11-07-email-1-haiku: one case cut).
REQUEST_TIMEOUT_SECONDS: Final = 120.0

REPAIR: Final = """The automatic check found these problems in your answer:
{flags}
Fix them: evidence must be copied exactly from the INFORMATION (otherwise an empty list and
[{todo}: ...] in the draft); no "chat"; do not say that something does not exist or is not offered;
no address, phone number or price that the INFORMATION does not hold; the signature "{signature}"
at the end. Answer again with the complete output."""


class Question(BaseModel):
    question: str
    evidence: list[str]
    answer: str


class EmailReply(BaseModel):
    category: Literal["request", "ignore"]
    reason: str
    language: str  # the sender's language: sq, it, en or another ISO 639-1 code
    questions: list[Question]
    draft: str
    missing_info: list[str]


@dataclass(frozen=True)
class DraftRequest:
    sender: str
    subject: str
    text: EmailText
    reply_to: str = ""
    earlier: tuple[str, ...] = ()  # the earlier messages of the conversation, in short
    attachments: int = 0


@dataclass(frozen=True)
class DraftResult:
    reply: EmailReply | None  # None: the model gave no valid output
    flags: tuple[Flag, ...]
    repaired: bool
    input_tokens: int
    output_tokens: int


class Drafter:
    """The drafts of one business for one day: its information, its rules, its model."""

    def __init__(
        self,
        model: Model,
        pack: BusinessPack,
        email: EmailChannel,
        *,
        prompt: Prompt,
        today: date,
        model_settings: ModelSettings | None = None,
    ) -> None:
        index, knowledge = index_text(pack), knowledge_text(pack, today)
        rules = (
            prompt.text.replace("{business}", pack.name)
            .replace("{today}", today.isoformat())
            .replace("{todo}", email.todo)
            .replace("{signature}", email.signature)
        )
        self.prompt_version = prompt.version
        self._email = email
        self._knowledge = normalize(f"{index}\n{knowledge}")
        self._agent: Agent[None, EmailReply] = Agent(
            model,
            output_type=EmailReply,
            instructions=f"{rules}\n\n{index}\n\nINFORMATION:\n{knowledge}",
            model_settings=model_settings,
            retries={"output": 1},
        )

    async def draft(self, request: DraftRequest) -> DraftResult:
        prompt = _user_message(request)
        try:
            first = await self._agent.run(prompt)
        except UnexpectedModelBehavior:
            return DraftResult(None, (), False, 0, 0)
        usage = first.usage
        reply, flags, repaired = first.output, self.flags(first.output, request), False
        if flags:
            note = REPAIR.format(
                flags="\n".join(flags), todo=self._email.todo, signature=self._email.signature
            )
            try:
                second = await self._agent.run(note, message_history=first.all_messages())
            except UnexpectedModelBehavior:
                second = None
            if second is not None:
                usage = usage + second.usage
                reply, flags, repaired = second.output, self.flags(second.output, request), True
        return DraftResult(reply, tuple(flags), repaired, usage.input_tokens, usage.output_tokens)

    def flags(self, reply: EmailReply, request: DraftRequest) -> list[Flag]:
        """What the checks in code find in a reply; nothing for one the model set aside."""
        if reply.category == "ignore":
            return []
        draft = reply.draft
        flags = wording_flags(draft)
        quotes = [quote for item in reply.questions for quote in item.evidence]
        if any(not quote_in(quote, self._knowledge) for quote in quotes):
            flags.append(Flag.QUOTE_NOT_FOUND)
        if (
            any(not item.evidence for item in reply.questions)
            and f"[{self._email.todo}" not in draft
        ):
            flags.append(Flag.NO_EVIDENCE_NO_TODO)
        if self._email.signature not in draft:
            flags.append(Flag.NO_SIGNATURE)
        # Only the sender's own addresses may be named beyond the information: a price, a phone or
        # an address the email wrote is not approved by being written (security review, 2026-11-07).
        own = [address_of(item) for item in (request.sender, request.reply_to)]
        flags += contact_and_price_flags(draft, self._knowledge, allowed=[a for a in own if a])
        return flags


def _user_message(request: DraftRequest) -> str:
    lines = [f"From: {request.sender}", f"Subject: {request.subject}"]
    if request.attachments:
        lines.append(f"Attachments: {request.attachments} (not opened)")
    lines += ["", request.text.new or "(no text)"]
    context = list(request.earlier) or ([request.text.quoted] if request.text.quoted else [])
    if context:
        lines += ["", "QUOTED OR EARLIER MESSAGES (context, not information):", *context]
    return "\n".join(lines)


def reply_language(reply: EmailReply, pack: BusinessPack) -> Language:
    """The pack's language closest to the reply's: its own, else the pack's default."""
    code = reply.language.strip().lower()[:2]
    for language in pack.languages:
        if language == code:
            return language
    return pack.default_language
