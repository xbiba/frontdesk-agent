"""Conversations for the email tests: example.com addresses only."""

from __future__ import annotations

from typing import Any

from frontdesk.email.mailbox import MailMessage, MailThread

MAILBOX = "info@example.com"


def message(body: str = "Buongiorno, quanto costa la fisioterapia?", **fields: Any) -> MailMessage:
    values: dict[str, Any] = {
        "id": "m1",
        "sender": "Anna Leka <anna.leka@example.com>",
        "subject": "Fisioterapia",
        "body": body,
        "message_id": "<m1@example.com>",
        **fields,
    }
    return MailMessage(**values)


def thread(*messages: MailMessage, thread_id: str = "t1") -> MailThread:
    return MailThread(id=thread_id, messages=messages or (message(),))


# --- a scripted model and a pack with email --------------------------------------------------------
from collections.abc import Callable  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from pydantic_ai import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.messages import ModelMessage  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402

from frontdesk.agent.prompts import read_prompt  # noqa: E402
from frontdesk.email.drafter import Drafter  # noqa: E402
from frontdesk.email.job import EmailPass  # noqa: E402
from frontdesk.email.ledger import MemoryLedger  # noqa: E402
from frontdesk.email.mailbox import FileMailbox  # noqa: E402
from frontdesk.tenancy.loader import load_pack  # noqa: E402
from frontdesk.tenancy.pack import BusinessPack, EmailChannel  # noqa: E402

NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
PHONE = "+355 00 000 0000"  # the demo clinic's
SENDER_SECRET = b"k" * 32


def reply(
    draft: str = "Gentile Anna, la fisioterapia c'è. [FIRMA]", **fields: Any
) -> dict[str, Any]:
    """A model's structured reply; no questions unless given."""
    return {
        "category": "request",
        "reason": "a question",
        "language": "it",
        "questions": [],
        "draft": draft,
        "missing_info": [],
        **fields,
    }


class Script:
    """A model that answers each request with the next reply, and keeps the requests it saw."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.seen: list[list[ModelMessage]] = []

    def model(self) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.seen.append(messages)
            answer = self.replies.pop(0)
            usage = RequestUsage(input_tokens=10_000, output_tokens=1_000)
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, answer)], usage=usage
            )

        return FunctionModel(respond)


def email_pack(businesses_dir: Path, **fields: Any) -> BusinessPack:
    pack = load_pack(businesses_dir / "demo-clinic")
    email = EmailChannel.model_validate({"mailbox": MAILBOX, "testers_only": False, **fields})
    return pack.model_copy(update={"email": email})


def make_pass(
    pack: BusinessPack,
    box: FileMailbox,
    script: Script | None,
    *,
    ledger: MemoryLedger | None = None,
    now: Callable[[], datetime] = lambda: NOW,
) -> EmailPass:
    assert pack.email is not None
    drafter = (
        None
        if script is None
        else Drafter(
            script.model(),
            pack,
            pack.email,
            prompt=read_prompt(pack.profile, "email"),
            today=now().date(),
        )
    )
    return EmailPass(
        mailbox=box,
        ledger=ledger or MemoryLedger(),
        drafter=drafter,
        pack=pack,
        now=now,
        dry_run=script is None,
        sender_secret=SENDER_SECRET,
    )
