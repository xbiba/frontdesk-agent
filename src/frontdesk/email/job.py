"""One pass of the email drafts ("il giro", design §3-§4), as Cloud Scheduler starts it every few
minutes: the new conversations of the mailbox, screened in code, one model request for each real
question, the checks, a draft in the conversation and a label. Nothing is ever sent.

Limits: 30 conversations a pass, 300 model requests a day, 3 drafts a day for one sender (beyond,
only the label "da verificare", no model), 6,000 characters of the sender's text (beyond, cut and
flagged). A dry run reads the mailbox and says what it would do, with no model, no draft, no
label and nothing written.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

from frontdesk.email.checks import Flag
from frontdesk.email.drafter import Drafter, DraftRequest, EmailReply, reply_language
from frontdesk.email.ledger import MIN_SECRET_BYTES, Handled, Ledger, sender_key
from frontdesk.email.mailbox import Draft, Mailbox, MailThread, reply_draft
from frontdesk.email.screening import Screen, screen
from frontdesk.email.text import shorten, split_quotes
from frontdesk.model_gateway.costs import Price
from frontdesk.observability.logs import log_event, log_warning
from frontdesk.safety.rules import clinical_rule, emergency_rule
from frontdesk.tenancy.pack import BusinessPack, EmailChannel, FixedTextKey, address_of

QUERY: Final = "in:inbox newer_than:3d"
THREADS_PER_PASS: Final = 30
MODEL_REQUESTS_PER_DAY: Final = 300
DRAFTS_PER_SENDER_PER_DAY: Final = 3
KEEP_DAYS: Final = 30
EARLIER_MESSAGES: Final = 2  # the earlier messages the model reads, in short
EARLIER_CHARS: Final = 500
LABEL_DRAFT: Final = "Bozza AI"
LABEL_REVIEW: Final = "Bozza AI \u2013 da verificare"  # an en dash, as the design names it
LABEL_IGNORED: Final = "AI ignorata"
REVIEW_NOTE: Final = "DA VERIFICARE"
logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    DRAFTED = "drafted"  # a draft that passed the checks
    REVIEW = "review"  # a draft a check flagged, or a sender past the limit (no draft)
    IGNORED = "ignored"  # screened in code: a list, an automatic message, a no-reply sender
    IGNORED_BY_MODEL = "ignored_by_model"  # the model saw no question
    NO_OUTPUT = "no_output"  # the model gave no valid output: flagged, no draft
    STAFF = "staff"  # the staff answered, a draft, or spam: nothing to do
    NOT_TESTER = "not_tester"  # tester mode, and the sender is not a tester
    SENDER_LIMIT = "sender_limit"
    WOULD_DRAFT = "would_draft"  # a dry run
    SEEN = "seen"  # its last message is handled already
    FAILED = "failed"  # the mailbox or the model failed: tried again at the next pass


# What the line at the top of a flagged draft says, for the staff (in Italian, as the labels).
_NOTES: Final = {
    Flag.QUOTE_NOT_FOUND: "una frase citata non è nelle informazioni",
    Flag.CHAT_WORDING: "parla di chat",
    Flag.ABSENCE_CLAIM: "dice che qualcosa non c'è",
    Flag.NO_EVIDENCE_NO_TODO: "una domanda senza risposta non ha il segnaposto",
    Flag.NO_SIGNATURE: "manca la firma",
    Flag.UNKNOWN_CONTACT: "un contatto non è nelle informazioni",
    Flag.UNKNOWN_PRICE: "un prezzo non è nelle informazioni",
    "emergency": "possibile urgenza: in cima il testo di emergenza",
    "clinical_advice": "domanda medica: in cima il testo fisso",
    "truncated": "email lunga: letta solo in parte",
    "attachments": "ha allegati, non aperti",
    "sender_limit": "troppe bozze oggi per questo mittente: nessuna bozza",
    "no_output": "il modello non ha dato una bozza valida",
}


@dataclass(frozen=True)
class ThreadResult:
    thread_id: str
    outcome: Outcome
    flags: tuple[str, ...] = ()
    reason: str = ""
    draft: Draft | None = None
    reply: EmailReply | None = None
    usd: float = 0.0


@dataclass(frozen=True)
class PassResult:
    threads: tuple[ThreadResult, ...]
    stopped: str | None = None  # why the pass stopped before the end
    purged: int = 0

    def count(self, outcome: Outcome) -> int:
        return sum(1 for item in self.threads if item.outcome is outcome)


class PassStoppedError(RuntimeError):
    """The pass cannot go on: the daily limit, the spend cap, the meter."""


@dataclass
class EmailPass:
    mailbox: Mailbox
    ledger: Ledger
    drafter: Drafter | None  # None: a dry run
    pack: BusinessPack
    now: Callable[[], datetime]
    price: Price | None = None
    stops: tuple[type[BaseException], ...] = ()  # model errors that end the pass
    dry_run: bool = False
    # The secret the sender's hash is keyed with (frontdesk-email-sender-key).
    sender_secret: bytes = b""
    _email: EmailChannel = field(init=False)

    def __post_init__(self) -> None:
        if self.pack.email is None:
            raise ValueError(f"the pack of {self.pack.id} has no email section (email:)")
        if self.drafter is None and not self.dry_run:
            raise ValueError("only a dry run goes without a model")
        if len(self.sender_secret) < MIN_SECRET_BYTES:
            raise ValueError(
                f"the sender key needs a secret of at least {MIN_SECRET_BYTES} bytes "
                "(frontdesk-email-sender-key)"
            )
        self._email = self.pack.email

    def today(self) -> date:
        return self.now().astimezone(ZoneInfo(self.pack.timezone)).date()

    async def run(self, thread_ids: Sequence[str] | None = None) -> PassResult:
        ids = (
            list(thread_ids)
            if thread_ids is not None
            else self.mailbox.thread_ids(QUERY, THREADS_PER_PASS)
        )
        results: list[ThreadResult] = []
        stopped = None
        for thread_id in ids[:THREADS_PER_PASS]:
            try:
                result = await self.handle(thread_id)
            except PassStoppedError as error:
                stopped = str(error)
                log_warning(logger, "email.pass_stopped", reason=stopped)
                break
            except Exception as error:  # a draft the mailbox refused: tried again next pass
                log_warning(
                    logger, "email.write_failed", thread_id=thread_id, error=type(error).__name__
                )
                result = ThreadResult(thread_id, Outcome.FAILED, reason=type(error).__name__)
            log_event(
                logger,
                "email.thread",
                thread_id=thread_id,
                outcome=result.outcome.value,
                flags=",".join(result.flags),
            )
            results.append(result)
        purged = 0 if self.dry_run else self.ledger.purge(self.today() - timedelta(days=KEEP_DAYS))
        return PassResult(tuple(results), stopped, purged)

    async def handle(self, thread_id: str) -> ThreadResult:
        try:
            thread = self.mailbox.thread(thread_id)
        except Exception as error:  # a mailbox that fails here is read again at the next pass
            log_warning(
                logger, "email.read_failed", thread_id=thread_id, error=type(error).__name__
            )
            return ThreadResult(thread_id, Outcome.FAILED, reason=type(error).__name__)
        if not thread.messages:
            return ThreadResult(thread_id, Outcome.SEEN)
        if self.ledger.last_message(thread.id) == thread.last.id:
            return ThreadResult(thread_id, Outcome.SEEN)
        return await self._decide(thread)

    async def _decide(self, thread: MailThread) -> ThreadResult:
        last = thread.last
        screened = screen(thread, mailbox=self._email.mailbox)
        if screened in {Screen.STAFF_ANSWERED, Screen.SPAM}:
            return self._settle(thread, Outcome.STAFF, reason=str(screened))
        if screened is not None:
            return self._settle(thread, Outcome.IGNORED, label=LABEL_IGNORED, reason=str(screened))
        if not self._email.drafts_for(last.sender):
            return self._settle(thread, Outcome.NOT_TESTER)
        key = sender_key(self.sender_secret, last.sender)
        today = self.today()
        if self.ledger.drafts_for(key, today) >= DRAFTS_PER_SENDER_PER_DAY:
            return self._settle(
                thread, Outcome.SENDER_LIMIT, label=LABEL_REVIEW, flags=("sender_limit",)
            )
        if self.ledger.model_requests_on(today) >= MODEL_REQUESTS_PER_DAY:
            raise PassStoppedError(f"{MODEL_REQUESTS_PER_DAY} model requests today")
        if self.drafter is None:
            return ThreadResult(thread.id, Outcome.WOULD_DRAFT)
        return await self._draft(thread, self.drafter)

    async def _draft(self, thread: MailThread, drafter: Drafter) -> ThreadResult:
        last = thread.last
        text = split_quotes(last.body)
        request = DraftRequest(
            sender=last.sender,
            subject=last.subject,
            text=text,
            reply_to=last.reply_to,
            earlier=self._earlier(thread),
            attachments=last.attachments,
        )
        try:
            drafted = await drafter.draft(request)
        except self.stops as error:
            raise PassStoppedError(f"{type(error).__name__}: {error}") from error
        except Exception as error:  # the model failed: no draft now, tried again at the next pass
            log_warning(
                logger, "email.model_failed", thread_id=thread.id, error=type(error).__name__
            )
            return ThreadResult(thread.id, Outcome.FAILED, reason=type(error).__name__)
        usd = self._cost(drafted.input_tokens, drafted.output_tokens)
        reply = drafted.reply
        if reply is None:
            return self._settle(
                thread, Outcome.NO_OUTPUT, label=LABEL_REVIEW, flags=("no_output",), usd=usd
            )
        if reply.category == "ignore":
            return self._settle(
                thread, Outcome.IGNORED_BY_MODEL, label=LABEL_IGNORED, reply=reply, usd=usd
            )
        flags: list[str] = list(drafted.flags)
        # The rules read the subject and the whole body, quotes and what is past the cut included:
        # an emergency told in the subject or below a quote is still one (security review).
        whole = f"{last.subject}\n{last.body}"
        alert = emergency_rule(whole) or clinical_rule(whole)
        if alert:
            flags.append(alert)
        if text.truncated:
            flags.append("truncated")
        if last.attachments:
            flags.append("attachments")
        body = self._body(reply, flags, alert)
        draft = reply_draft(thread, mailbox=self._email.mailbox, body=body)
        outcome = Outcome.REVIEW if flags else Outcome.DRAFTED
        label = LABEL_REVIEW if flags else LABEL_DRAFT
        return self._settle(
            thread, outcome, label=label, flags=tuple(flags), draft=draft, reply=reply, usd=usd
        )

    def _body(self, reply: EmailReply, flags: Sequence[str], alert: str | None) -> str:
        parts = []
        if flags:
            parts.append(f"[{REVIEW_NOTE}: " + "; ".join(_NOTES.get(f, f) for f in flags) + "]")
        if alert:
            language = reply_language(reply, self.pack)
            key = FixedTextKey.EMERGENCY if alert == "emergency" else FixedTextKey.CLINICAL
            parts.append(self.pack.fixed_texts[language][key])
        parts.append(reply.draft.strip())
        return "\n\n".join(parts)

    def _earlier(self, thread: MailThread) -> tuple[str, ...]:
        earlier = thread.messages[:-1][-EARLIER_MESSAGES:]
        mailbox = self._email.mailbox
        return tuple(
            f"{'The staff' if address_of(item.sender) == mailbox else 'The sender'} wrote: "
            + shorten(split_quotes(item.body).new, EARLIER_CHARS)
            for item in earlier
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.price is None:
            return 0.0
        return self.price.cost(input_units=input_tokens, output_units=output_tokens)

    def _settle(
        self,
        thread: MailThread,
        outcome: Outcome,
        *,
        label: str | None = None,
        flags: tuple[str, ...] = (),
        reason: str = "",
        draft: Draft | None = None,
        reply: EmailReply | None = None,
        usd: float = 0.0,
    ) -> ThreadResult:
        """Create the draft, put the label, record the message; in a dry run, none of it."""
        result = ThreadResult(thread.id, outcome, flags, reason, draft, reply, usd)
        if self.dry_run:
            return result
        if draft is not None:
            self.mailbox.create_draft(draft)
        last = thread.last
        # Recorded once the draft exists, before the label: a label that fails costs no second
        # draft at the next pass.
        self.ledger.record(
            Handled(
                thread_id=thread.id,
                message_id=last.id,
                outcome=outcome.value,
                sender_key=sender_key(self.sender_secret, last.sender),
                day=self.today(),
                at=self.now(),
                usd=usd,
            )
        )
        if label is not None:
            try:
                self.mailbox.add_labels(thread.id, [label])
            except Exception as error:
                log_warning(
                    logger, "email.label_failed", thread_id=thread.id, error=type(error).__name__
                )
        return result
