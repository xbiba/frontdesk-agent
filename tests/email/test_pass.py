"""One pass of the email drafts on a mailbox in memory, with a scripted model: screening, limits,
checks, the correction, the labels, the ledger. Nothing is ever sent."""

from datetime import timedelta
from pathlib import Path

import pytest

from frontdesk.email.job import (
    LABEL_DRAFT,
    LABEL_IGNORED,
    LABEL_REVIEW,
    EmailPass,
    Outcome,
    PassStoppedError,
)
from frontdesk.email.ledger import Handled, MemoryLedger, sender_key
from frontdesk.email.mailbox import FileMailbox
from frontdesk.email.text import MAX_CHARS
from tests.email.helpers import (
    MAILBOX,
    NOW,
    PHONE,
    Script,
    email_pack,
    make_pass,
    message,
    reply,
    thread,
)


@pytest.fixture
def pack(businesses_dir: Path):  # type: ignore[no-untyped-def]
    return email_pack(businesses_dir)


async def test_a_question_gets_a_draft_in_its_conversation(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread()])
    script = Script(reply(f"Gentile Anna, la reception risponde al {PHONE}. [FIRMA]"))
    ledger = MemoryLedger()
    result = await make_pass(pack, box, script, ledger=ledger).run()
    (only,) = result.threads
    assert only.outcome is Outcome.DRAFTED, only.flags
    assert box.labels == {"t1": [LABEL_DRAFT]}
    (draft,) = box.drafts
    assert draft.in_reply_to == "<m1@example.com>"
    assert draft.sender == MAILBOX
    assert draft.body.startswith("Gentile Anna")
    assert ledger.last_message("t1") == "m1"
    prompt = str(script.seen[0])
    assert "quanto costa la fisioterapia" in prompt
    assert pack.location.phone in prompt  # the information rides in the instructions


async def test_a_handled_message_is_not_drafted_again_but_a_new_one_is(pack) -> None:  # type: ignore[no-untyped-def]
    ledger = MemoryLedger()
    first = FileMailbox([thread()])
    await make_pass(pack, first, Script(reply()), ledger=ledger).run()
    again = await make_pass(pack, first, Script(), ledger=ledger).run()
    assert again.threads[0].outcome is Outcome.SEEN
    later = FileMailbox(
        [thread(message(), message(id="m2", sender=MAILBOX), message(id="m3", body="E il prezzo?"))]
    )
    result = await make_pass(
        pack, later, Script(reply()), ledger=ledger, now=lambda: NOW + timedelta(hours=1)
    ).run()
    assert result.threads[0].outcome is Outcome.DRAFTED
    assert ledger.last_message("t1") == "m3"


@pytest.mark.parametrize(
    ("fields", "outcome", "labels"),
    [
        (
            {"headers": {"list-unsubscribe": "<mailto:x@example.com>"}},
            Outcome.IGNORED,
            [LABEL_IGNORED],
        ),
        ({"sender": "noreply@example.com"}, Outcome.IGNORED, [LABEL_IGNORED]),
        ({"sender": MAILBOX}, Outcome.STAFF, None),
        ({"labels": frozenset({"SPAM"})}, Outcome.STAFF, None),
    ],
)
async def test_screened_messages_reach_no_model(pack, fields, outcome, labels) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread(message(**fields))])
    script = Script()
    result = await make_pass(pack, box, script).run()
    assert result.threads[0].outcome is outcome
    assert box.labels.get("t1") == labels
    assert box.drafts == []
    assert script.seen == []


async def test_the_model_sets_aside_what_asks_nothing(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread()])
    result = await make_pass(pack, box, Script(reply("", category="ignore"))).run()
    assert result.threads[0].outcome is Outcome.IGNORED_BY_MODEL
    assert box.labels == {"t1": [LABEL_IGNORED]}
    assert box.drafts == []


async def test_a_failed_check_gets_one_correction(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread()])
    script = Script(reply("Ci scriva qui in chat. [FIRMA]"), reply())
    result = await make_pass(pack, box, script).run()
    assert result.threads[0].outcome is Outcome.DRAFTED
    assert len(script.seen) == 2
    assert "chat_wording" in str(script.seen[1])


async def test_a_draft_that_fails_twice_goes_to_the_staff_flagged(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread()])
    invented = reply(
        "Costa 80 EUR. Scriva a altro@example.com. [FIRMA]",
        questions=[{"question": "Prezzo?", "evidence": ["Costa 80 EUR"], "answer": "80"}],
    )
    result = await make_pass(pack, box, Script(invented, invented)).run()
    found = result.threads[0]
    assert found.outcome is Outcome.REVIEW
    assert set(found.flags) == {"quote_not_found", "unknown_contact", "unknown_price"}
    assert box.labels == {"t1": [LABEL_REVIEW]}
    assert box.drafts[0].body.startswith("[DA VERIFICARE: ")


async def test_a_question_without_evidence_needs_the_placeholder(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread()])
    bare = reply(questions=[{"question": "Parcheggio?", "evidence": [], "answer": "?"}])
    result = await make_pass(pack, box, Script(bare, bare)).run()
    assert "no_evidence_no_todo" in result.threads[0].flags


async def test_a_medical_question_starts_with_the_packs_text(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread(message("Che farmaco mi consigliate per il ginocchio?"))])
    result = await make_pass(pack, box, Script(reply())).run()
    found = result.threads[0]
    assert found.outcome is Outcome.REVIEW
    assert "clinical_advice" in found.flags
    assert pack.fixed_texts["it"]["clinical"] in box.drafts[0].body


async def test_an_emergency_starts_with_the_emergency_text(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread(message("Ho un forte dolore al petto e non riesco a respirare"))])
    result = await make_pass(pack, box, Script(reply(language="en"))).run()
    assert "emergency" in result.threads[0].flags
    assert pack.fixed_texts["en"]["emergency"] in box.drafts[0].body


async def test_long_mail_and_attachments_are_flagged(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread(message("parola " * MAX_CHARS, attachments=2))])
    result = await make_pass(pack, box, Script(reply())).run()
    assert {"truncated", "attachments"} <= set(result.threads[0].flags)


async def test_tester_mode_drafts_only_for_testers(businesses_dir: Path) -> None:
    pack = email_pack(businesses_dir, testers_only=True, testers=["tester@example.com"])
    box = FileMailbox(
        [thread(), thread(message(sender="Tester <tester@example.com>"), thread_id="t2")]
    )
    result = await make_pass(pack, box, Script(reply())).run()
    assert [item.outcome for item in result.threads] == [Outcome.NOT_TESTER, Outcome.DRAFTED]
    assert "t1" not in box.labels


async def test_a_sender_past_three_drafts_a_day_gets_only_the_label(pack) -> None:  # type: ignore[no-untyped-def]
    ledger = MemoryLedger()
    key = sender_key(pack.id, "anna.leka@example.com")
    for index in range(3):
        ledger.record(Handled(f"old{index}", "m", "drafted", key, NOW.date(), NOW))
    box = FileMailbox([thread()])
    script = Script()
    result = await make_pass(pack, box, script, ledger=ledger).run()
    assert result.threads[0].outcome is Outcome.SENDER_LIMIT
    assert box.labels == {"t1": [LABEL_REVIEW]}
    assert script.seen == []


async def test_the_daily_limit_stops_the_pass(pack, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("frontdesk.email.job.MODEL_REQUESTS_PER_DAY", 1)
    ledger = MemoryLedger()
    ledger.record(Handled("old", "m", "drafted", "someone", NOW.date(), NOW))
    result = await make_pass(pack, FileMailbox([thread()]), Script(), ledger=ledger).run()
    assert result.stopped is not None
    assert result.threads == ()


async def test_a_dry_run_reaches_nothing(pack) -> None:  # type: ignore[no-untyped-def]
    box = FileMailbox([thread(), thread(message(sender="noreply@example.com"), thread_id="t2")])
    ledger = MemoryLedger()
    result = await make_pass(pack, box, None, ledger=ledger).run()
    assert [item.outcome for item in result.threads] == [Outcome.WOULD_DRAFT, Outcome.IGNORED]
    assert box.drafts == []
    assert box.labels == {}
    assert ledger.rows == []


async def test_old_rows_are_purged(pack) -> None:  # type: ignore[no-untyped-def]
    ledger = MemoryLedger()
    old = NOW - timedelta(days=31)
    ledger.record(Handled("old", "m", "drafted", "k", old.date(), old))
    result = await make_pass(pack, FileMailbox([]), Script(), ledger=ledger).run()
    assert result.purged == 1
    assert ledger.rows == []


async def test_a_model_failure_leaves_the_thread_for_the_next_pass(pack) -> None:  # type: ignore[no-untyped-def]
    class Broken(Script):
        def model(self):  # type: ignore[no-untyped-def]
            from pydantic_ai.models.function import FunctionModel

            def fail(messages, info):  # type: ignore[no-untyped-def]
                raise ConnectionError("down")

            return FunctionModel(fail)

    ledger = MemoryLedger()
    box = FileMailbox([thread()])
    result = await make_pass(pack, box, Broken(), ledger=ledger).run()
    assert result.threads[0].outcome is Outcome.FAILED
    assert ledger.rows == []
    assert box.drafts == []


async def test_a_stop_error_ends_the_pass(pack) -> None:  # type: ignore[no-untyped-def]
    class Broken(Script):
        def model(self):  # type: ignore[no-untyped-def]
            from pydantic_ai.models.function import FunctionModel

            def fail(messages, info):  # type: ignore[no-untyped-def]
                raise PermissionError("cap")

            return FunctionModel(fail)

    email_pass = make_pass(pack, FileMailbox([thread(), thread(thread_id="t2")]), Broken())
    stopping = EmailPass(
        mailbox=email_pass.mailbox,
        ledger=email_pass.ledger,
        drafter=email_pass.drafter,
        pack=pack,
        now=email_pass.now,
        stops=(PermissionError,),
    )
    result = await stopping.run()
    assert result.stopped is not None
    assert result.threads == ()
    with pytest.raises(PassStoppedError):
        await stopping.handle("t1")


def test_a_pack_without_email_is_refused(businesses_dir: Path) -> None:
    from frontdesk.tenancy.loader import load_pack

    with pytest.raises(ValueError, match="no email section"):
        EmailPass(
            mailbox=FileMailbox([]),
            ledger=MemoryLedger(),
            drafter=None,
            pack=load_pack(businesses_dir / "demo-clinic"),
            now=lambda: NOW,
            dry_run=True,
        )
