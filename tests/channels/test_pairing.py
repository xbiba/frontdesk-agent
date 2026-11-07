"""Guessing the pairing code on a public channel (the site's chat, security review of 2026-11-07):
a few misses lock a session out, many lock the process out, for an hour; the alert says only
which lock, never the session or the text."""

import logging
import uuid

import pytest
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.channels.inbox import Inbox
from frontdesk.channels.messages import Inbound, InboundKind
from frontdesk.channels.pairing import PairingGuard
from frontdesk.db.engine import make_engine, make_session_factory
from tests.channels.fakes import Clock

CODE = "web-pairing-code-with-enough-entropy"


class Ticks:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_session_is_locked_after_its_misses_and_freed_after_the_window() -> None:
    ticks = Ticks()
    guard = PairingGuard(per_user=3, per_process=100, window_seconds=3600, clock=ticks)
    for _ in range(3):
        assert guard.locked("s1") is False
        guard.miss("s1")
    assert guard.locked("s1") is True
    assert guard.locked("s2") is False
    ticks.now += 3601
    assert guard.locked("s1") is False


def test_the_process_is_locked_after_many_misses(caplog: pytest.LogCaptureFixture) -> None:
    guard = PairingGuard(per_user=100, per_process=4, window_seconds=3600, clock=Ticks())
    with caplog.at_level(logging.INFO, logger="frontdesk.channels.pairing"):
        for n in range(4):
            guard.miss(f"s{n}")
    assert guard.locked("fresh") is True
    alerts = [r for r in caplog.records if r.msg == "channel.pairing_locked"]
    assert [r.levelno for r in alerts] == [logging.ERROR]
    assert alerts[0].fields == {"scope": "process"}  # type: ignore[attr-defined]


@pytest.fixture
def factory(app_url: str) -> sessionmaker[Session]:
    return make_session_factory(make_engine(app_url))


@pytest.mark.db
def test_a_locked_session_cannot_pair_even_with_the_code(
    factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    guard = PairingGuard(per_user=3, per_process=100, window_seconds=3600, clock=Ticks())
    inbox = Inbox(
        factory,
        business_id="demo-clinic",
        channel="web",
        account_id="demo-clinic",
        testers_only=True,
        pairing_code=CODE,
        now=Clock(),
        pairing_guard=guard,
    )
    guesser, honest = "guesser-session-0001", "honest-session-00001"

    def say(user: str, text: str) -> tuple[str, ...]:
        event = Inbound("demo-clinic", user, InboundKind.TEXT, uuid.uuid4().hex, text)
        return inbox.accept([event]).work

    with caplog.at_level(logging.INFO, logger="frontdesk.channels.pairing"):
        for guess in ("1234", "0000", "abcd"):
            assert say(guesser, guess) == ()
        assert say(guesser, CODE) == ()  # locked: the right code no longer pairs
    assert inbox.count_testers() == 0
    alerts = [r for r in caplog.records if r.msg == "channel.pairing_locked"]
    assert [r.fields for r in alerts] == [{"scope": "session"}]  # type: ignore[attr-defined]
    assert say(honest, CODE) != ()
    assert inbox.count_testers() == 1
