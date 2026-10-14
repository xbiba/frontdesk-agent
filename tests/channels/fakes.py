"""Stand-ins for the channel tests: a clock that moves when told, a channel that records what it
sends, and the engine's turn."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from frontdesk.channels.messages import SendOutcome, SendStatus
from frontdesk.conversation.state import ConversationState
from frontdesk.conversation.turn import TurnOutcome, TurnResult, TurnTrace

NOW = datetime(2026, 10, 14, 9, tzinfo=UTC)


class Clock:
    def __init__(self, at: datetime = NOW) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def advance(self, **delta: float) -> None:
        self.at = self.at + timedelta(**delta)


class FakeSender:
    """The channel: records each send, answers with the outcomes it was given, then SENT."""

    name = "instagram"
    max_message_bytes = 1000

    def __init__(self, *outcomes: SendOutcome) -> None:
        self.sent: list[tuple[str, str]] = []
        self._outcomes = list(outcomes)

    async def send(self, user_id: str, text: str) -> SendOutcome:
        self.sent.append((user_id, text))
        if self._outcomes:
            return self._outcomes.pop(0)
        return SendOutcome(SendStatus.SENT, message_id=f"out-{len(self.sent)}")


class FakeTurn:
    """The engine: it answers "Risposta: <text>" in Italian, or what `reply` makes of the text."""

    def __init__(self, reply: Callable[[str], str] | None = None, rule: str = "test") -> None:
        self.texts: list[str] = []
        self.states: list[ConversationState] = []
        self._reply = reply or (lambda text: f"Risposta: {text}")
        self._rule = rule

    async def __call__(
        self, text: str, state: ConversationState
    ) -> tuple[TurnResult, ConversationState]:
        self.texts.append(text)
        self.states.append(state)
        trace = TurnTrace(rule=self._rule)
        result = TurnResult(self._reply(text), TurnOutcome.REPLIED, "it", trace)
        return result, state.model_copy(update={"turns": state.turns + 1, "language": "it"})
