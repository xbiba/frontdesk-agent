"""The daily spend cap in production (Albi, 2026-10-13, before Instagram).

What the business's model requests cost each of its days is kept in Postgres (migration 0006).
Before a request, its worst case is held against the cap in one statement, which refuses it once
the day has reached the cap (`SpendCapReachedError`; the turn gives reception's contact), so that
requests in parallel cannot all slip under it. After the answer, the request's cost replaces its
hold. A request that ends without one (cancelled, cut, answered without usage) keeps its hold, as
the provider may have billed it; one the provider answered with an error status, or that never
left, gives it back. Two alerts go to the logs for an alert policy, each once a day: `spend.alert`
at ALERT_SHARE of the cap, `spend.cap_reached` at the cap. The meter of the live evaluations
(frontdesk.model_gateway.metering) is another thing: a run's budget, not a day's.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Final
from zoneinfo import ZoneInfo

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session
from frontdesk.db.models import DailySpend
from frontdesk.model_gateway.costs import Price
from frontdesk.model_gateway.metering import AGENT_INPUT_TOKENS_PER_REQUEST, UNBILLED
from frontdesk.observability.logs import log_event

ALERT_SHARE: Final = 0.8
DEFAULT_OUTPUT_TOKENS: Final = 2_000  # a request's output when its settings name no cap
_ALERTS: Final = ((1, ALERT_SHARE, "spend.alert"), (2, 1.0, "spend.cap_reached"))
logger = logging.getLogger(__name__)


class SpendCapReachedError(RuntimeError):
    """The business's model requests reached today's cap: no request is sent."""


@dataclass(frozen=True)
class DailyCap:
    session_factory: sessionmaker[Session]
    business_id: str
    cap_usd: float
    price: Price
    zone: ZoneInfo
    clock: Callable[[], datetime]

    def hold(self, usd: float) -> date:
        """Holds `usd` of today's cap; raises SpendCapReachedError once the day has reached it.
        The day it was held on, for `settle`."""
        day = self._today()
        held = Decimal(str(usd))
        with business_session(self.session_factory, self.business_id) as session:
            statement = insert(DailySpend).values(business_id=self.business_id, day=day, usd=held)
            row = session.execute(
                statement.on_conflict_do_update(
                    index_elements=["business_id", "day"],
                    set_={"usd": DailySpend.usd + statement.excluded.usd},
                    where=DailySpend.usd < Decimal(str(self.cap_usd)),
                ).returning(DailySpend.usd, DailySpend.alerted)
            ).one_or_none()
            if row is None:
                raise SpendCapReachedError(
                    f"today's model spend reached the cap of {self.cap_usd} USD"
                )
            self._alert(session, day, row.usd, row.alerted)
        return day

    def settle(self, day: date, held: float, used: float) -> None:
        """The request's cost replaces its hold."""
        change = Decimal(str(used)) - Decimal(str(held))
        with business_session(self.session_factory, self.business_id) as session:
            row = session.execute(
                update(DailySpend)
                .where(DailySpend.business_id == self.business_id, DailySpend.day == day)
                .values(usd=DailySpend.usd + change)
                .returning(DailySpend.usd, DailySpend.alerted)
            ).one()
            self._alert(session, day, row.usd, row.alerted)

    def _alert(self, session: Session, day: date, usd: Decimal, alerted: int) -> None:
        """Each alert once a day, when the day's spend reaches its share of the cap."""
        reached = max(
            (level for level, share, _ in _ALERTS if usd >= Decimal(str(share * self.cap_usd))),
            default=0,
        )
        for level, _, event in _ALERTS:
            if alerted < level <= reached:
                log_event(
                    logger,
                    event,
                    business_id=self.business_id,
                    day=day.isoformat(),
                    usd=float(usd),
                    cap_usd=self.cap_usd,
                )
        if reached > alerted:
            session.execute(
                update(DailySpend)
                .where(DailySpend.business_id == self.business_id, DailySpend.day == day)
                .values(alerted=reached)
            )

    def _today(self) -> date:
        return self.clock().astimezone(self.zone).date()


class CappedModel(WrapperModel):
    """A model whose requests stop at the business's daily cap."""

    def __init__(self, wrapped: Model, cap: DailyCap) -> None:
        super().__init__(wrapped)
        self._cap = cap

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        output = (model_settings or {}).get("max_tokens") or DEFAULT_OUTPUT_TOKENS
        held = self._cap.price.cost(input_units=AGENT_INPUT_TOKENS_PER_REQUEST, output_units=output)
        day = await asyncio.to_thread(self._cap.hold, held)
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except UNBILLED:
            await asyncio.to_thread(self._cap.settle, day, held, 0.0)  # nothing was billed
            raise
        usage = response.usage
        if usage.input_tokens + usage.output_tokens > 0:  # without usage, the hold stays
            used = self._cap.price.cost(
                input_units=usage.input_tokens, output_units=usage.output_tokens
            )
            await asyncio.to_thread(self._cap.settle, day, held, used)
        return response
