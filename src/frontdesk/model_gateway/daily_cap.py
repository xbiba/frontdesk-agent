"""The daily spend cap in production (Albi, 2026-10-13, before Instagram).

What the business's model requests cost each of its days is kept in Postgres (migration 0006).
Past the cap no request is sent (`SpendCapReachedError`, and the turn gives reception's contact),
until the business's next day. Two alerts go to the logs, for an alert policy: `spend.alert` when
the day reaches ALERT_SHARE of the cap, `spend.cap_reached` when it reaches the cap. The meter of the
live evaluations (frontdesk.model_gateway.metering) is another thing: a run's budget, not a day's.
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
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session
from frontdesk.db.models import DailySpend
from frontdesk.model_gateway.costs import Price
from frontdesk.observability.logs import log_event

ALERT_SHARE: Final = 0.8
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

    def check(self) -> None:
        """Raises SpendCapReachedError once today's spend has reached the cap."""
        if self._spent(self._today()) >= self.cap_usd:
            raise SpendCapReachedError(f"today's model spend reached the cap of {self.cap_usd} USD")

    def add(self, usd: float) -> None:
        """Today's spend grows by `usd`; crossing the alert share or the cap is logged once."""
        day = self._today()
        added = Decimal(str(usd))
        with business_session(self.session_factory, self.business_id) as session:
            statement = insert(DailySpend).values(business_id=self.business_id, day=day, usd=added)
            total = (
                session.scalar(
                    statement.on_conflict_do_update(
                        index_elements=["business_id", "day"],
                        set_={"usd": DailySpend.usd + statement.excluded.usd},
                    ).returning(DailySpend.usd)
                )
                or added
            )
        before = total - added
        for share, event in ((ALERT_SHARE, "spend.alert"), (1.0, "spend.cap_reached")):
            if before < Decimal(str(share * self.cap_usd)) <= total:
                log_event(
                    logger,
                    event,
                    business_id=self.business_id,
                    day=day.isoformat(),
                    usd=float(total),
                    cap_usd=self.cap_usd,
                )

    def _spent(self, day: date) -> float:
        with business_session(self.session_factory, self.business_id) as session:
            usd = session.scalar(
                select(DailySpend.usd).where(
                    DailySpend.business_id == self.business_id, DailySpend.day == day
                )
            )
        return float(usd or 0)

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
        await asyncio.to_thread(self._cap.check)
        response = await super().request(messages, model_settings, model_request_parameters)
        usage = response.usage
        usd = self._cap.price.cost(input_units=usage.input_tokens, output_units=usage.output_tokens)
        await asyncio.to_thread(self._cap.add, usd)
        return response
