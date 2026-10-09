"""A bounded number of order lookups per conversation (anti-enumeration).

One INSERT ... ON CONFLICT DO UPDATE ... WHERE used < limit: concurrent turns
cannot both take the last attempt. The caller commits before looking up.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from frontdesk.db.models import OrderLookupAttempts

MAX_ORDER_LOOKUPS: Final = 3


def consume_order_lookup(
    session: Session,
    *,
    business_id: str,
    conversation_id: str,
    now: datetime,
    limit: int = MAX_ORDER_LOOKUPS,
) -> bool:
    statement = (
        insert(OrderLookupAttempts)
        .values(business_id=business_id, conversation_id=conversation_id, used=1, updated_at=now)
        .on_conflict_do_update(
            index_elements=["business_id", "conversation_id"],
            set_={"used": OrderLookupAttempts.used + 1, "updated_at": now},
            where=OrderLookupAttempts.used < limit,
        )
        .returning(OrderLookupAttempts.used)
    )
    return session.execute(statement).scalar_one_or_none() is not None
