"""Synthetic order status for the support-sales demo (a real shop reads its own order system).

The lookup gives the same answer for an unknown order and for a wrong postal
code, so it cannot be used to discover which orders exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.tenancy.pack import Language, Order

_LABELS: dict[str, dict[str, str]] = {
    "order": {"sq": "Porosia", "it": "Ordine", "en": "Order"},
    "eta": {"sq": "Dorëzimi i pritur", "it": "Consegna prevista", "en": "Expected delivery"},
}


@dataclass(frozen=True)
class OrderStore:
    orders: tuple[Order, ...]

    def lookup(
        self, reference: str, postal_code: str, *, language: Language, now: datetime
    ) -> Evidence | None:
        for order in self.orders:
            if order.reference == reference and order.postal_code == postal_code:
                return _evidence(order, language, now)
        return None


def _evidence(order: Order, language: Language, now: datetime) -> Evidence:
    status = order.status.get(language) or next(iter(order.status.values()))
    content = f"{_LABELS['order'][language]} {order.reference}: {status}"
    values = {"status": status}
    if order.eta is not None:
        content += f". {_LABELS['eta'][language]}: {order.eta.isoformat()}"
        values["eta"] = order.eta.isoformat()
    return Evidence.create(
        kind=EvidenceKind.LIVE,
        source_ref=f"order:{order.reference}",
        version=now.isoformat(timespec="seconds"),
        title=_LABELS["order"][language],
        content=content,
        values=values,
    )
