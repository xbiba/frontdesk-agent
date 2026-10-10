"""What live calls cost, and the budget they may not exceed (M3 design §1, §4.4).

The budget file names the cap in euros, the exchange rate and the US-dollar list prices of each
route, each with the page it was read from and the date. Spend goes to a ledger beside the file,
one JSON line per paid call, so every live run of the phase counts against the same cap and a run
that stops loses nothing. The ledger is the state: the spend is read from it, never kept apart. It
keeps each cost unrounded, and one run at a time holds it, so it never counts less than was spent.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from frontdesk.tenancy.pack import NonBlankStr


class BudgetError(ValueError):
    """The budget file cannot be read, or it lacks what a run needs."""


class BudgetExhaustedError(RuntimeError):
    """The next call could cost more than the budget has left."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Price(_Frozen):
    """A list price in US dollars: per million tokens, or per thousand characters."""

    unit: Literal["tokens_1m", "characters_1k"]
    input_usd: float = Field(ge=0)
    output_usd: float = Field(default=0.0, ge=0)

    def cost(self, *, input_units: int, output_units: int = 0) -> float:
        per = 1_000_000 if self.unit == "tokens_1m" else 1_000
        return (input_units * self.input_usd + output_units * self.output_usd) / per


class ExchangeRate(_Frozen):
    usd_per_eur: float = Field(gt=0)
    source: NonBlankStr
    date: date


class PriceList(_Frozen):
    source: NonBlankStr
    retrieved: date
    routes: dict[str, Price] = Field(min_length=1)


class BudgetFile(_Frozen):
    cap_eur: float = Field(gt=0)
    ledger: NonBlankStr = "spend.jsonl"
    exchange_rate: ExchangeRate
    prices: PriceList


@dataclass(frozen=True)
class Budget:
    path: Path
    config: BudgetFile

    @property
    def ledger_path(self) -> Path:
        return self.path.parent / self.config.ledger

    @property
    def cap_usd(self) -> float:
        return self.config.cap_eur * self.config.exchange_rate.usd_per_eur

    def eur(self, usd: float) -> float:
        return usd / self.config.exchange_rate.usd_per_eur

    def price(self, route_id: str) -> Price:
        price = self.config.prices.routes.get(route_id)
        if price is None:
            raise BudgetError(f"{self.path}: no price for route '{route_id}'")
        return price

    def spent_usd(self) -> float:
        """Everything the ledger records; a missing ledger means nothing was spent yet."""
        if not self.ledger_path.exists():
            return 0.0
        try:
            text = self.ledger_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise BudgetError(f"{self.ledger_path}: is not UTF-8 text") from error
        total = 0.0
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                total += float(json.loads(line)["usd"])
            except (ValueError, KeyError, TypeError) as error:
                raise BudgetError(
                    f"{self.ledger_path}: line {number} is not a spend entry"
                ) from error
        return total

    def remaining_eur(self) -> float:
        return self.eur(self.cap_usd - self.spent_usd())

    def ensure_room(self, usd: float) -> None:
        """Refuse a call that could cost `usd` when that would go past the cap."""
        spent = self.spent_usd()
        if spent + usd > self.cap_usd:
            raise BudgetExhaustedError(
                f"the budget of {self.config.cap_eur:.2f} EUR has "
                f"{self.eur(self.cap_usd - spent):.4f} EUR left, and the next call may cost "
                f"{self.eur(usd):.4f} EUR"
            )

    def record(self, *, run_id: str, route_id: str, kind: str, usd: float) -> None:
        """Append one paid call to the ledger and make it durable before the next one. The cost is
        kept unrounded: rounded to the nearest, it could count less than was spent."""
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "run": run_id,
            "route": route_id,
            "kind": kind,
            "usd": usd,
        }
        with self.ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(entry) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the ledger for one run. Two runs at once would each see the same room and could
        together pass the cap, so the second is refused while the first holds it."""
        lock_path = self.ledger_path.with_name(f"{self.ledger_path.name}.lock")
        with lock_path.open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BudgetError(
                    f"{self.ledger_path}: another live run is using this budget; "
                    "run this one when it has ended"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_budget(path: Path) -> Budget:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BudgetError(f"{path}: cannot be read ({error.strerror or error})") from error
    except yaml.YAMLError as error:
        raise BudgetError(f"{path}: invalid YAML") from error
    try:
        return Budget(path, BudgetFile.model_validate(raw))
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
            for problem in error.errors(include_url=False, include_input=False)
        )
        raise BudgetError(f"{path}: {problems}") from error
