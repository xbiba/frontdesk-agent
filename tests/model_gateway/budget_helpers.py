"""A budget file for tests: prices in round numbers, so costs are easy to check by hand."""

from __future__ import annotations

from pathlib import Path

from frontdesk.model_gateway.costs import Budget, load_budget

ROUTE = "vertex-eu-gemini-flash-lite"


def write_budget(directory: Path, *, cap_eur: float = 10.0, usd_per_eur: float = 1.25) -> Budget:
    path = directory / "budget.yaml"
    path.write_text(
        f"""cap_eur: {cap_eur}
ledger: spend.jsonl
exchange_rate:
  usd_per_eur: {usd_per_eur}
  source: https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml
  date: 2026-10-09
prices:
  source: https://cloud.google.com/vertex-ai/generative-ai/pricing
  retrieved: 2026-10-10
  routes:
    {ROUTE}:
      unit: tokens_1m
      input_usd: 1.0
      output_usd: 10.0
    vertex-ew1-gemini-embedding-001:
      unit: characters_1k
      input_usd: 0.5
""",
        encoding="utf-8",
    )
    return load_budget(path)
