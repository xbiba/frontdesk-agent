"""The budget file, the spend ledger and the cap (M3 design §1, §4.4)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from frontdesk.model_gateway.costs import BudgetError, BudgetExhaustedError, Price, load_budget
from tests.model_gateway.budget_helpers import ROUTE, write_budget


def test_a_price_per_million_tokens_counts_input_and_output() -> None:
    price = Price(unit="tokens_1m", input_usd=0.275, output_usd=1.65)
    assert price.cost(input_units=1_000_000, output_units=0) == pytest.approx(0.275)
    assert price.cost(input_units=2_000, output_units=100) == pytest.approx(0.00055 + 0.000165)


def test_a_price_per_thousand_characters() -> None:
    assert Price(unit="characters_1k", input_usd=0.00015).cost(input_units=4_000) == pytest.approx(
        0.0006
    )


def test_the_cap_is_in_euros_and_the_prices_in_dollars(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=10.0, usd_per_eur=1.25)
    assert budget.cap_usd == pytest.approx(12.5)
    assert budget.eur(2.5) == pytest.approx(2.0)
    assert budget.remaining_eur() == pytest.approx(10.0)


def test_every_paid_call_is_a_ledger_line_and_the_spend_is_their_sum(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.record(run_id="r1", route_id=ROUTE, kind="chat", usd=1.25)
    budget.record(run_id="r2", route_id=ROUTE, kind="chat", usd=2.5)
    lines = budget.ledger_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run"] for line in lines] == ["r1", "r2"]
    assert budget.spent_usd() == pytest.approx(3.75)
    assert budget.remaining_eur() == pytest.approx(7.0)
    # Every run of the phase reads the same ledger: a new Budget sees the spend so far.
    assert load_budget(budget.path).spent_usd() == pytest.approx(3.75)


def test_a_call_that_could_pass_the_cap_is_refused(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=1.0, usd_per_eur=1.0)
    budget.record(run_id="r", route_id=ROUTE, kind="chat", usd=0.9)
    budget.ensure_room(0.1)  # exactly the cap is still allowed
    with pytest.raises(BudgetExhaustedError, match=r"has 0\.1000 EUR left"):
        budget.ensure_room(0.11)


def test_a_route_without_a_price_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="no price for route 'vertex-eu-opus'"):
        write_budget(tmp_path).price("vertex-eu-opus")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("cap_eur: 10.0", "cap_eur: 0"), "cap_eur"),
        (lambda text: text.replace("usd_per_eur: 1.25", "usd_per_eur: -1"), "usd_per_eur"),
        (lambda text: text.replace("unit: tokens_1m", "unit: requests"), "unit"),
        (lambda text: text.replace("  retrieved: 2026-10-10\n", ""), "prices.retrieved"),
    ],
)
def test_a_malformed_budget_file_is_refused_naming_the_field(
    tmp_path: Path, change: Callable[[str], str], message: str
) -> None:
    path = write_budget(tmp_path).path
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(BudgetError, match=message):
        load_budget(path)


def test_a_damaged_ledger_line_is_reported_not_skipped(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text('{"usd": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(BudgetError, match="line 2"):
        budget.spent_usd()


def test_the_ledger_keeps_each_cost_unrounded(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    budget.record(run_id="r", route_id=ROUTE, kind="chat", usd=4e-9)
    assert budget.spent_usd() == pytest.approx(4e-9)  # rounded, it would have counted nothing


def test_two_runs_cannot_hold_the_budget_at_once(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    with (
        budget.exclusive(),
        pytest.raises(BudgetError, match="another live run is using this budget"),
        load_budget(budget.path).exclusive(),
    ):
        pass
    with budget.exclusive():  # released when the first run ended
        pass


def test_a_budget_file_that_cannot_be_read_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match=r"budget\.yaml: cannot be read"):
        load_budget(tmp_path / "budget.yaml")


def test_a_budget_file_that_is_not_yaml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "budget.yaml"
    path.write_text("cap_eur: [", encoding="utf-8")
    with pytest.raises(BudgetError, match="invalid YAML"):
        load_budget(path)


def test_a_missing_ledger_means_nothing_was_spent_and_blank_lines_count_for_nothing(
    tmp_path: Path,
) -> None:
    budget = write_budget(tmp_path)
    assert not budget.ledger_path.exists()
    assert budget.spent_usd() == 0.0
    budget.ledger_path.write_text('{"usd": 1}\n\n{"usd": 2}\n', encoding="utf-8")
    assert budget.spent_usd() == pytest.approx(3.0)
