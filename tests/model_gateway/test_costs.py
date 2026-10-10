"""The budget file, the spend ledger and the cap (M3 design §1, §4.4)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from frontdesk.model_gateway.costs import (
    Budget,
    BudgetError,
    BudgetExhaustedError,
    Price,
    load_budget,
)
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


@pytest.mark.parametrize(
    "line",
    [
        pytest.param('{"usd": NaN}', id="nan"),
        pytest.param('{"usd": "nan"}', id="nan-as-text"),
        pytest.param('{"usd": Infinity}', id="infinity"),
        pytest.param('{"usd": 1e999}', id="too-large-to-be-a-float"),
        pytest.param('{"usd": ' + "9" * 400 + "}", id="integer-beyond-a-float"),
        pytest.param('{"usd": -100}', id="negative"),
        pytest.param('{"usd": true}', id="boolean"),
        pytest.param('{"usd": "1.5"}', id="text"),
    ],
)
def test_a_ledger_line_that_is_not_an_amount_of_money_is_reported_and_opens_no_room(
    tmp_path: Path, line: str
) -> None:
    # Hand-edited values like these used to fail open: `nan > cap` is False, so room was granted.
    budget = write_budget(tmp_path)
    budget.ledger_path.write_text('{"usd": 1}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(BudgetError, match="line 2"):
        budget.spent_usd()
    with pytest.raises(BudgetError, match="line 2"):
        budget.ensure_room(50.0)


@pytest.mark.parametrize("usd", [float("nan"), float("inf"), -0.01])
def test_the_ledger_is_never_given_an_amount_that_is_not_money(tmp_path: Path, usd: float) -> None:
    budget = write_budget(tmp_path)
    with pytest.raises(BudgetError, match="not an amount"):
        budget.record(run_id="r", route_id=ROUTE, kind="chat", usd=usd)
    assert not budget.ledger_path.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("cap_eur: 10.0", "cap_eur: .inf"), "cap_eur"),
        (lambda text: text.replace("cap_eur: 10.0", "cap_eur: .nan"), "cap_eur"),
        (lambda text: text.replace("usd_per_eur: 1.25", "usd_per_eur: .inf"), "usd_per_eur"),
        (lambda text: text.replace("usd_per_eur: 1.25", "usd_per_eur: .nan"), "usd_per_eur"),
        (lambda text: text.replace("input_usd: 1.0", "input_usd: .inf"), "input_usd"),
        (lambda text: text.replace("output_usd: 10.0", "output_usd: .nan"), "output_usd"),
    ],
)
def test_a_budget_file_never_holds_an_infinite_or_missing_number(
    tmp_path: Path, change: Callable[[str], str], message: str
) -> None:
    # An infinite cap would be no cap at all.
    path = write_budget(tmp_path).path
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(BudgetError, match=message):
        load_budget(path)


def test_the_lock_names_the_ledger_directory_it_needs(tmp_path: Path) -> None:
    budget = write_budget(tmp_path)
    away = Budget(budget.path, budget.config.model_copy(update={"ledger": "gone/spend.jsonl"}))
    with pytest.raises(BudgetError, match=r"spend\.jsonl\.lock"), away.exclusive():
        pytest.fail("a run must not start without its ledger directory")


def test_over_the_cap_the_message_says_so_without_a_negative_amount(tmp_path: Path) -> None:
    budget = write_budget(tmp_path, cap_eur=1.0, usd_per_eur=1.0)
    budget.record(run_id="r", route_id=ROUTE, kind="chat", usd=5.0)
    with pytest.raises(BudgetExhaustedError) as caught:
        budget.ensure_room(0.01)
    message = str(caught.value)
    assert "has 0.0000 EUR left (over by 4.0000 EUR)" in message
    assert "-" not in message
