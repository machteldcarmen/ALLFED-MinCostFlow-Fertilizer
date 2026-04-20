"""Unit tests for allfed-fertilizer-min-cost-flow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import FertilizerMCF, _rebalance  # noqa: E402
from src.postprocessing import summary_table  # noqa: E402


@pytest.fixture
def toy_scenario():
    supply = pd.Series({"A": 100, "B": 80, "C": 20, "D": 30})
    demand = pd.Series({"A": 40, "B": 30, "C": 70, "D": 90})
    edges = pd.DataFrame(
        [
            ("A", "C", 1.0, 1000),
            ("A", "D", 2.5, 1000),
            ("B", "C", 1.5, 1000),
            ("B", "D", 0.8, 1000),
            ("A", "B", 0.2, 1000),
        ],
        columns=["from", "to", "cost", "capacity"],
    )
    return supply, demand, edges


def test_basic_solve_is_feasible(toy_scenario):
    supply, demand, edges = toy_scenario
    r = FertilizerMCF(supply, demand, edges).solve()
    assert r.feasible
    assert r.total_flow > 0
    assert r.total_cost >= 0


def test_all_countries_receive_their_demand(toy_scenario):
    supply, demand, edges = toy_scenario
    r = FertilizerMCF(supply, demand, edges).solve()
    avail = r.availability()
    for c in demand.index:
        assert avail[c] == pytest.approx(
            demand[c], abs=1e-6
        ), f"{c}: avail={avail[c]} demand={demand[c]}"


def test_mass_balance_per_country(toy_scenario):
    supply, demand, edges = toy_scenario
    r = FertilizerMCF(supply, demand, edges).solve()
    M = r.flow_matrix
    for c in M.index:
        inflow = M[c].sum()
        outflow = M.loc[c].sum()
        # supply + imports - exports should equal "consumed" = demand after rebalance
        assert supply[c] + inflow - outflow >= -1e-6
        assert supply[c] + inflow - outflow <= demand[c] + 1e-6


def test_rebalance_sums_to_exactly_zero():
    net = pd.Series({"A": -100.0, "B": -50.0, "C": 70.0, "D": 60.0})
    out = _rebalance(net)
    assert abs(out.sum()) < 1e-10


def test_negative_cost_raises():
    supply = pd.Series({"A": 10, "B": 0})
    demand = pd.Series({"A": 0, "B": 10})
    edges = pd.DataFrame(
        [("A", "B", -1.0, 100)], columns=["from", "to", "cost", "capacity"]
    )
    with pytest.raises(ValueError, match="Negative edge costs"):
        FertilizerMCF(supply, demand, edges)


def test_missing_column_raises():
    supply = pd.Series({"A": 10, "B": 0})
    demand = pd.Series({"A": 0, "B": 10})
    edges = pd.DataFrame([("A", "B", 1.0)], columns=["from", "to", "cost"])
    # no capacity column -> OK (has a default)
    FertilizerMCF(supply, demand, edges)

    bad_edges = pd.DataFrame([("A", "B")], columns=["from", "to"])
    with pytest.raises(ValueError, match="missing columns"):
        FertilizerMCF(supply, demand, bad_edges)


def test_shock_increases_cost_or_makes_infeasible(toy_scenario):
    supply, demand, edges = toy_scenario
    baseline = FertilizerMCF(supply, demand, edges).solve()
    shocked_supply = supply.copy()
    shocked_supply["A"] *= 0.2  # cut the biggest producer
    shocked = FertilizerMCF(shocked_supply, demand, edges).solve()
    if shocked.feasible:
        assert shocked.total_cost >= baseline.total_cost - 1e-6
    else:
        # infeasibility is a valid outcome; no flow is returned
        assert shocked.total_flow == 0


def test_summary_table_has_expected_columns(toy_scenario):
    supply, demand, edges = toy_scenario
    r = FertilizerMCF(supply, demand, edges).solve()
    df = summary_table(r)
    expected = {
        "supply",
        "demand",
        "imports_received",
        "exports_sent",
        "availability",
        "coverage_%",
        "unmet_demand",
    }
    assert expected.issubset(df.columns)
