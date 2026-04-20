"""Min-cost-flow LP model for global fertilizer trade.

Given per-country supply (production + imports accepted by default) and
demand (observed consumption), plus a set of allowed trade edges with
unit costs and capacities, we solve the linear program

    minimise    sum_{(i,j) in E} c_{ij} x_{ij}
    subject to  sum_j x_{ji} - sum_j x_{ij}  =  d_i - s_i   for all i
                0 <= x_{ij} <= u_{ij}                       for all (i,j)

where

* ``s_i``   = supply of country i
* ``d_i``   = demand of country i
* ``c_{ij}``= unit transport cost on edge i->j
* ``u_{ij}``= capacity of edge i->j
* ``x_{ij}``= flow we're solving for (t / y in the same unit as s, d).

NetworkX's ``min_cost_flow`` solver does the heavy lifting; this module
just builds the graph, massages the per-node net demand so it sums to
exactly zero (a NetworkX precondition), and wraps the result in a
handy :class:`MCFResult` dataclass.

The full derivation, unit conventions and assumptions live in
``docs/methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


DEFAULT_LARGE_CAPACITY = 1e12  # "infinity" for uncapped edges


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MCFResult:
    """Outputs of one min-cost-flow solve.

    Attributes
    ----------
    supply, demand
        Per-country supply and demand (same units as the edge flows).
    flow
        Nested dict ``{from: {to: flow}}`` as returned by NetworkX.
    flow_matrix
        The same ``flow`` as a DataFrame (row = exporter, col = importer).
    total_cost
        Sum of ``cost * flow`` across all edges.
    total_flow
        Sum of ``flow`` across all edges.
    feasible
        Whether NetworkX found a feasible solution.
    """

    supply: pd.Series
    demand: pd.Series
    flow: dict[str, dict[str, float]]
    flow_matrix: pd.DataFrame
    total_cost: float
    total_flow: float
    feasible: bool

    @property
    def imports_received(self) -> pd.Series:
        s = self.flow_matrix.sum(axis=0)
        s.name = "imports_received"
        return s

    @property
    def exports_sent(self) -> pd.Series:
        s = self.flow_matrix.sum(axis=1)
        s.name = "exports_sent"
        return s

    def availability(self) -> pd.Series:
        """``supply + imports - exports`` per country."""
        idx = self.supply.index.union(self.demand.index)
        s = self.supply.reindex(idx, fill_value=0)
        f_in = self.imports_received.reindex(idx, fill_value=0)
        f_out = self.exports_sent.reindex(idx, fill_value=0)
        out = s + f_in - f_out
        out.name = "availability"
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class FertilizerMCF:
    """Fertilizer trade min-cost-flow LP.

    Parameters
    ----------
    supply, demand
        Country-indexed Series of supply (production + carry-in) and
        demand (consumption). Must be non-negative. Countries can appear
        in either, both or none — missing entries default to zero.
    edges
        DataFrame with at least three columns: ``from``, ``to``, ``cost``.
        Optional column ``capacity`` (defaults to a very large number,
        i.e. effectively uncapped). Edges with ``cost < 0`` are not
        supported (NetworkX requires non-negative weights for
        ``min_cost_flow``).
    balance
        * ``"drop_largest"`` (default): if total demand != total supply,
          scale whichever side is larger so that the totals match.
          Keeps the LP feasible.
        * ``"raise"``: raise if totals differ by more than ``1e-6``.
        * ``"none"``: pass the imbalance straight to NetworkX (will
          raise :class:`networkx.NetworkXUnfeasible` if it can't be
          satisfied).
    """

    def __init__(
        self,
        supply: pd.Series,
        demand: pd.Series,
        edges: pd.DataFrame,
        balance: str = "drop_largest",
    ) -> None:
        countries = sorted(
            set(supply.index)
            | set(demand.index)
            | set(edges["from"])
            | set(edges["to"])
        )
        self.countries = countries
        self.supply = (
            supply.reindex(countries, fill_value=0).astype(float).clip(lower=0)
        )
        self.supply.name = "supply"
        self.demand = (
            demand.reindex(countries, fill_value=0).astype(float).clip(lower=0)
        )
        self.demand.name = "demand"

        required = {"from", "to", "cost"}
        if missing := required - set(edges.columns):
            raise ValueError(f"edges DataFrame is missing columns: {missing}")
        edges = edges.copy()
        if "capacity" not in edges.columns:
            edges["capacity"] = DEFAULT_LARGE_CAPACITY
        if (edges["cost"] < 0).any():
            raise ValueError(
                "Negative edge costs are not supported by "
                "networkx.min_cost_flow — shift them to non-negative."
            )
        self.edges = edges.reset_index(drop=True)

        if balance not in {"drop_largest", "raise", "none"}:
            raise ValueError(f"Unknown balance mode: {balance!r}")
        self.balance = balance

    # ── Graph construction ────────────────────────────────────────────────────
    def build_graph(self) -> nx.DiGraph:
        """Build the NetworkX DiGraph with per-node demand and per-edge weights."""
        G = nx.DiGraph()
        # Per NetworkX convention, node ``demand`` is positive for sinks
        # (need inflow) and negative for sources (provide outflow).
        net = self.demand - self.supply

        if self.balance == "drop_largest":
            net = _rebalance(net)
        elif self.balance == "raise":
            if abs(net.sum()) > 1e-6:
                raise ValueError(
                    f"supply-demand imbalance is {net.sum():.4f}; "
                    "pass balance='drop_largest' to auto-fix."
                )

        for c in self.countries:
            G.add_node(c, demand=float(net[c]))

        for _, row in self.edges.iterrows():
            G.add_edge(
                row["from"],
                row["to"],
                weight=float(row["cost"]),
                capacity=float(row["capacity"]),
            )
        return G

    # ── Solve ─────────────────────────────────────────────────────────────────
    def solve(self) -> MCFResult:
        """Run the LP and return an :class:`MCFResult`."""
        G = self.build_graph()

        try:
            flow = nx.min_cost_flow(G, capacity="capacity", weight="weight")
            feasible = True
        except nx.NetworkXUnfeasible:
            flow = {n: {} for n in G.nodes()}
            feasible = False

        flow_matrix = _flow_to_matrix(flow, self.countries)

        total_cost, total_flow = 0.0, 0.0
        for u, outs in flow.items():
            for v, f in outs.items():
                if f > 0 and G.has_edge(u, v):
                    total_cost += G[u][v].get("weight", 0.0) * f
                    total_flow += f

        return MCFResult(
            supply=self.supply,
            demand=self.demand,
            flow=flow,
            flow_matrix=flow_matrix,
            total_cost=float(total_cost),
            total_flow=float(total_flow),
            feasible=feasible,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _rebalance(net: pd.Series) -> pd.Series:
    """Scale the larger side (sinks or sources) so that ``net.sum() == 0``."""
    pos = net[net > 0].sum()  # total demand (sinks)
    neg = -net[net < 0].sum()  # total supply (sources)

    if pos == 0 or neg == 0:
        return net * 0.0

    out = net.copy()
    if pos > neg:
        out[out > 0] *= neg / pos
    elif neg > pos:
        out[out < 0] *= pos / neg

    # NetworkX requires *exact* balance; the scaling above leaves tiny FP
    # residuals. Sweep them onto the single largest node.
    residual = float(out.sum())
    if abs(residual) > 0:
        largest = out.abs().idxmax()
        out[largest] -= residual
    return out


def _flow_to_matrix(
    flow: dict[str, dict[str, float]], countries: list[str]
) -> pd.DataFrame:
    """Convert ``networkx.min_cost_flow`` output to a dense DataFrame."""
    M = pd.DataFrame(0.0, index=countries, columns=countries)
    for u, outs in flow.items():
        for v, f in outs.items():
            if u in M.index and v in M.columns:
                M.loc[u, v] = float(f)
    return M
