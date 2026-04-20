"""Data loading helpers for the min-cost-flow model.

Reads the two FAOSTAT CSV extracts (``production_*.csv`` and
``trade_*.csv``) that the original scripts used, and returns tidy objects
shaped for :class:`src.model.FertilizerMCF`:

* ``supply`` : pd.Series   — production per country
* ``demand`` : pd.Series   — estimated consumption (production + imports - exports)
* ``edges``  : pd.DataFrame with columns ``from``, ``to``, ``cost``, ``capacity``

Unit convention throughout: tonnes (FAOSTAT "Value" for Nutrient items).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ELEMENT_IMPORT = 5610
ELEMENT_EXPORT = 5910


# ──────────────────────────────────────────────────────────────────────────────
# Production
# ──────────────────────────────────────────────────────────────────────────────
def load_production(path: str | Path, year: int | None = None) -> pd.Series:
    """Load a FAOSTAT production CSV and return a country-indexed Series."""
    df = pd.read_csv(path, low_memory=False)
    if year is not None and "Year" in df.columns:
        df = df[df["Year"] == year]
    df = df[["Area", "Value"]].dropna()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0).clip(lower=0)
    s = df.groupby("Area")["Value"].sum()
    s.name = "production"
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Trade → supply / demand / edges
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_year_col(df: pd.DataFrame, year: int) -> str:
    col = f"Y{year}"
    if col in df.columns:
        return col
    y_cols = sorted(c for c in df.columns if c.startswith("Y"))
    if not y_cols:
        raise ValueError("No Y-prefixed year columns found in trade dataframe.")
    return y_cols[-1]  # fall back to most recent


def load_trade(path: str | Path) -> pd.DataFrame:
    """Load a FAOSTAT trade CSV (raw)."""
    return pd.read_csv(path, low_memory=False)


def country_flows(trade_df: pd.DataFrame, year: int) -> tuple[pd.Series, pd.Series]:
    """Aggregate imports and exports per *reporter* country for ``year``.

    Returns
    -------
    total_import, total_export
        Two country-indexed Series in tonnes.
    """
    col = _resolve_year_col(trade_df, year)
    df = trade_df[["Reporter Countries", "Element Code", col]].copy()
    df = df.rename(columns={"Reporter Countries": "Reporter", col: "Value"})
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Reporter", "Value"])
    df = df[df["Value"] > 0]

    imp = df[df["Element Code"] == ELEMENT_IMPORT].groupby("Reporter")["Value"].sum()
    exp = df[df["Element Code"] == ELEMENT_EXPORT].groupby("Reporter")["Value"].sum()
    imp.name, exp.name = "total_import", "total_export"
    return imp, exp


def estimate_demand(
    production: pd.Series,
    total_import: pd.Series,
    total_export: pd.Series,
    normalise_to_supply: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(supply, demand)`` series aligned on the union of indices.

    * ``supply_i   = production_i + total_import_i``
    * ``demand_i   = max(0, production_i + total_import_i - total_export_i)``

    If ``normalise_to_supply`` is True (default), demand is rescaled so
    that ``demand.sum() == supply.sum()`` (common FAOSTAT reporting
    mismatch; keeps the LP feasible).
    """
    countries = sorted(
        set(production.index) | set(total_import.index) | set(total_export.index)
    )
    prod = production.reindex(countries, fill_value=0)
    imp = total_import.reindex(countries, fill_value=0)
    exp = total_export.reindex(countries, fill_value=0)

    supply = prod + imp
    demand = (prod + imp - exp).clip(lower=0)
    supply.name, demand.name = "supply", "demand"

    if normalise_to_supply and demand.sum() > 0 and supply.sum() > 0:
        demand = demand * (supply.sum() / demand.sum())
    return supply, demand


# ──────────────────────────────────────────────────────────────────────────────
# Edges
# ──────────────────────────────────────────────────────────────────────────────
def build_edges(
    trade_df: pd.DataFrame,
    year: int,
    countries: set[str] | None = None,
    cost: str = "inverse_volume",
    capacity_factor: float = 1.0,
) -> pd.DataFrame:
    """Build an edges DataFrame from the FAOSTAT detailed trade matrix.

    Parameters
    ----------
    trade_df
        FAOSTAT detailed trade matrix (must have ``Reporter Countries``,
        ``Partner Countries``, ``Element Code`` and a ``Y{year}`` column).
    year
        Calendar year to use.
    countries
        Optional set of countries to restrict to. If given, edges whose
        endpoints are outside this set are dropped.
    cost
        How to compute the edge cost:

        * ``"inverse_volume"`` — ``cost = 1 / (value + 1)``
          (used in the original `min_cost_flow_npk.py`; established trade
          routes get a lower cost per unit flow).
        * ``"unit"`` — uniform cost of 1.0 on every edge (shortest-route
          behaviour, effectively a max-flow with minimum hops).

        *Note:* NetworkX ``min_cost_flow`` requires **non-negative**
        integer-or-float weights.
    capacity_factor
        Multiplier applied to the historical volume to get the edge
        capacity (``1.0`` = keep historical volume; ``0.5`` = simulate
        50% loss of shipping capacity, etc.).

    Returns
    -------
    pd.DataFrame with columns ``from``, ``to``, ``cost``, ``capacity``.
    """
    if cost not in {"inverse_volume", "unit"}:
        raise ValueError(f"Unknown cost mode: {cost!r}")

    col = _resolve_year_col(trade_df, year)
    df = trade_df[
        ["Reporter Countries", "Partner Countries", "Element Code", col]
    ].copy()
    df = df.rename(
        columns={
            "Reporter Countries": "Reporter",
            "Partner Countries": "Partner",
            col: "Value",
        }
    )
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Reporter", "Partner", "Value"])
    df = df[df["Value"] > 0]

    exports = df[df["Element Code"] == ELEMENT_EXPORT].rename(
        columns={"Reporter": "from", "Partner": "to"}
    )[["from", "to", "Value"]]
    # Imports reported by the receiver: reverse the direction.
    imports = df[df["Element Code"] == ELEMENT_IMPORT].rename(
        columns={"Partner": "from", "Reporter": "to"}
    )[["from", "to", "Value"]]

    edges = pd.concat([exports, imports], ignore_index=True)
    edges = edges.groupby(["from", "to"], as_index=False)["Value"].sum()

    if countries is not None:
        edges = edges[edges["from"].isin(countries) & edges["to"].isin(countries)]

    if cost == "inverse_volume":
        edges["cost"] = 1.0 / (edges["Value"] + 1.0)
    else:  # "unit"
        edges["cost"] = 1.0
    edges["capacity"] = edges["Value"] * capacity_factor
    edges = edges.drop(columns=["Value"]).reset_index(drop=True)
    return edges


# ──────────────────────────────────────────────────────────────────────────────
# One-shot convenience
# ──────────────────────────────────────────────────────────────────────────────
def load_scenario(
    production_path: str | Path,
    trade_path: str | Path,
    year: int,
    cost: str = "inverse_volume",
    capacity_factor: float = 1.0,
) -> dict[str, object]:
    """Load production + trade for one year and return
    ``{countries, supply, demand, edges}``.
    """
    prod = load_production(production_path, year=year)
    trade = load_trade(trade_path)
    imp, exp = country_flows(trade, year=year)
    supply, demand = estimate_demand(prod, imp, exp)
    countries = sorted(set(supply.index))
    edges = build_edges(
        trade,
        year=year,
        countries=set(countries),
        cost=cost,
        capacity_factor=capacity_factor,
    )
    return {
        "countries": countries,
        "supply": supply,
        "demand": demand,
        "edges": edges,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shocks
# ──────────────────────────────────────────────────────────────────────────────
def apply_supply_shock(supply: pd.Series, shock: dict[str, float]) -> pd.Series:
    """Multiply ``supply[c]`` by ``factor`` for each ``c`` in ``shock``.

    ``shock`` maps country -> surviving fraction (``0.4`` = 60% cut).
    """
    out = supply.copy()
    for c, f in shock.items():
        if c in out.index:
            out[c] = out[c] * f
    return out


def apply_edge_shock(
    edges: pd.DataFrame,
    per_edge: dict[tuple[str, str], float] | None = None,
    global_factor: float = 1.0,
) -> pd.DataFrame:
    """Reduce edge capacities.

    ``global_factor`` scales every edge; ``per_edge`` then overrides
    specific ``(from, to)`` pairs.  Use this to model shipping
    disruptions, port closures, sanctions, etc.
    """
    out = edges.copy()
    out["capacity"] = out["capacity"] * global_factor
    if per_edge:
        for (u, v), factor in per_edge.items():
            mask = (out["from"] == u) & (out["to"] == v)
            out.loc[mask, "capacity"] = out.loc[mask, "capacity"] * factor
    return out
