"""Post-processing: summaries, comparisons and CSV export for the MCF model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import MCFResult


def summary_table(result: MCFResult) -> pd.DataFrame:
    """Per-country overview: supply, demand, imports, exports, availability, coverage."""
    idx = result.supply.index.union(result.demand.index)
    imp = result.imports_received.reindex(idx, fill_value=0)
    exp = result.exports_sent.reindex(idx, fill_value=0)
    avail = result.availability().reindex(idx, fill_value=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = avail / result.demand.reindex(idx).replace(0, np.nan) * 100

    df = pd.DataFrame(
        {
            "supply": result.supply.reindex(idx, fill_value=0),
            "demand": result.demand.reindex(idx, fill_value=0),
            "imports_received": imp,
            "exports_sent": exp,
            "availability": avail,
            "coverage_%": coverage.round(1),
            "unmet_demand": (result.demand - avail).clip(lower=0),
        }
    )
    return df


def global_summary(baseline: MCFResult, shocked: MCFResult) -> pd.DataFrame:
    """Baseline vs shocked globals: supply, demand, total cost, total flow, feasibility."""
    rows = {
        "Total supply": (baseline.supply.sum(), shocked.supply.sum()),
        "Total demand": (baseline.demand.sum(), shocked.demand.sum()),
        "Total flow": (baseline.total_flow, shocked.total_flow),
        "Total cost": (baseline.total_cost, shocked.total_cost),
        "Feasible": (baseline.feasible, shocked.feasible),
    }
    df = pd.DataFrame(rows, index=["baseline", "shocked"]).T
    return df


def build_comparison(
    baseline: MCFResult, shocked: MCFResult, drop_zero_demand: bool = True
) -> pd.DataFrame:
    """Country-level comparison between a baseline and a shocked run."""
    sb, ss = summary_table(baseline), summary_table(shocked)
    idx = sb.index.union(ss.index)
    df = pd.DataFrame(
        {
            "supply_baseline": sb["supply"].reindex(idx, fill_value=0),
            "supply_shocked": ss["supply"].reindex(idx, fill_value=0),
            "demand": sb["demand"].reindex(idx, fill_value=0),
            "avail_baseline": sb["availability"].reindex(idx, fill_value=0),
            "avail_shocked": ss["availability"].reindex(idx, fill_value=0),
            "cov_base_%": sb["coverage_%"].reindex(idx),
            "cov_shock_%": ss["coverage_%"].reindex(idx),
        }
    )
    df["change_pp"] = df["cov_shock_%"] - df["cov_base_%"]
    if drop_zero_demand:
        df = df[df["demand"] > 0]
    return df.round(2)


def save_result(result: MCFResult, out_dir: str | Path, tag: str) -> dict[str, Path]:
    """Write flow matrix and summary to ``out_dir`` as two CSVs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / f"flow_{tag}.csv"
    s_path = out / f"summary_{tag}.csv"
    result.flow_matrix.to_csv(x_path)
    summary_table(result).to_csv(s_path)
    return {"flow": x_path, "summary": s_path}
