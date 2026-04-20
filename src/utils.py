"""Plotting helpers (plotly + matplotlib) for MCF results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import MCFResult


ALLFED_MPLSTYLE_URL = (
    "https://raw.githubusercontent.com/allfed/"
    "ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


def use_allfed_style() -> bool:
    """Activate the ALLFED matplotlib style sheet.

    Tries the online stylesheet first; falls back silently to the default
    matplotlib style if offline or if the URL is unreachable.

    Returns:
        bool: True if the ALLFED style was successfully applied.
    """
    import matplotlib.pyplot as plt

    try:
        plt.style.use(ALLFED_MPLSTYLE_URL)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Plotly
# ──────────────────────────────────────────────────────────────────────────────
def plot_flow_sankey(
    result: MCFResult,
    title: str = "Min-cost-flow — trade flows",
    min_flow: float = 1e-6,
    top_k: int | None = None,
):
    """Sankey diagram of the solved flow matrix.

    If ``top_k`` is given, only the ``top_k`` largest flows are shown.
    """
    import plotly.graph_objects as go

    M = result.flow_matrix
    stacked = M.stack()
    stacked = stacked[stacked > min_flow]
    if top_k is not None:
        stacked = stacked.nlargest(top_k)

    labels = list(
        dict.fromkeys([u for u, _ in stacked.index] + [v for _, v in stacked.index])
    )
    idx = {c: i for i, c in enumerate(labels)}

    src = [idx[u] for u, _ in stacked.index]
    tgt = [idx[v] for _, v in stacked.index]
    val = stacked.tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    label=labels,
                    line=dict(color="black", width=0.5),
                ),
                link=dict(
                    source=src,
                    target=tgt,
                    value=val,
                    color="rgba(100,149,237,0.45)",
                    hovertemplate="%{source.label} -> %{target.label}"
                    "<br>%{value:,.1f}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(title=title, font_size=11, height=560)
    return fig


def plot_flow_heatmap(
    result: MCFResult,
    title: str = "MCF flow matrix",
    top_n_countries: int = 25,
):
    """Heatmap of the flow matrix, limited to the ``top_n_countries`` largest players."""
    import plotly.express as px

    M = result.flow_matrix
    activity = M.sum(axis=1) + M.sum(axis=0)
    top = activity.nlargest(top_n_countries).index.tolist()
    sub = M.loc[top, top]
    fig = px.imshow(
        sub,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Importer", y="Exporter", color="Flow"),
        text_auto=".0f",
        title=title,
    )
    fig.update_layout(height=620)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib
# ──────────────────────────────────────────────────────────────────────────────
def plot_coverage_comparison(
    comparison: pd.DataFrame,
    title: str = "Coverage: baseline vs shocked",
    top_k: int = 20,
):
    """Horizontal grouped bar of the `top_k` most-affected countries."""
    import matplotlib.pyplot as plt

    use_allfed_style()

    top = comparison.nsmallest(top_k, "change_pp")
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        y, top["cov_base_%"], height=0.4, label="Baseline", color="#2980b9", alpha=0.85
    )
    ax.barh(
        y + 0.4,
        top["cov_shock_%"],
        height=0.4,
        label="Shocked",
        color="#c0392b",
        alpha=0.85,
    )
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Supply coverage (%)")
    ax.set_title(title)
    ax.axvline(100, color="gray", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig
