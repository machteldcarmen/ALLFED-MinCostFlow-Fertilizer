# src

Core Python package for the min-cost-flow (MCF) model.

| Module              | What it does                                                                 |
|---------------------|------------------------------------------------------------------------------|
| `model.py`          | `FertilizerMCF` class + `MCFResult` dataclass (LP construction and solve)    |
| `preprocessing.py`  | FAOSTAT loaders; builds supply/demand vectors and edges list                 |
| `postprocessing.py` | Summary tables, baseline-vs-shock comparison, CSV export                     |
| `utils.py`          | Plotly + matplotlib helpers (Sankey, heatmap, coverage comparison bars)      |

For the equations, see `../docs/methodology.md`.
