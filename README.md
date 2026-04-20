# allfed-fertilizer-min-cost-flow

[![Testing](https://github.com/machteldcarmen/ALLFED-MinCostFlow-Fertilizer/actions/workflows/testing.yml/badge.svg)](https://github.com/machteldcarmen/ALLFED-MinCostFlow-Fertilizer/actions/workflows/testing.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A **min-cost-flow (linear-programming) model** of the global fertilizer trade
network, built on [NetworkX](https://networkx.org/) and FAOSTAT data.

Given per-country supply and demand and a set of trade routes with unit
costs and capacities, the model finds the **cheapest combination of
flows** that satisfies all demand (or tells you the problem is
infeasible and shows which countries can't be served).

This is a companion to [`allfed-fertilizer-ras`](../allfed-fertilizer-ras)
— same data, different modelling assumption:

| Question                                                        | Model                        |
|-----------------------------------------------------------------|------------------------------|
| Given a shock, how do **actual historical** trade patterns adjust? | RAS (iterative proportional fitting) |
| Given a shock, what would the **cost-optimal** trade network look like? | Min-cost-flow (this repo)             |

---

## Installation

```bash
git clone https://github.com/<your-username>/allfed-fertilizer-min-cost-flow.git
cd allfed-fertilizer-min-cost-flow
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
```

Python >= 3.10.

---

## Quick start

```python
import pandas as pd
from src.model import FertilizerMCF

supply = pd.Series({"A": 100, "B":  80, "C":  20, "D":  30})
demand = pd.Series({"A":  40, "B":  30, "C":  70, "D":  90})

edges = pd.DataFrame([
    ("A", "C", 1.0, 1000),
    ("A", "D", 2.5, 1000),
    ("B", "C", 1.5, 1000),
    ("B", "D", 0.8, 1000),
    ("A", "B", 0.2, 1000),
], columns=["from", "to", "cost", "capacity"])

result = FertilizerMCF(supply, demand, edges).solve()
print(result.flow_matrix)
print("total cost =", result.total_cost)
```

For a real run on FAOSTAT nitrogen data, see
[`scripts/real_data.ipynb`](scripts/real_data.ipynb).

---

## What the model is solving

Variables: the flow `x[i,j]` on each edge `(i, j)` (tonnes of fertilizer).

**Objective:**

```
minimise   Σ_{(i,j) in E}  c_{ij} · x_{ij}
```

**Constraints:**

```
(mass balance)   Σ_j x_{ji}  −  Σ_j x_{ij}  =  d_i − s_i      for every country i
(capacity)       0  ≤  x_{ij}  ≤  u_{ij}                      for every edge (i, j)
```

where

- `s_i` = supply (production + carry-in) of country *i*
- `d_i` = demand (consumption) of country *i*
- `c_{ij}` = unit transport cost on edge *i → j*
- `u_{ij}` = capacity of edge *i → j*

The right-hand side `d_i − s_i` is exactly NetworkX's `demand` node
attribute: positive for net importers, negative for net exporters. If
global supply ≠ global demand (they rarely match exactly in FAOSTAT),
the larger side is rescaled so the LP has a feasible solution — see
`src.preprocessing.estimate_demand` and
`src.model._rebalance`.

Full derivation, unit conventions and assumptions are in
[`docs/methodology.md`](docs/methodology.md).

---

## Repository layout

```
allfed-fertilizer-min-cost-flow/
├── .github/workflows/       # CI: automated testing + linting
│   ├── testing.yml
│   └── lint.yml
├── src/
│   ├── model.py             # FertilizerMCF class + MCFResult dataclass
│   ├── preprocessing.py     # FAOSTAT loaders + shock helpers
│   ├── postprocessing.py    # summaries + comparisons + CSV export
│   ├── utils.py             # plotly / matplotlib helpers
│   └── README.md
├── scripts/
│   ├── toy_example.ipynb    # 4-country demo, no external data
│   ├── real_data.ipynb      # full FAOSTAT pipeline with a shock
│   └── README.md
├── data/
│   └── README.md            # download instructions for FAOSTAT CSVs
├── results/                 # output of save_result(...) (git-ignored)
│   └── README.md
├── tests/
│   ├── test_mcf.py          # pytest suite: invariants + edge cases
│   └── README.md
├── docs/
│   ├── methodology.md       # equations + references
│   └── README.md
├── pyproject.toml / setup.py
├── requirements.txt
├── environment.yml          # conda environment
├── .flake8                  # lint config
├── .gitignore / LICENSE
└── README.md
```

---

## Running the tests

```bash
pytest
```

---

## License

Apache-2.0 (same as the RAS companion repo).
