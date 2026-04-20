# Methodology — min-cost-flow model of global fertilizer trade

This document derives the linear program the model solves, lists all
assumptions, and points at the code that implements each step.

## 1 · Symbols

| Symbol            | Meaning                                          | Unit |
|-------------------|--------------------------------------------------|------|
| `N`               | set of countries                                 |  –   |
| `E ⊂ N × N`       | set of admissible trade edges (i → j)            |  –   |
| `s_i`             | **supply** of country *i* (production + carry-in) | tonnes |
| `d_i`             | **demand** of country *i* (consumption)           | tonnes |
| `c_{ij}`          | **unit cost** on edge *i → j*                     | arb. (USD / tonne or unitless) |
| `u_{ij}`          | **capacity** of edge *i → j*                      | tonnes |
| `x_{ij}`          | **flow** on edge *i → j* (decision variable)      | tonnes |

## 2 · Optimisation problem

The LP is

$$
\min_{x \geq 0}\ \sum_{(i,j) \in E} c_{ij}\, x_{ij}  \tag{1}
$$

subject to

$$
\sum_{j:\, (j,i) \in E} x_{ji} - \sum_{j:\, (i,j) \in E} x_{ij} = d_i - s_i \qquad \forall\, i \in N  \tag{2}
$$

$$
0 \leq x_{ij} \leq u_{ij} \qquad \forall\, (i,j) \in E  \tag{3}
$$

Equation (2) is the classical mass-balance / conservation constraint.
`d_i − s_i` is positive for **importers** (demand node) and negative
for **exporters** (supply node). By convention the residual net-demand
must sum to zero across the whole network:

$$
\sum_{i \in N} (d_i - s_i) = 0  \tag{4}
$$

We call this the **balance condition**. Satisfying (4) is what lets a
feasible solution exist; if total demand exceeds total supply, no
allocation can cover all needs and the LP is infeasible (see §4).

## 3 · Inputs from FAOSTAT

Sources (`data/README.md`):

* `Inputs_FertilizersNutrient_E_All_Data.csv` (**production**, one row per country-year-nutrient)
* `Fertilizers_DetailedTradeMatrix_E_All_Data.csv` (**bilateral trade**, one row per reporter-partner-element-year)

For a single nutrient (N, P, or K) and a single year *t* we compute:

$$
\text{prod}_i = \text{FAOSTAT\;production}_i^{(t)}  \tag{5}
$$

$$
\text{imp}_i = \sum_j T_{ij}^{(t,\text{element}=5610)}, \qquad
\text{exp}_i = \sum_j T_{ij}^{(t,\text{element}=5910)}  \tag{6}
$$

and use (by default) the simple **consumption-from-flow** identity:

$$
s_i = \text{prod}_i + \text{imp}_i,  \tag{7}
$$

$$
d_i = \max\bigl(0,\;\text{prod}_i + \text{imp}_i - \text{exp}_i\bigr)  \tag{8}
$$

Because FAOSTAT supply and demand don't balance exactly (different
reporting dates, re-exports, measurement noise), we then rescale `d_i`
so that `Σ d_i = Σ s_i` — this is the normalise-to-supply step in
`estimate_demand(...)` (`src/preprocessing.py`).

## 4 · Balancing the LP

NetworkX's `min_cost_flow` requires that the sum of node demands is
**exactly zero**. After step (8) the sums can still differ slightly. We
apply (in `src.model._rebalance`):

1. Scale whichever side is larger (sources or sinks) so the magnitudes
   match.
2. Sweep any remaining floating-point residual onto the single
   largest-demand node.

If you prefer to let the LP fail loudly instead, construct the model
with `FertilizerMCF(..., balance="raise")`.

## 5 · Edge costs

`build_edges(...)` supports two cost models:

1. **`inverse_volume` (default):**

   $$
   c_{ij} \;=\; \frac{1}{T_{ij} + 1}  \tag{9}
   $$

   where `T_{ij}` is the historical trade volume on the route.
   Established routes are preferred because they carry a *lower* cost
   per unit flow. This is the formulation used in the original
   `min_cost_flow_npk.py` script.

2. **`unit`:** `c_{ij} = 1` for every edge. The optimiser then picks
   any feasible flow; useful as a sanity-check or when you just want
   max-flow behaviour with positive weights.

If you have real per-tonne transport cost estimates (for instance from
FAOSTAT's *detailed* trade matrix via
`Value (1000 USD) / Quantity (tonne)`), plug them in yourself:

```python
edges = build_edges(trade, year=2020)
edges["cost"] = my_cost_per_tonne(edges)
```

## 6 · Edge capacities and shocks

$$
u_{ij} = T_{ij} \cdot \kappa  \tag{10}
$$

where `κ ∈ [0, ∞)` is the `capacity_factor` argument. `κ = 0.5` models
"50% of historical shipping capacity survives" on every route.
`apply_edge_shock(...)` lets you override specific `(i, j)` pairs, which
is how port/chokepoint disruptions are simulated.

Supply shocks act directly on `s_i`:

$$
s_i^\text{shocked} = s_i \cdot \phi_i,  \qquad \phi_i \in [0, 1]  \tag{11}
$$

with default `ϕ_i = 1` (no shock). See `apply_supply_shock(...)`.

## 7 · Outputs and metrics

After solving we report, per country:

$$
\text{imports}_i = \sum_j x_{ji},  \qquad
\text{exports}_i = \sum_j x_{ij}  \tag{12}
$$

$$
\text{avail}_i = s_i + \text{imports}_i - \text{exports}_i  \tag{13}
$$

$$
\text{coverage}_i = 100 \cdot \frac{\text{avail}_i}{d_i}  \tag{14}
$$

A coverage of 100% means the country receives exactly its demand; below
100% indicates unmet need (only possible when the LP is infeasible and
we clip to a feasible sub-problem, or when the country has genuinely no
connecting route).

## 8 · Assumptions & limitations

* **Single nutrient.** Runs are per-nutrient (N, P or K). NPK fertilizer
  blends aren't modelled separately.
* **Annual, non-time-dependent.** No inventories / stocks carry across
  years. Each year is its own independent LP.
* **Cost proxies.** Unless the user supplies real USD/t costs, the model
  uses `1 / (volume + 1)` as a proxy, which rewards established routes
  but has no physical meaning as a transport cost.
* **LP relaxation.** Flows are continuous; we don't model indivisible
  shipments or minimum-load constraints.
* **Balance trick.** If FAOSTAT supply exceeds demand we scale demand up
  (and vice versa) to make the LP feasible. This can nudge individual
  country demands slightly away from FAOSTAT.
* **Non-negative costs.** `networkx.min_cost_flow` requires
  `c_{ij} ≥ 0`. The constructor raises if you pass negative costs.

## 9 · Code map

| Step                              | File : function                       |
|-----------------------------------|---------------------------------------|
| Load production CSV               | `src/preprocessing.py : load_production` |
| Aggregate imports/exports         | `src/preprocessing.py : country_flows`   |
| Derive `s_i`, `d_i`               | `src/preprocessing.py : estimate_demand` |
| Build the edge list               | `src/preprocessing.py : build_edges`     |
| Apply supply or edge shocks       | `src/preprocessing.py : apply_*_shock`   |
| Construct the LP                  | `src/model.py : FertilizerMCF`           |
| Rebalance net demand to sum to 0  | `src/model.py : _rebalance`              |
| Solve (NetworkX)                  | `src/model.py : FertilizerMCF.solve`     |
| Country summary, comparisons      | `src/postprocessing.py`                  |
| Plots (Sankey / heatmap / bars)   | `src/utils.py`                           |

## 10 · References

* Ford & Fulkerson (1962). *Flows in Networks*. Princeton UP.
* Ahuja, Magnanti & Orlin (1993). *Network Flows*. Prentice Hall.
* NetworkX developers. *`networkx.algorithms.flow.min_cost_flow`*.
* FAOSTAT. *Fertilizers by Nutrient; Detailed Trade Matrix.*
  <https://www.fao.org/faostat/>
