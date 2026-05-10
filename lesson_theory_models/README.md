# `lesson_theory_models` — Inventory & Logistics Operations Research

A self-contained Python library implementing classical **Inventory Management** and **Logistics** models from first principles, validated against academic problem sets.

---

## 📁 Files

| File | Purpose |
|---|---|
| `inventory_models.py` | Core analytical functions (EOQ, Newsvendor, Safety Stock, etc.) |
| `example_problem.py` | Validated problems from Sheet 1–3, benchmarked against known answers |
| `gsheet_pipeline.py` | Functional-programming pipeline to pull problem data from Google Sheets |
| `requirements.txt` | All pip dependencies (core + recommended transition packages) |

---

## 🧮 Models Implemented

### 1. Statistical Distributions
- Normal CDF / Quantile (`calc_normal_prob`, `calc_normal_quantile`)
- Poisson PMF / CDF for slow-moving items (`calc_poisson_prob`)
- Discrete demand expected value and expected units short

### 2. Single-Period Inventory — Newsvendor Model
- Critical Ratio: `CR = (p - c + B) / (p - g + B)`
- Optimal Order Quantity Q* for Normal and Uniform demand
- Expected Units Short using the unit normal loss function `G(z)`
- Expected Profit computation with optional stockout penalty

### 3. Continuous Review Policy `(s, Q)`
- Economic Order Quantity (EOQ): `Q* = sqrt(2DS/H)`
- EOQ with Planned Backorders
- Safety Stock: `SS = z * σ_DL`
- Reorder Point: `s = μ_DL + SS`
- Item Fill Rate (IFR) and implied stockout cost B1
- Implied CSL from a given safety stock level

### 4. Periodic Review Policy `(R, S)`
- Cycle Stock, Safety Stock, and Order-Up-To Level
- Covers combined review + lead time horizon `(R + L)`

### 5. Variable Lead Time — Hadley-Whitin Model
- `μ_DL = μ_D × μ_L`
- `σ_DL = sqrt(μ_L × σ_D² + μ_D² × σ_L²)`
- Reduces to `σ_D × sqrt(L)` for constant lead time

### 6. Transportation & Logistics
- Carrier comparison: pipeline inventory cost + transport cost
- Breakeven holding charge between two carriers
- TL vs LTL full cost comparison (Cycle Stock + Pipeline + Ordering + Purchase)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all validated example problems
python example_problem.py
```

---

## 🔄 Transition to Specialized Libraries

This module is built on first principles using `numpy` and `scipy`.
The table below shows how each model area maps to production-grade packages:

| This module covers... | Equivalent in specialized packages |
|---|---|
| EOQ, Newsvendor, Safety Stock | [`stockpyl`](https://stockpyl.readthedocs.io/) — drop-in academic inventory models |
| ABC/XYZ, large SKU portfolios | [`supplychainpy`](http://supplychainpy.org/) — practical inventory analytics |
| Carrier selection, TL vs LTL | [`pulp`](https://coin-or.github.io/pulp/) — LP-based transportation models |
| Complex routing / VRP | [`ortools`](https://developers.google.com/optimization) — Google OR-Tools |
| Policy simulation | [`simpy`](https://simpy.readthedocs.io/) — discrete-event simulation |

### Stockpyl Transition Example

```python
# --- Current (inventory_models.py) ---
from inventory_models import calc_eoq, calc_safety_stock_continuous
eoq = calc_eoq(demand=13000, order_cost=1127, holding_cost=25)

# --- Using Stockpyl ---
from stockpyl.eoq import economic_order_quantity
Q, cost = economic_order_quantity(holding_cost=25, fixed_cost=1127, demand_rate=13000)
```

### PuLP Transition Example (Carrier / Transportation LP)

```python
# --- Current (inventory_models.py) ---
from inventory_models import calc_carrier_savings
result = calc_carrier_savings(value_per_load=56900, holding_rate=0.24, ...)

# --- Using PuLP for multi-lane, multi-carrier optimization ---
import pulp
prob = pulp.LpProblem("carrier_selection", pulp.LpMinimize)
# Define decision variables, constraints, and objective → solves at scale
```

---

## 📦 Dependencies

Install everything needed:

```bash
pip install -r requirements.txt
```

| Package | Role |
|---|---|
| `numpy` | Array math, square roots, statistical computation |
| `scipy` | Normal/Poisson distributions, loss functions, root-finding |
| `polars` | Fast DataFrame processing in `gsheet_pipeline.py` |
| `gspread` | Google Sheets API read/write |
| `google-auth` | OAuth2 authentication for Google APIs |
| `stockpyl` | *(Transition)* Formal academic inventory optimization |
| `pulp` | *(Transition)* Linear programming for transportation models |
| `matplotlib` | *(Optional)* Plotting inventory levels, cost curves |

---

## 📚 Theory References

- **Newsvendor Model** — Perakis & Roels (2008); Cachon & Terwiesch *Matching Supply with Demand*
- **EOQ / (s,Q) Continuous Review** — Harris (1913); Silver, Pyke & Thomas *Inventory Management*
- **Hadley-Whitin Variable Lead Time** — Hadley & Whitin (1963) *Analysis of Inventory Systems*
- **Periodic Review (R,S)** — Zipkin *Foundations of Inventory Management* (2000)
- **TL vs LTL / Carrier Selection** — Chopra & Meindl *Supply Chain Management* (6th ed.)
