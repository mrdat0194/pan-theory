import numpy as np
from scipy.stats import norm, poisson
import math

# ==========================================
# 1. Statistical Distributions & Probabilities
# ==========================================

def calc_normal_prob(mean: float, std: float, x: float, cdf: bool = True) -> float:
    """
    Calculate the probability of demand.
    If cdf is True, returns P(X <= x).
    If cdf is False, returns P(X > x) = 1 - P(X <= x).
    """
    if cdf:
        return norm.cdf(x, mean, std)
    else:
        return 1 - norm.cdf(x, mean, std)

def calc_normal_quantile(mean: float, std: float, target_prob: float) -> float:
    """
    Calculate the value x at a specific probability (e.g., to have 95% probability of not running out).
    x = norm.ppf(target_prob, mean, std)
    """
    return norm.ppf(target_prob, mean, std)

def calc_poisson_prob(mean: float, x: int, cdf: bool = True) -> float:
    """
    Calculate Poisson probabilities for slow-moving items.
    If cdf is True, returns P(X <= x).
    If cdf is False, returns P(X = x) exactly.
    """
    if cdf:
        return poisson.cdf(x, mean)
    else:
        return poisson.pmf(x, mean)

def calc_discrete_expected_demand(demand_levels: list, probabilities: list) -> float:
    """
    Calculate expected demand from a discrete probability distribution.
    E[D] = sum(d_i * p_i)
    """
    return sum(d * p for d, p in zip(demand_levels, probabilities))

def calc_discrete_expected_units_short(q: float, demand_levels: list, probabilities: list) -> float:
    """
    Calculate expected units short for a discrete demand distribution.
    E[US] = sum(max(d - Q, 0) * P(demand = d))
    """
    return sum(max(d - q, 0) * p for d, p in zip(demand_levels, probabilities))

# ==========================================
# 2. Single-Period Inventory (Newsvendor Model)
# ==========================================

def calc_critical_ratio(price: float, cost: float, salvage: float, penalty: float = 0.0) -> float:
    """
    Calculate Critical Ratio (CR).
    CR = (Price - Cost + Penalty) / (Price - Salvage + Penalty)
    Note: For negative salvage (disposal cost), salvage should be negative.
    """
    numerator = price - cost + penalty
    denominator = price - salvage + penalty
    return numerator / denominator

def calc_critical_ratio_cs_ce(cs: float, ce: float) -> float:
    """
    Calculate Critical Ratio using cost of shortage and cost of excess.
    CR = cs / (cs + ce)
    where:
      cs = cost of understocking (lost margin + penalty)
      ce = cost of overstocking (cost - salvage)
    """
    return cs / (cs + ce)

def calc_optimal_order_quantity(mean: float, std: float, cr: float) -> float:
    """
    Calculate Optimal Order Quantity Q* for Normal distribution.
    Q* = mean + z * std, where z = norm.ppf(CR)
    """
    z = norm.ppf(cr)
    return mean + z * std

def calc_optimal_order_uniform(low: float, high: float, cr: float) -> float:
    """
    Calculate Optimal Order Quantity Q* for Uniform distribution.
    Q* = low + CR * (high - low)
    """
    return low + cr * (high - low)

def calc_expected_units_short(std: float, z: float) -> float:
    """
    Calculate Expected Units Short E[US] using the unit normal loss function G(z).
    E[US] = std * G(z)
    G(z) = norm.pdf(z) - z * (1 - norm.cdf(z))
    """
    g_z = norm.pdf(z) - z * (1 - norm.cdf(z))
    return std * g_z

def calc_expected_profit(price: float, cost: float, salvage: float, mean: float,
                         q: float, expected_units_short: float, penalty: float = 0.0) -> float:
    """
    Calculate Expected Profit.
    E[Profit] = (p - g)*E[D] - (c - g)*Q - (p - g + B)*E[US]
    """
    term1 = (price - salvage) * mean
    term2 = (cost - salvage) * q
    term3 = (price - salvage + penalty) * expected_units_short
    return term1 - term2 - term3

def calc_newsvendor_with_penalty(price: float, cost: float, salvage: float, penalty: float,
                                  mean: float, std: float) -> dict:
    """
    Full Newsvendor model with optional stockout penalty.
    Returns: Q*, CR, z, E[US], E[Profit]
    """
    cr = calc_critical_ratio(price, cost, salvage, penalty)
    q = calc_optimal_order_quantity(mean, std, cr)
    z = norm.ppf(cr)
    eus = calc_expected_units_short(std, z)
    profit = calc_expected_profit(price, cost, salvage, mean, q, eus, penalty)
    return {"CR": cr, "Q": q, "z": z, "E_US": eus, "E_Profit": profit}

# ==========================================
# 3. Continuous Review Policy (s, Q)
# ==========================================

def calc_eoq(demand: float, order_cost: float, holding_cost: float) -> float:
    """
    Calculate Economic Order Quantity (EOQ).
    EOQ = sqrt((2 * D * S) / H)
    """
    return np.sqrt((2 * demand * order_cost) / holding_cost)

def calc_eoq_planned_backorders(q_pbo: float, cost_overstocking: float,
                                 cost_understocking: float) -> float:
    """
    Convert EOQ with Planned Backorders (QPBO*) to EOQ without backorders (Q*).
    CR = cs / (cs + ce)
    Q* = QPBO* * sqrt(1 / CR) = QPBO* / sqrt(CR)
    or equivalently:
    Q* = QPBO* * sqrt((cs + ce) / cs)
    """
    cr = cost_understocking / (cost_understocking + cost_overstocking)
    return q_pbo * np.sqrt(1 / cr)

def calc_safety_stock_continuous(std_dl: float, cr: float) -> float:
    """
    Calculate Standard Safety Stock given the standard deviation of demand over lead time.
    SS = z * std_dl
    """
    z = norm.ppf(cr)
    return z * std_dl

def calc_reorder_point(mean_dl: float, safety_stock: float) -> float:
    """
    Calculate Reorder Point s.
    s = mean_dl + safety_stock
    """
    return mean_dl + safety_stock

def calc_item_fill_rate(std_dl: float, z: float, q: float) -> float:
    """
    Calculate Item Fill Rate (IFR) using the unit normal loss function.
    IFR = 1 - E[US] / Q = 1 - (sigma_DL * G(z)) / Q
    """
    eus = calc_expected_units_short(std_dl, z)
    return 1 - (eus / q)

def calc_implied_stockout_cost_b1(csl: float, demand: float, holding_cost_per_unit: float,
                                   std_dl: float, q: float) -> float:
    """
    Calculate B1 (Implied Cost per Stockout Event) given a Target CSL.
    Formula: k = sqrt(2 * ln( B1 * D / (ce * sigma_DL * Q * sqrt(2*pi)) ))
    Rearranged: B1 = exp(k^2/2) * (ce * sigma_DL * Q * sqrt(2*pi)) / D
    where ce = holding_cost_per_unit (cost per unit per year to hold)
    """
    k = norm.ppf(csl)
    b1 = np.exp(k**2 / 2) * (holding_cost_per_unit * std_dl * q * np.sqrt(2 * np.pi)) / demand
    return b1

def calc_implied_csl_from_safety_stock(safety_stock: float, std_dl: float) -> float:
    """
    Calculate implied CSL from a given safety stock level.
    SS = z * sigma_DL => z = SS / sigma_DL
    CSL = norm.cdf(z)
    """
    z = safety_stock / std_dl
    return norm.cdf(z)

# ==========================================
# 4. Periodic Review Policy
# ==========================================

def calc_periodic_cycle_stock(demand: float, review_period: float) -> float:
    """
    Calculate Average Cycle Stock for Periodic Review.
    Cycle Stock = (Demand * Review Period) / 2
    """
    return (demand * review_period) / 2

def calc_periodic_safety_stock(std: float, review_period: float, lead_time: float, cr: float) -> float:
    """
    Calculate Safety Stock for Periodic Review.
    SS = z * std * sqrt(R + L)
    """
    z = norm.ppf(cr)
    return z * std * np.sqrt(review_period + lead_time)

def calc_order_upto_level(demand: float, review_period: float, lead_time: float, safety_stock: float) -> float:
    """
    Calculate Order-Up-To Level.
    Order-Up-To = Demand * (R + L) + SS
    """
    return demand * (review_period + lead_time) + safety_stock

# ==========================================
# 5. Variable Lead Time (Hadley-Whitin Model)
# ==========================================

def calc_mean_demand_leadtime(mean_d: float, mean_l: float) -> float:
    """
    Calculate Expected Demand over Lead Time.
    mu_DL = mu_D * mu_L
    """
    return mean_d * mean_l

def calc_std_demand_leadtime(mean_d: float, std_d: float, mean_l: float, std_l: float) -> float:
    """
    Calculate Standard Deviation of Demand over Lead Time (Hadley-Whitin).
    sigma_DL = sqrt(mu_L * sigma_D^2 + mu_D^2 * sigma_L^2)
    Use std_l=0 for constant lead time: sigma_DL = std_D * sqrt(mu_L)
    """
    variance_dl = (mean_l * (std_d ** 2)) + ((mean_d ** 2) * (std_l ** 2))
    return np.sqrt(variance_dl)

def calc_pipeline_inventory_cost(item_cost: float, holding_rate: float,
                                  mean_l: float, demand: float) -> float:
    """
    Calculate Pipeline Inventory Cost.
    Cost = item_cost * holding_rate * LeadTime * Demand
    Note: Lead time and Demand must be in consistent time units.
    """
    return (item_cost * holding_rate) * mean_l * demand

# ==========================================
# 6. Transportation & Logistics
# ==========================================

def calc_carrier_savings(value_per_load: float, holding_rate: float,
                          c1_cost_per_load: float, c1_days: float,
                          c2_cost_per_load: float, c2_days: float,
                          loads_per_year: float, days_per_year: float = 365) -> dict:
    """
    Compare two carriers: pipeline inventory cost + transportation cost.
    Savings = C1_total - C2_total.
    Positive savings means C2 is cheaper.
    """
    c1_pipeline = calc_pipeline_inventory_cost(value_per_load, holding_rate,
                                                c1_days / days_per_year, loads_per_year)
    c2_pipeline = calc_pipeline_inventory_cost(value_per_load, holding_rate,
                                                c2_days / days_per_year, loads_per_year)
    c1_transport = c1_cost_per_load * loads_per_year
    c2_transport = c2_cost_per_load * loads_per_year
    c1_total = c1_pipeline + c1_transport
    c2_total = c2_pipeline + c2_transport
    savings = c1_total - c2_total
    return {
        "C1_Pipeline": c1_pipeline,
        "C2_Pipeline": c2_pipeline,
        "C1_Transport": c1_transport,
        "C2_Transport": c2_transport,
        "C1_Total": c1_total,
        "C2_Total": c2_total,
        "Savings": savings
    }

def calc_breakeven_holding_charge(c1_cost_per_load: float, c1_days: float,
                                   c2_cost_per_load: float, c2_days: float,
                                   value_per_load: float,
                                   loads_per_year: float,
                                   days_per_year: float = 365) -> float:
    """
    Find the holding charge h* where costs of two carriers are equal.
    C1_total = C2_total
    (value * h * c1_days/365 * loads) + (c1_cost * loads) = (value * h * c2_days/365 * loads) + (c2_cost * loads)
    Solving for h:
    h * value * loads * (c1_days - c2_days)/365 = (c2_cost - c1_cost) * loads
    h = (c2_cost - c1_cost) * days_per_year / (value * (c1_days - c2_days))
    """
    numerator = (c2_cost_per_load - c1_cost_per_load) * days_per_year
    denominator = value_per_load * (c1_days - c2_days)
    return numerator / denominator

def compare_tl_ltl_full(demand: float, item_cost: float, holding_rate: float,
                         order_cost: float,
                         tl_cost: float, tl_time: float, tl_capacity: float,
                         ltl_cost_per_item: float, ltl_time: float,
                         days_per_year: float = 365) -> dict:
    """
    Compare Truck Load (TL) vs Less-Than-Truck Load (LTL) total costs.
    TL Option A: ship full truckloads (Q = truck capacity)
    TL Option B: ship EOQ (use item_cost only for EOQ calc, then add transport cost)
    LTL Option C: ship EOQ with LTL (item_cost + ltl_cost_per_item for holding)
    """
    # Option A: Full TL
    a_item_cost = item_cost + tl_cost / tl_capacity
    a_cycle = a_item_cost * holding_rate * (tl_capacity / 2)
    a_pipeline = (demand / days_per_year) * tl_time * holding_rate * a_item_cost
    a_ordering = (demand / tl_capacity) * order_cost
    a_purchase = demand * a_item_cost
    a_total = a_cycle + a_pipeline + a_ordering + a_purchase

    # Option B: TL with EOQ (use base item_cost for EOQ calculation)
    eoq_b = calc_eoq(demand, order_cost, item_cost * holding_rate)
    b_item_cost = item_cost + tl_cost / eoq_b
    b_cycle = b_item_cost * holding_rate * (eoq_b / 2)
    b_pipeline = (demand / days_per_year) * tl_time * holding_rate * b_item_cost
    b_ordering = (demand / eoq_b) * order_cost
    b_purchase = demand * b_item_cost
    b_total = b_cycle + b_pipeline + b_ordering + b_purchase

    # Option C: LTL with EOQ (use base item_cost for EOQ)
    c_item_cost = item_cost + ltl_cost_per_item
    c_eoq = calc_eoq(demand, order_cost, item_cost * holding_rate)  # EOQ uses base cost
    c_cycle = c_item_cost * holding_rate * (c_eoq / 2)
    c_pipeline = (demand / days_per_year) * ltl_time * holding_rate * c_item_cost
    c_ordering = (demand / c_eoq) * order_cost
    c_purchase = demand * c_item_cost
    c_total = c_cycle + c_pipeline + c_ordering + c_purchase

    return {
        "Option_A_TL_Full": round(a_total),
        "Option_B_TL_EOQ": round(b_total),
        "Option_C_LTL_EOQ": round(c_total),
        "EOQ_Base": round(eoq_b)
    }
