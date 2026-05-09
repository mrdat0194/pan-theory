import numpy as np
from scipy.stats import norm
from inventory_models import (
    calc_normal_prob, calc_normal_quantile, calc_poisson_prob,
    calc_critical_ratio, calc_critical_ratio_cs_ce,
    calc_optimal_order_quantity, calc_optimal_order_uniform,
    calc_expected_units_short, calc_expected_profit,
    calc_discrete_expected_demand, calc_discrete_expected_units_short,
    calc_periodic_cycle_stock, calc_periodic_safety_stock, calc_order_upto_level,
    calc_eoq, calc_eoq_planned_backorders,
    calc_safety_stock_continuous, calc_reorder_point,
    calc_item_fill_rate, calc_implied_stockout_cost_b1, calc_implied_csl_from_safety_stock,
    calc_pipeline_inventory_cost, calc_mean_demand_leadtime, calc_std_demand_leadtime,
    calc_carrier_savings, calc_breakeven_holding_charge
)


# ===========================================================================
# SHEET 1 PROBLEMS
# ===========================================================================

def run_sheet1_examples():
    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Bagels (Normal Distribution) ---")
    # mean=250, std=75
    prob_over_350 = calc_normal_prob(250, 75, 350, cdf=False)
    print(f"  Q1 P(demand > 350) = {prob_over_350:.4f}  [answer: 0.0912]")
    stock_95 = calc_normal_quantile(250, 75, 0.95)
    print(f"  Q2 Stock for 5% stockout risk = {stock_95:.2f}  [answer: 373.36]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Wilson Chew Toys & Dexter Delight (Normal, Newsvendor) ---")
    # SKU #87990_A Wilson: cost=5.95, price=9.99, mean=625, std=225
    cr_wilson = calc_critical_ratio(9.99, 5.95, 0)
    q_wilson = calc_optimal_order_quantity(625, 225, cr_wilson)
    prob_at_mean_wilson = 0.5  # P(D > mean) = 0.5 for any symmetric distribution
    print(f"  Wilson CR={cr_wilson:.4f}, Q*={q_wilson:.2f}")
    print(f"  P(demand > 625) = {prob_at_mean_wilson} [answer: 0.5, symmetric dist]")

    # SKU #333_99_J_4 Dexter: cost=4.25, price=8.00, mean=630, std=50
    cr_dexter = calc_critical_ratio(8.00, 4.25, 0)
    q_dexter = calc_optimal_order_quantity(630, 50, cr_dexter)
    prob_at_mean_dexter = 0.5
    print(f"  Dexter CR={cr_dexter:.4f}, Q*={q_dexter:.2f}")
    print(f"  P(demand > 630) = {prob_at_mean_dexter} [answer: 0.5, symmetric dist]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Hank's Slow Mover Rule (Poisson) ---")
    # Hank's rule: stock = mean + 1 unit.
    # P(D > stock) = 1 - P(D <= stock) = 1 - poisson.cdf(stock, mean)
    # Griffin's Dog Bed: mean=3, stock level = 4 (=3+1)
    #   P(demand EXCEEDS 4) = 1 - P(D<=4)  [sheet answer: 0.1847]
    prob_dog_bed = 1 - calc_poisson_prob(3, 4)   # P(D<=4), then subtract
    print(f"  Griffin Dog Bed P(D>4) = {prob_dog_bed:.4f}  [answer: 0.1847]")
    # Cody Chewables: mean=6, stock level = 7 (=6+1)
    #   P(demand EXCEEDS 7) = 1 - P(D<=7)  [sheet answer: 0.2560]
    prob_chewables = 1 - calc_poisson_prob(6, 7)
    print(f"  Cody Chewables P(D>7) = {prob_chewables:.4f}  [answer: 0.2560]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: NFL Replica Jersey (Newsvendor + Penalty) ---")
    # mean=32000, std=11000, cost=10.90, price=24, salvage=7, penalty=100
    # Part 1: No penalty — Stockout prob at Q=mean = 0.5
    prob_so_at_mean = 0.5
    print(f"  P(stockout) if order mean = {prob_so_at_mean} [answer: 0.5]")
    # Part 2: Base newsvendor (no penalty): cost=10.90, price=24, salvage=7
    cr_jersey_base = calc_critical_ratio(24, 10.90, 7)
    q_jersey_base = calc_optimal_order_quantity(32000, 11000, cr_jersey_base)
    print(f"  Base CR={cr_jersey_base:.4f}, Q*={q_jersey_base:.2f}  [answer: 40148.64]")
    # Part 3: With penalty $100/unit short -> CR and Q are higher
    cr_jersey = calc_critical_ratio(24, 10.90, 7, 100)
    q_jersey = calc_optimal_order_quantity(32000, 11000, cr_jersey)
    print(f"  With penalty $100: CR={cr_jersey:.4f}, Q*={q_jersey:.2f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Fenway Park Beer (Newsvendor) ---")
    # mean=750, std=175, cost=0.75, price=7.50, salvage=0
    cr_beer = calc_critical_ratio(7.50, 0.75, 0)
    q_beer = calc_optimal_order_quantity(750, 175, cr_beer)
    print(f"  CR={cr_beer:.4f}, Q*={q_beer:.2f}  [answer: 974.27]")
    z_beer = norm.ppf(cr_beer)
    eus_beer = calc_expected_units_short(175, z_beer)
    print(f"  E[Units Short] = {eus_beer:.2f}  [answer: ~8.3]")
    n_games_out = 80 * (1 - cr_beer)
    print(f"  Games run out (80 games) = {n_games_out:.1f}  [answer: 8.0]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Kendall Water Bottles (Newsvendor) ---")
    # mean=400, std=100
    # cost=0.55+0.10=0.65 (purchase+chilling), price=1.00, salvage=0 (discarded)
    # Base: cost=0.55+0.10=0.65, price=1.00, salvage=0
    cr_water = calc_critical_ratio(1.00, 0.65, 0)
    q_water = calc_optimal_order_quantity(400, 100, cr_water)
    print(f"  Base: CR={cr_water:.4f}, Q*={q_water:.2f}  [answer: 361.47]")

    # With penalty=$0.50/lost sale:
    # ce = c - g = 0.65 - 0 = 0.65 (cost of excess per unit)
    # cs = p - c + B = 1.00 - 0.65 + 0.50 = 0.85 (cost of shortage per unit)
    # CR = cs / (cs + ce) = 0.85 / 1.50 = 0.5667  => Q*=416.79
    # Note: sheet answer 472.15 may use cs=p-g+B=1.00+0.50=1.50 with ce=0.65 differently
    cr_water_p = calc_critical_ratio(1.00, 0.65, 0, 0.50)
    q_water_p = calc_optimal_order_quantity(400, 100, cr_water_p)
    print(f"  With penalty $0.50: CR={cr_water_p:.4f}, Q*={q_water_p:.2f}")
    # Alternative: if penalty applies as lost gross margin B=p=1.00, CE=0.65, CS=1.50
    cr_water_p2 = calc_critical_ratio_cs_ce(cs=1.50, ce=0.65)
    q_water_p2 = calc_optimal_order_quantity(400, 100, cr_water_p2)
    print(f"  Alt penalty interpretation: CR={cr_water_p2:.4f}, Q*={q_water_p2:.2f}  [answer: 472.15]")

    # Plastic labels: cost=0.55+0.10+0.10=0.75, salvage=0.65, price=1.00, no extra penalty
    # cs=p-c=1.00-0.75=0.25+salvage effect; use: CR=(p-c)/(p-g)=(0.25)/(0.35)=0.714
    cr_water_pl = calc_critical_ratio(1.00, 0.75, 0.65)
    q_water_pl = calc_optimal_order_quantity(400, 100, cr_water_pl)
    print(f"  Plastic labels: CR={cr_water_pl:.4f}, Q*={q_water_pl:.2f}  [answer: 416.79]")

    # Expected profit at Q*=361 (base case)
    z_base = norm.ppf(cr_water)
    eus_base = calc_expected_units_short(100, z_base)
    profit_base = calc_expected_profit(1.00, 0.65, 0, 400, q_water, eus_base)
    print(f"  Expected daily profit = ${profit_base:.2f}  [answer: ~$165]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Banapples (Discrete Newsvendor) ---")
    # Discrete demand: 0,50,100,150,200,250,300 with given probabilities
    # p=1.50, c=0.30, g=-0.15 (negative salvage = disposal cost), B=0
    demands = [0, 50, 100, 150, 200, 250, 300]
    probs   = [0.04, 0.15, 0.30, 0.25, 0.11, 0.10, 0.05]
    price_b, cost_b, salvage_b = 1.50, 0.30, -0.15
    e_demand = calc_discrete_expected_demand(demands, probs)
    print(f"  E[Demand] = {e_demand:.1f}  [answer: 137]")
    # At Q=150 from prior analysis
    eus_150 = calc_discrete_expected_units_short(150, demands, probs)
    print(f"  E[US] at Q=150 = {eus_150:.1f}  [answer: 23]")
    profit_150 = (price_b - salvage_b)*e_demand - (cost_b - salvage_b)*150 - (price_b - salvage_b)*eus_150
    print(f"  E[Profit] at Q=150 = ${profit_150:.2f}  [answer: ~$120.6/day, $600/week]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: EOQ with Planned Backorders ---")
    # co=$2 (overstocking cost), cu=$8 (understocking/backorder cost), QPBO*=100
    # Q* = QPBO* * sqrt(CR) where CR = cu/(cu+co) = 8/10 = 0.8
    # Q* = 100 * sqrt(0.8) = 89.44 => round to 90
    cr_bo = 8 / (8 + 2)
    q_star = 100 * np.sqrt(cr_bo)
    print(f"  CR={cr_bo:.2f}, Q* = {q_star:.2f}, rounded = {round(q_star/10)*10}  [answer: 90]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: MIT Book Store (Newsvendor) ---")
    # cost=100, price=160, salvage=60
    # Normal: mean=220, std=30  |  Uniform: 0-300
    cr_mit = calc_critical_ratio(160, 100, 60)
    q_mit_norm = calc_optimal_order_quantity(220, 30, cr_mit)
    print(f"  Normal: CR={cr_mit:.4f}, Q*={q_mit_norm:.2f}")
    q_mit_uni = calc_optimal_order_uniform(0, 300, cr_mit)
    print(f"  Uniform [0,300]: Q*={q_mit_uni:.2f}")
    # P(excess) < 0.20 => Q such that P(D<Q) > 0.80 => Q = min + 0.80*(max-min)
    q_mit_uni_excess = calc_optimal_order_uniform(0, 300, 0.80)
    print(f"  Uniform: Q for P(excess)<20% = {q_mit_uni_excess:.2f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Firefighter MRE's (Newsvendor) ---")
    mre_mean, mre_std = 3500, 275
    # Part 1: P(SO)=0.001
    q_mre_1 = calc_optimal_order_quantity(mre_mean, mre_std, 0.999)
    print(f"  P(SO)=0.001 -> Q={q_mre_1:.2f}  [answer: ~4349.8]")
    # Part 2: cost=2.75, price(helicopter)=15.00, salvage(fly-back)=-1.00
    cr_mre_2 = calc_critical_ratio(15.00, 2.75, -1.00)
    q_mre_2 = calc_optimal_order_quantity(mre_mean, mre_std, cr_mre_2)
    z_mre_2 = norm.ppf(cr_mre_2)
    eus_mre_2 = calc_expected_units_short(mre_std, z_mre_2)
    print(f"  With costs: CR={cr_mre_2:.4f}, Q={q_mre_2:.2f}")
    print(f"  P(SO)={1-cr_mre_2:.4f}, E[US]={eus_mre_2:.2f}")
    # Part 5: new demand mean=3800, std=735, salvage=2.00
    cr_mre_3 = calc_critical_ratio(15.00, 2.75, 2.00)
    q_mre_3 = calc_optimal_order_quantity(3800, 735, cr_mre_3)
    print(f"  New demand (mean=3800,std=735): Q={q_mre_3:.2f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 1: Lapiwi Logistics (Newsvendor) ---")
    # cost=$575/day, revenue=$1590/day, salvage=0, Normal(mean=15, std=3)
    cr_lapiwi = calc_critical_ratio(1590, 575, 0)
    q_lapiwi = calc_optimal_order_quantity(15, 3, cr_lapiwi)
    print(f"  CR={cr_lapiwi:.4f}, Optimal trucks={q_lapiwi:.2f}")


# ===========================================================================
# SHEET 2 PROBLEMS
# ===========================================================================

def run_sheet2_examples():
    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Pipeline Inventory (R-Taylor SKU#0172) ---")
    # Annual demand = 240 units/month * 12 months, item_cost=50, h=6%, LT=0.5 months
    demand_annual = 240 * 12
    lt_years = 0.5 / 12
    pipeline_cost = calc_pipeline_inventory_cost(50, 0.06, lt_years, demand_annual)
    print(f"  Annual Pipeline Cost = ${pipeline_cost:.2f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Item Fill Rate (IFR) & Implied Cost per Stockout ---")
    # D=13000/yr, RMSE=1316, S=1127, item_cost=250, h=10%, CSL=95%, LT=2 weeks
    d_ifr = 13000
    rmse_ifr = 1316
    s_ifr = 1127
    item_cost_ifr = 250
    h_ifr = item_cost_ifr * 0.10
    lt_ifr_yrs = 2 / 52
    q_ifr = calc_eoq(d_ifr, s_ifr, h_ifr)
    sigma_dl = rmse_ifr * np.sqrt(lt_ifr_yrs)
    z_ifr = norm.ppf(0.95)
    ifr = calc_item_fill_rate(sigma_dl, z_ifr, q_ifr)
    print(f"  EOQ={q_ifr:.2f}, sigma_DL={sigma_dl:.2f}, IFR={ifr:.4f}")
    b1 = calc_implied_stockout_cost_b1(0.95, d_ifr, h_ifr, sigma_dl, q_ifr)
    print(f"  Implied B1 (cost per stockout) = ${b1:.2f}  [hint: ~$5169]")
    # Implied CSL at SS=200 (if manager sets safety stock = 200 units)
    csl_200 = calc_implied_csl_from_safety_stock(200, sigma_dl)
    print(f"  Implied CSL at SS=200 = {csl_200:.4f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Uncle's Pharmacy / BetterBrain (Periodic Review) ---")
    # R=10 days, L=3 days, mean_D=20 units/day, std_D=5, CSL=95%
    r, l, d, std_d = 10, 3, 20, 5
    cs = calc_periodic_cycle_stock(d, r)
    ss = calc_periodic_safety_stock(std_d, r, l, 0.95)
    oul = calc_order_upto_level(d, r, l, ss)
    print(f"  Cycle Stock={cs:.2f}, Safety Stock={ss:.2f}, Order-Up-To={oul:.2f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Pumper's (s,Q) Continuous Review ---")
    # D=2400/yr, S=5, item_cost=10, h=25%, penalty B1=6/unit
    d_p, s_p, ic_p, h_rate_p = 2400, 5, 10, 0.25
    h_p = ic_p * h_rate_p
    eoq_p = calc_eoq(d_p, s_p, h_p)
    print(f"  EOQ={eoq_p:.2f} units")
    # With B1=6: optimal z satisfies norm.pdf(z)/norm.sf(z) ≈ h*Q/(B1*D)
    # Approximate: use CSL from ratio
    # For exact: CSL s.t. G(z) = h*Q / (B1*D) iteratively
    ratio = h_p * eoq_p / (6 * d_p)
    # G(z) = ratio, solve numerically
    from scipy.optimize import brentq
    loss_fn = lambda z: norm.pdf(z) - z * norm.sf(z) - ratio
    z_pump = brentq(loss_fn, -3, 3)
    csl_pump = norm.cdf(z_pump)
    sigma_dl_pump = ic_p / np.sqrt(d_p)  # Hint: placeholder; need actual sigma_D & LT
    print(f"  Hint: Optimal z={z_pump:.3f}, implied CSL={csl_pump:.4f}")
    print(f"  (sigma_DL needed from problem data to compute exact SS & reorder point)")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: ABC Ocean Carrier (Variable Lead Time) ---")
    # D=32 containers/wk, std_D=12, value=100000, h=22%, LT~N(4,1) weeks
    d_oc, std_oc, val_oc, h_oc = 32, 12, 100000, 0.22
    lt_mean_oc, lt_std_oc = 4, 1
    sigma_dl_oc = calc_std_demand_leadtime(d_oc, std_oc, lt_mean_oc, lt_std_oc)
    ss_oc = calc_safety_stock_continuous(sigma_dl_oc, 0.95)
    mu_dl_oc = calc_mean_demand_leadtime(d_oc, lt_mean_oc)
    s_oc = calc_reorder_point(mu_dl_oc, ss_oc)
    pipeline_oc = calc_pipeline_inventory_cost(val_oc, h_oc, lt_mean_oc, d_oc)
    print(f"  sigma_DL={sigma_dl_oc:.2f}, SS={ss_oc:.2f}, ROP s={s_oc:.2f}")
    print(f"  Pipeline cost = ${pipeline_oc:,.2f}/week")

    # Slow steaming: LT_mean=5, LT_std=0.5 — find discount needed
    sigma_dl_ss = calc_std_demand_leadtime(d_oc, std_oc, 5, 0.5)
    ss_slow = calc_safety_stock_continuous(sigma_dl_ss, 0.95)
    pipeline_slow = calc_pipeline_inventory_cost(val_oc, h_oc, 5, d_oc)
    extra_pipeline = (pipeline_slow - pipeline_oc) + (ss_slow - ss_oc) * val_oc * h_oc / 52
    print(f"  Slow steam: sigma_DL={sigma_dl_ss:.2f}, SS={ss_slow:.2f}")
    print(f"  Additional annual inventory cost = ${extra_pipeline:,.2f}")
    print(f"  Hint: Carrier must discount freight by at least this amount to be cost-neutral")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Loon Hardware vs Wilson Express (LTL, Variable LT) ---")
    # D=1.5 pallets/day, std_D=0.25, Wilson LT: mean=2 days, std=1 day, CSL=99%
    d_loon, std_loon = 1.5, 0.25
    mu_dl_loon = calc_mean_demand_leadtime(d_loon, 2)
    sigma_dl_loon = calc_std_demand_leadtime(d_loon, std_loon, 2, 1)
    ss_loon = calc_safety_stock_continuous(sigma_dl_loon, 0.99)
    s_loon = calc_reorder_point(mu_dl_loon, ss_loon)
    print(f"  Wilson Express: sigma_DL={sigma_dl_loon:.4f}, SS={ss_loon:.4f}, ROP={s_loon:.4f}")
    # Loon Hardware: LT constant=2 days, std_L=0
    sigma_dl_loon_const = calc_std_demand_leadtime(d_loon, std_loon, 2, 0)
    ss_loon_const = calc_safety_stock_continuous(sigma_dl_loon_const, 0.99)
    s_loon_const = calc_reorder_point(mu_dl_loon, ss_loon_const)
    print(f"  Loon Hardware: sigma_DL={sigma_dl_loon_const:.4f}, SS={ss_loon_const:.4f}, ROP={s_loon_const:.4f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Widget Delivery (Variable Lead Time) ---")
    # D=130,000/yr = 356/day, sigma_D=13,000/yr => daily sigma = 13000/sqrt(365)
    # LT ~ N(3, 2) days
    d_w = 130000 / 365
    std_w = 13000 / np.sqrt(365)
    mu_dl_w = calc_mean_demand_leadtime(d_w, 3)
    sigma_dl_w = calc_std_demand_leadtime(d_w, std_w, 3, 2)
    ss_w = calc_safety_stock_continuous(sigma_dl_w, 0.95)
    s_w = calc_reorder_point(mu_dl_w, ss_w)
    print(f"  mu_DL={mu_dl_w:.2f}, sigma_DL={sigma_dl_w:.2f}, SS={ss_w:.2f}, ROP={s_w:.2f}")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: Waterville Valley Chairs / SKU #A452 (TL vs LTL) ---")
    # D=5000/yr, S=90, item_cost=40, h=15%, LTL adds $2.60/unit
    eoq_tl = calc_eoq(5000, 90, 40 * 0.15)
    eoq_ltl = calc_eoq(5000, 90, (40 + 2.60) * 0.15)
    print(f"  EOQ (TL, base cost $40) = {eoq_tl:.2f}")
    print(f"  EOQ (LTL, effective cost $42.60) = {eoq_ltl:.2f}")
    print(f"  Hint: Compare total annual costs (holding + ordering + transport) to decide")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 2: SKU #001237 O'Yeah (Newsvendor) ---")
    # price=225, cost=175, salvage=225*0.25=56.25, mean=2500, std=400
    cr_oy = calc_critical_ratio(225, 175, 225 * 0.25)
    q_oy = calc_optimal_order_quantity(2500, 400, cr_oy)
    z_oy = norm.ppf(cr_oy)
    eus_oy = calc_expected_units_short(400, z_oy)
    print(f"  CR={cr_oy:.4f}, Q*={q_oy:.0f}, E[US]={eus_oy:.2f}")


# ===========================================================================
# SHEET 3 PROBLEMS
# ===========================================================================

def run_sheet3_examples():
    # -----------------------------------------------------------------------
    print("\n--- Sheet 3: EOQ Problem ---")
    # D=250/month=3000/yr, holding cost=145*0.19, order cost=144
    annual_demand = 250 * 12
    h_cost = 145 * 0.19
    eoq = calc_eoq(annual_demand, 144, h_cost)
    eoq_rounded = round(eoq / 10) * 10
    print(f"  EOQ={eoq:.2f}, rounded={eoq_rounded}  [answer: ~182]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 3: Carrier Selection (Lane Cost Comparison) ---")
    # value_per_load=56900, h=24%, loads/yr=250
    # Carrier 1: $1060/load, 4 days | Carrier 2: $1200/load, 2 days
    result = calc_carrier_savings(
        value_per_load=56900, holding_rate=0.24,
        c1_cost_per_load=1060, c1_days=4,
        c2_cost_per_load=1200, c2_days=2,
        loads_per_year=250
    )
    print(f"  Carrier 1 total = ${result['C1_Total']:,.2f}")
    print(f"  Carrier 2 total = ${result['C2_Total']:,.2f}")
    print(f"  Savings switching to Carrier 2 = ${result['Savings']:,.2f}")

    # Breakeven holding charge
    h_break = calc_breakeven_holding_charge(1060, 4, 1200, 2, 56900, 250)
    print(f"  Breakeven holding charge h* = {h_break:.4f} ({h_break*100:.2f}%)")

    # If value changes to $42,675 per load
    result2 = calc_carrier_savings(
        value_per_load=42675, holding_rate=0.24,
        c1_cost_per_load=1060, c1_days=4,
        c2_cost_per_load=1200, c2_days=2,
        loads_per_year=250
    )
    print(f"  At $42,675/load: Savings = ${result2['Savings']:,.2f}  [negative = C1 cheaper]")

    # -----------------------------------------------------------------------
    print("\n--- Sheet 3: Maria Fast Fashion Group (SKU #J65991X) ---")
    # Normal(mean=59, std=23)
    prob_over_72 = calc_normal_prob(59, 23, 72, cdf=False)
    print(f"  P(demand > 72) = {prob_over_72:.4f}")
    q_14 = calc_normal_quantile(59, 23, 1 - 0.14)
    print(f"  Stock level for 14% stockout risk = {q_14:.2f}")


# ===========================================================================
# ORIGINAL FOAM ROLLER EXAMPLE (from lessonLearnt.py)
# ===========================================================================

def run_original_example():
    print("\n--- Original: Yes4All Premium Foam Roller (Periodic Review) ---")
    # Periodic review: R=5 days, L=4 days, D=50/day, std=20, h=1.6, S=4000, CSL=95%
    demand, std_d = 50, 20
    review_period, lead_time = 5, 4
    order_cost, holding_cost = 4000, 1.6
    csl = 0.95
    eoq = calc_eoq(demand=demand, order_cost=order_cost, holding_cost=holding_cost)
    cs = calc_periodic_cycle_stock(demand=demand, review_period=review_period)
    ss = calc_periodic_safety_stock(std=std_d, review_period=review_period,
                                    lead_time=lead_time, cr=csl)
    oul = calc_order_upto_level(demand=demand, review_period=review_period,
                                lead_time=lead_time, safety_stock=ss)
    print(f"  EOQ={eoq:.2f} pcs, Cycle Stock={cs:.2f}, Safety Stock={ss:.2f}, OUL={oul:.2f}")


if __name__ == "__main__":
    run_original_example()
    run_sheet1_examples()
    run_sheet2_examples()
    run_sheet3_examples()
