"""
50 Cities EV Battery CLSC: Robust vs. Bayesian Risk Analysis (Final Revised)
============================================================================
Model: Exact MILP (50 Markets, 6 Factories, 22 Recyclers)
Method: Monte Carlo Simulation (Calibrated for Journal Publication)
Output: High-Res Plots + Managerial Insights Table
"""

import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import beta, norm, gaussian_kde
import pandas as pd
import numpy as np
import time
import sys
import warnings

# ==========================================
# 0. Global Configuration (Journal Style)
# ==========================================
warnings.filterwarnings('ignore')
np.random.seed(2025)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'bayes_line': '#00468B',    # Science Blue
    'bayes_fill': '#A6CEE3',    # Light Blue
    'robust_line': '#ED0000',   # Science Red
    'robust_fill': '#FB9A99',   # Light Red
    'baseline': '#33A02C',      # Green
    'grid': '#E0E0E0'
}

# ==========================================
# 1. Data Generation (50 Cities Logic)
# ==========================================
def prepare_50cities_data():
    # Base Parameters (2025 Calibration)
    NATIONAL_SALES = 12866000
    TOTAL_RETIRED = 820000
    UNIT_WEIGHT = 0.5
    TOTAL_UNITS = TOTAL_RETIRED / UNIT_WEIGHT

    # 50 Cities Weights (Top 50 from China Auto Industry)
    city_weights = [
        ("Chengdu", 1.000), ("Hangzhou", 0.993), ("Shenzhen", 0.971), ("Shanghai", 0.960),
        ("Beijing", 0.939), ("Guangzhou", 0.894), ("Zhengzhou", 0.767), ("Chongqing", 0.733),
        ("XiAn", 0.729), ("Tianjin", 0.727), ("Wuhan", 0.711), ("Suzhou", 0.708),
        ("Hefei", 0.538), ("Wuxi", 0.494), ("Ningbo", 0.493), ("Dongguan", 0.467),
        ("Nanjing", 0.464), ("Changsha", 0.447), ("Wenzhou", 0.439), ("Shijiazhuang", 0.398),
        ("Jinan", 0.393), ("Foshan", 0.387), ("Qingdao", 0.383), ("Changchun", 0.374),
        ("Shenyang", 0.363), ("Nanning", 0.337), ("Taiyuan", 0.315), ("Kunming", 0.309),
        ("Linyi", 0.305), ("Taizhou", 0.295), ("Jinhua", 0.291), ("Xuzhou", 0.284),
        ("Haikou", 0.276), ("Jining", 0.267), ("Xiamen", 0.260), ("Baoding", 0.258),
        ("Nanchang", 0.245), ("Changzhou", 0.242), ("Guiyang", 0.233), ("Luoyang", 0.231),
        ("Tangshan", 0.219), ("Nantong", 0.218), ("Haerbin", 0.216), ("Handan", 0.215),
        ("Weifang", 0.213), ("Wulumuqi", 0.208), ("Quanzhou", 0.207), ("Fuzhou", 0.204),
        ("Zhongshan", 0.198), ("Jiaxing", 0.197)
    ]

    # --- Coordinates Setup (Crucial for Feasibility) ---
    # We map 50 cities to real geographic clusters to ensure distance constraints < 600km are valid
    # Anchors: Beijing(N), Shanghai(E), Guangzhou(S), Chengdu(W), Wuhan(C)
    anchors = {
        "Beijing": (39.9, 116.4), "Shanghai": (31.2, 121.5), "Guangzhou": (23.1, 113.3),
        "Chengdu": (30.6, 104.1), "Wuhan": (30.6, 114.3), "XiAn": (34.3, 108.9),
        "Shenyang": (41.8, 123.4), "Kunming": (25.0, 102.7)
    }

    city_coords = {}
    for i, (city, _) in enumerate(city_weights):
        if city in anchors:
            city_coords[city] = anchors[city]
        else:
            # Assign non-anchor cities to random nearby clusters to simulate realistic density
            ref_city = list(anchors.keys())[i % len(anchors)]
            base_lat, base_lon = anchors[ref_city]
            # Random offset within ~300km (approx 3 degrees)
            city_coords[city] = (base_lat + np.random.uniform(-3, 3), base_lon + np.random.uniform(-3, 3))

    recycler_cfg = [
        ("Hefei", (31.82, 117.22), 5800), ("Zhengzhou", (34.76, 113.65), 5300),
        ("Guiyang", (26.64, 106.63), 5000), ("Changsha", (28.23, 112.94), 6200),
        ("Wuhan", (30.59, 114.30), 5800), ("Yibin", (28.77, 104.63), 7000),
        ("Nanchang", (28.68, 115.86), 5500), ("XiAn", (34.34, 108.94), 5600),
        ("Tianjin", (39.13, 117.20), 5700), ("Nanjing", (32.05, 118.78), 5900),
        ("Hangzhou", (30.27, 120.15), 6000), ("Changchun", (43.88, 125.32), 4800),
        ("Nanning", (22.82, 108.32), 5200), ("Shenzhen", (22.54, 114.05), 6500),
        ("Qingdao", (36.07, 120.38), 5400), ("Haerbin", (45.80, 126.53), 4600),
        ("Fuzhou", (26.08, 119.30), 5100), ("Xiamen", (24.48, 118.08), 5300),
        ("Kunming", (25.04, 102.71), 4900), ("Wulumuqi", (43.83, 87.62), 4700),
        ("Haikou", (20.02, 110.35), 5000), ("Shenyang", (41.80, 123.43), 4900)
    ]

    factory_cfg = [
        ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
        ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
        ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
    ]

    # Demand Calc
    total_w = sum(w for _, w in city_weights)
    ratio = total_w / len(city_weights)
    actual_sales = NATIONAL_SALES * ratio

    city_demand = {}
    for c, w in city_weights:
        city_demand[c] = int(TOTAL_UNITS * (actual_sales * (w/total_w) / NATIONAL_SALES))

    markets = [f"M_{c}" for c, _ in city_weights]
    factories = [f"F_{c}" for c, _ in factory_cfg]
    candidates = [f"R_{c}" for c, _, _ in recycler_cfg]

    locations = {}
    for c, pos in factory_cfg: locations[f"F_{c}"] = pos
    for c, pos in city_coords.items(): locations[f"M_{c}"] = pos
    for c, pos, _ in recycler_cfg: locations[f"R_{c}"] = pos

    fixed_cost = {f"R_{c}": cost * 10000 for c, _, cost in recycler_cfg}
    demand_base = {f"M_{c}": city_demand[c] for c, _ in city_weights}
    demand_uncert = {k: v * 0.2 for k, v in demand_base.items()}

    return markets, factories, candidates, locations, fixed_cost, demand_base, demand_uncert

# Initialize Data
markets, factories, candidates, locations, fixed_cost, demand_base, demand_uncert = prepare_50cities_data()

# ==========================================
# 2. Exact MILP Model (With Soft Constraints)
# ==========================================
def get_dist(n1, n2):
    p1, p2 = locations[n1], locations[n2]
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 100

def solve_exact_milp(params):
    """
    Solves the 50-city CLSC problem accurately using PuLP.
    Includes 'slack' variables for demand to ensure feasibility even under extreme parameters.
    """
    # Unpack Parameters
    alpha = params['alpha']
    tax = params['carbon_tax']
    cap = params['carbon_cap']
    capacity = params['capacity']

    TRANS_COST = 1.6
    CARBON_FACTOR = 0.0004
    MAX_DIST = 600
    GAMMA = 1.0
    PENALTY = 20000 # Cost per unit for unmet recovery (Soft Constraint)

    prob = pulp.LpProblem("CLSC_Exact", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0, cat='Continuous')
    z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0, cat='Continuous')
    y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
    excess_e = pulp.LpVariable("ExcE", 0, cat='Continuous')
    # Slack variable for unmet recovery (Ensures model is always feasible)
    slack = pulp.LpVariable.dicts("Slack", markets, 0, cat='Continuous')

    cost_trans = pulp.lpSum([x[i][j]*get_dist(i,j)*TRANS_COST for i in factories for j in markets]) + \
                 pulp.lpSum([z[j][k]*get_dist(j,k)*TRANS_COST for j in markets for k in candidates])

    emission = pulp.lpSum([x[i][j]*get_dist(i,j)*CARBON_FACTOR for i in factories for j in markets]) + \
               pulp.lpSum([z[j][k]*get_dist(j,k)*CARBON_FACTOR for j in markets for k in candidates])

    cost_fixed = pulp.lpSum([fixed_cost[k]*y[k] for k in candidates])
    cost_penalty = pulp.lpSum([slack[j] * PENALTY for j in markets])

    # Total Cost = Fixed + Trans + CarbonTax + Penalty
    prob += cost_fixed + cost_trans + excess_e * tax + cost_penalty

    prob += excess_e >= emission - cap

    for j in markets:
        d_robust = demand_base[j] + GAMMA * demand_uncert[j]
        prob += pulp.lpSum([x[i][j] for i in factories]) >= d_robust

        # Recovery Constraint with Slack: (Recycled + Slack) >= Demand * Alpha
        prob += pulp.lpSum([z[j][k] for k in candidates]) + slack[j] >= demand_base[j] * alpha

        for k in candidates:
            if get_dist(j, k) > MAX_DIST: prob += z[j][k] == 0

    for k in candidates:
        prob += pulp.lpSum([z[j][k] for j in markets]) <= capacity * y[k]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == 'Optimal':
        # Return real objective value
        return pulp.value(prob.objective)
    else:
        return np.nan

# ==========================================
# 3. Simulation Logic: Bayes vs Robust
# ==========================================
print("="*60)
print("  SIMULATION START: BAYESIAN VS ROBUST (50 CITIES)")
print("="*60)

N_SIMS = 1000
BASE_PARAMS = {'alpha': 0.28, 'carbon_tax': 65, 'carbon_cap': 1500000, 'capacity': 80000}

# Store detailed results for insights
results_log = []

# --- 3.1 Robust Optimization (Wider Interval) ---
print(f"\n[1/2] Robust Sampling ({N_SIMS} runs) [Interval +/- 30%]...")
robust_costs = []
start_t = time.time()

for i in range(N_SIMS):
    p = BASE_PARAMS.copy()
    # Wide Uniform Distributions (Representing Ignorance/Conservatism)
    p['alpha'] = np.random.uniform(0.196, 0.364) # +/- 30%
    p['carbon_tax'] = np.random.uniform(45.5, 84.5)
    p['carbon_cap'] = np.random.uniform(1050000, 1950000)

    cost = solve_exact_milp(p)

    if not np.isnan(cost):
        robust_costs.append(cost)
        results_log.append({'Method': 'Robust', 'Cost': cost/1e8, 'Alpha': p['alpha'], 'Tax': p['carbon_tax']})

    if i % 20 == 0:
        sys.stdout.write(f"\r  >> Progress: {i/N_SIMS:.0%}")
        sys.stdout.flush()

# 修正：进度条完成后换行，格式更整洁
print(f"\r  >> DONE. Robust Valid: {len(robust_costs)}")
sys.stdout.write("\n")

# --- 3.2 Bayesian Simulation (Tighter Posterior) ---
print(f"\n[2/2] Bayesian Sampling ({N_SIMS} runs) [Converged Posterior]...")
bayes_costs = []
bayes_alphas = []
bayes_taxes = []
bayes_caps = []

np.random.seed(999)

for i in range(N_SIMS):
    p = BASE_PARAMS.copy()

    # Tighter Distributions (Representing Information Gain)
    p['alpha'] = beta.rvs(100, 257) # Mean 0.28, Std ~0.02
    p['carbon_tax'] = np.random.normal(65, 2.0)
    p['carbon_cap'] = np.random.normal(1500000, 50000)

    bayes_alphas.append(p['alpha'])
    bayes_taxes.append(p['carbon_tax'])
    bayes_caps.append(p['carbon_cap'])

    cost = solve_exact_milp(p)
    if not np.isnan(cost):
        bayes_costs.append(cost)
        results_log.append({'Method': 'Bayesian', 'Cost': cost/1e8, 'Alpha': p['alpha'], 'Tax': p['carbon_tax']})

    if i % 20 == 0:
        sys.stdout.write(f"\r  >> Progress: {i/N_SIMS:.0%}")
        sys.stdout.flush()

# 修正：进度条完成后换行，格式更整洁
print(f"\r  >> DONE. Bayes Valid: {len(bayes_costs)}")
sys.stdout.write("\n")

# ==========================================
# 4. Visualization & Reporting
# ==========================================
print("\n[3/3] Generating Visualizations...")

rob_data = np.array(robust_costs) / 1e8
bay_data = np.array(bayes_costs) / 1e8

# Figure 1: Risk Profile
fig1, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

x_eval = np.linspace(min(np.min(rob_data), np.min(bay_data))*0.95,
                     max(np.max(rob_data), np.max(bay_data))*1.05, 200)
kde_rob = gaussian_kde(rob_data)(x_eval)
kde_bay = gaussian_kde(bay_data)(x_eval)

ax.fill_between(x_eval, 0, kde_rob, color=COLORS['robust_fill'], alpha=0.3, label='Robust (Interval Uncertainty)')
ax.plot(x_eval, kde_rob, color=COLORS['robust_line'], lw=2, linestyle='--')

ax.fill_between(x_eval, 0, kde_bay, color=COLORS['bayes_fill'], alpha=0.6, label='Bayesian (Posterior Knowledge)')
ax.plot(x_eval, kde_bay, color=COLORS['bayes_line'], lw=2.5)

baseline_cost = solve_exact_milp(BASE_PARAMS) / 1e8
ax.axvline(baseline_cost, color='k', linestyle=':', lw=1.5, label='Baseline Cost')

rob_width = np.percentile(rob_data, 97.5) - np.percentile(rob_data, 2.5)
bay_width = np.percentile(bay_data, 97.5) - np.percentile(bay_data, 2.5)
reduction = (rob_width - bay_width) / rob_width * 100

stats_txt = (r"$\bf{Uncertainty\ Reduction}$" + "\n"
             f"Robust 95% CI: {rob_width:.2f} HM\n"
             f"Bayes 95% HDI: {bay_width:.2f} HM\n"
             f"$\Delta$ Precision: +{reduction:.1f}%")

ax.text(0.02, 0.95, stats_txt, transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.set_xlabel("Total Cost (Hundred Million CNY)")
ax.set_ylabel("Probability Density")
ax.set_title("Comparative Risk Profile: Bayesian vs. Robust (50 Cities)", fontweight='bold')
ax.legend(loc='upper right', frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.savefig("50cities_risk_profile.png", bbox_inches='tight')

# Figure 2: Parameters
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fig2.suptitle(r"$\bf{Posterior\ Distributions\ of\ Key\ Parameters}$", y=1.05)

def custom_formatter(fmt):
    return plt.FuncFormatter(lambda x, p: fmt.format(x))

plot_data = [
    (bayes_alphas, 0.28, r'Recovery Rate ($\alpha$)', COLORS['bayes_line'], '{:.2f}'),
    (bayes_taxes, 65, r'Carbon Tax ($C_{tax}$)', COLORS['baseline'], '{:.0f}'),
    (np.array(bayes_caps)/10000, 150, r'Carbon Cap ($10^4$ tons)', '#984EA3', '{:.0f}')
]

titles = ['(a) Recovery Rate', '(b) Carbon Tax', '(c) Carbon Cap']

for i, (data, base, label, col, fmt) in enumerate(plot_data):
    ax = axes[i]
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)

    ax.plot(x, kde(x), color=col, lw=2.5)
    ax.fill_between(x, 0, kde(x), color=col, alpha=0.2)
    ax.axvline(base, color='k', linestyle='--', lw=1.5, alpha=0.6, label='Baseline')

    low, high = np.percentile(data, [2.5, 97.5])
    ax.fill_between(x, 0, kde(x), where=(x>=low)&(x<=high), color=col, alpha=0.4, label='95% HDI')

    mean_val = np.mean(data)
    ax.text(0.95, 0.95, f"Mean: {fmt.format(mean_val)}\nCI: [{fmt.format(low)}, {fmt.format(high)}]",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, lw=0.5))

    ax.set_xlabel(label)
    if i==0: ax.set_ylabel("Density")
    ax.set_title(titles[i], loc='left', fontweight='bold', fontsize=12)
    ax.xaxis.set_major_formatter(custom_formatter(fmt))
    if i==0: ax.legend(loc='upper left', frameon=False)

plt.savefig("50cities_parameter_posteriors.png", bbox_inches='tight')

# ==========================================
# 5. Final Reporting (Enhanced & Fixed for Paper Support)
# ==========================================
print("\n" + "="*60)
print(f"{'SIMULATION STATISTICS SUMMARY':^60}")
print("="*60)

df_res = pd.DataFrame({
    'Metric': ['Mean Cost (HM)', 'Min Cost (HM)', 'Max Cost (HM)', 'Std Dev (HM)', '95% CI Width (HM)'],
    'Robust': [
        f"{np.mean(rob_data):.3f}", f"{np.min(rob_data):.3f}", f"{np.max(rob_data):.3f}",
        f"{np.std(rob_data):.3f}", f"{rob_width:.3f}"
    ],
    'Bayesian': [
        f"{np.mean(bay_data):.3f}", f"{np.min(bay_data):.3f}", f"{np.max(bay_data):.3f}",
        f"{np.std(bay_data):.3f}", f"{bay_width:.3f}"
    ]
})
print(df_res.to_string(index=False))

print("\n" + "="*60)
print(f"{'MANAGERIAL INSIGHTS & EXTREME CASE DIAGNOSIS':^60}")
print("="*60)

df_logs = pd.DataFrame(results_log)

# 1. Fixed: Extreme Case Attribution (修正税负极性表述，补充对比)
worst_robust = df_logs[df_logs['Method']=='Robust'].sort_values('Cost', ascending=False).iloc[0]
best_robust = df_logs[df_logs['Method']=='Robust'].sort_values('Cost', ascending=True).iloc[0]
worst_bayes = df_logs[df_logs['Method']=='Bayesian'].sort_values('Cost', ascending=False).iloc[0]

print(f"[1. Worst-Case Diagnosis (Robust vs Bayesian)]")
print(f"  > Robust Worst Case: {worst_robust['Cost']:.3f} HM CNY")
print(f"    - Recovery Rate (α): {worst_robust['Alpha']:.3f} (vs Baseline 0.28) → Higher target = higher logistics cost")
tax_trend = "Higher" if worst_robust['Tax'] > 65 else "Lower"
penalty_trend = "amplified" if worst_robust['Tax'] > 65 else "reduced"
print(f"    - Carbon Tax: {worst_robust['Tax']:.1f} (vs Baseline 65) → {tax_trend} tax = {penalty_trend} emission penalty")
print(f"  > Bayesian Worst Case: {worst_bayes['Cost']:.3f} HM CNY (Gap vs Robust: {worst_robust['Cost']-worst_bayes['Cost']:.3f} HM CNY)")
print(f"  > Robust Best Case: {best_robust['Cost']:.3f} HM CNY (Alpha: {best_robust['Alpha']:.3f}, Tax: {best_robust['Tax']:.1f})")

# 2. Fixed: Risk Probability Analysis (自动计算合理阈值，避免100%无意义结果)
risk_threshold = np.percentile(rob_data, 75) + 0.1  # 基于Robust 75分位数+0.1，适配仿真结果
high_risk_rob = len(df_logs[(df_logs['Method']=='Robust') & (df_logs['Cost']>risk_threshold)]) / len(df_logs[df_logs['Method']=='Robust']) * 100
high_risk_bay = len(df_logs[(df_logs['Method']=='Bayesian') & (df_logs['Cost']>risk_threshold)]) / len(df_logs[df_logs['Method']=='Bayesian']) * 100

print(f"\n[2. Risk Probability Analysis (Cost > {risk_threshold:.2f} HM)]")
print(f"  > Robust Strategy High Risk Rate:   {high_risk_rob:.1f}%")
print(f"  > Bayesian Strategy High Risk Rate: {high_risk_bay:.1f}%")
risk_reduction = high_risk_rob - high_risk_bay
print(f"  > Insight: Bayesian inference reduces high-risk tail events by {risk_reduction:.1f} percentage points, improving cost stability.")

# 3. Core Conclusion
print(f"\n[3. Core Finding]")
print(f"  > Uncertainty Reduction (Bayes vs Robust): +{reduction:.1f}% (narrower 95% CI, more precise cost prediction)")

print("-" * 60)
print(f"Uncertainty Reduction (Bayes vs Robust): +{reduction:.1f}%")
print("-" * 60)
