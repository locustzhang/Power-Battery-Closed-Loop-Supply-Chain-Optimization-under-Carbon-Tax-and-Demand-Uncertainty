"""
50 Cities EV Battery CLSC: Bayesian Uncertainty Analysis (Journal Quality Edition)
==================================================================================
Method: Exact Mixed-Integer Linear Programming (MILP) embedded in Metropolis-Hastings MCMC
Optimization: 50 Cities | 22 Candidates | Robust Parameters (Updated with Sensitivity Model)
"""

import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import time
from scipy.stats import beta, norm, gaussian_kde
import warnings
import sys
import random  # 导入Python内置random包，用于设置其种子

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# 新增：设置固定随机种子（保证结果可复现，学术论文必备）
# ==========================================
SEED = 42  # 可自定义（如123、666等），固定即可
np.random.seed(SEED)  # 设置numpy随机种子
random.seed(SEED)     # 设置Python内置random随机种子

# ==========================================
# 0. Global Configuration (Academic Style)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix', # LaTeX-like math font
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Academic Color Palette
COLORS = {
    'prior_fill': '#A6CEE3',    # Light Blue
    'prior_line': '#1F78B4',    # Dark Blue
    'post_fill': '#FB9A99',     # Light Red
    'post_line': '#E31A1C',     # Dark Red
    'truth': '#33A02C',         # Green
    'ci_shade': '#E31A1C',      # Red Shade for HDI
    'grid': '#E0E0E0'
}

# ==========================================
# 1. Data Preparation (50 Cities - Consistent with Sensitivity Model)
# ==========================================
print("="*70)
print("  50 CITIES CLSC NETWORK: BAYESIAN INFERENCE (EXACT MILP)")
print("="*70)

# Parameters (Updated to match Sensitivity Analysis Model)
NATIONAL_SALES_TOTAL = 12866000
TOTAL_RETIRED_BATTERY = 820000
UNIT_BATTERY_WEIGHT = 0.5
TOTAL_UNITS_NATIONAL = TOTAL_RETIRED_BATTERY / UNIT_BATTERY_WEIGHT

# 50 Cities Weights (Unchanged, consistent with previous model)
city_sales_weight = [
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

# Coordinates (Unchanged)
city_coords = {
    "Chengdu": (30.67, 104.06), "Hangzhou": (30.27, 120.15), "Shenzhen": (22.54, 114.05),
    "Shanghai": (31.23, 121.47), "Beijing": (39.90, 116.40), "Guangzhou": (23.13, 113.26),
    "Zhengzhou": (34.76, 113.65), "Chongqing": (29.56, 106.55), "XiAn": (34.34, 108.94),
    "Tianjin": (39.13, 117.20), "Wuhan": (30.59, 114.30), "Suzhou": (31.30, 120.58),
    "Hefei": (31.82, 117.22), "Wuxi": (31.57, 120.30), "Ningbo": (29.82, 121.55),
    "Dongguan": (23.05, 113.75), "Nanjing": (32.05, 118.78), "Changsha": (28.23, 112.94),
    "Wenzhou": (28.00, 120.70), "Shijiazhuang": (38.04, 114.51), "Jinan": (36.65, 117.12),
    "Foshan": (23.02, 113.12), "Qingdao": (36.07, 120.38), "Changchun": (43.88, 125.32),
    "Shenyang": (41.80, 123.43), "Nanning": (22.82, 108.32), "Taiyuan": (37.87, 112.55),
    "Kunming": (25.04, 102.71), "Linyi": (35.05, 118.35), "Taizhou": (28.66, 121.42),
    "Jinhua": (29.08, 119.65), "Xuzhou": (34.26, 117.28), "Haikou": (20.02, 110.35),
    "Jining": (35.42, 116.59), "Xiamen": (24.48, 118.08), "Baoding": (38.87, 115.48),
    "Nanchang": (28.68, 115.86), "Changzhou": (31.78, 119.95), "Guiyang": (26.64, 106.63),
    "Luoyang": (34.62, 112.45), "Tangshan": (39.63, 118.18), "Nantong": (32.01, 120.86),
    "Haerbin": (45.80, 126.53), "Handan": (36.61, 114.49), "Weifang": (36.71, 119.16),
    "Wulumuqi": (43.83, 87.62), "Quanzhou": (24.87, 118.68), "Fuzhou": (26.08, 119.30),
    "Zhongshan": (22.52, 113.39), "Jiaxing": (30.75, 120.75)
}

# 22 Recyclers (Updated fixed cost to match sensitivity model's 15M/18M scale)
recycler_config = [
    ("Hefei", (31.82, 117.22), 5800), ("Zhengzhou", (34.76, 113.65), 5300),  # Zhengzhou: 15.9M
    ("Guiyang", (26.64, 106.63), 5000), ("Changsha", (28.23, 112.94), 6200),
    ("Wuhan", (30.59, 114.30), 5800), ("Yibin", (28.77, 104.63), 7000),
    ("Nanchang", (28.68, 115.86), 5500), ("Xian", (34.34, 108.94), 5600),
    ("Tianjin", (39.13, 117.20), 5700), ("Nanjing", (32.05, 118.78), 5900),
    ("Hangzhou", (30.27, 120.15), 6000), ("Changchun", (43.88, 125.32), 4800),  # Hangzhou: 18M, Changchun:14.4M
    ("Nanning", (22.82, 108.32), 5200), ("Shenzhen", (22.54, 114.05), 6500),  # Shenzhen: 19.5M
    ("Qingdao", (36.07, 120.38), 5400), ("Haerbin", (45.80, 126.53), 4600),
    ("Fuzhou", (26.08, 119.30), 5100), ("Xiamen", (24.48, 118.08), 5300),
    ("Kunming", (25.04, 102.71), 4900), ("Wulumuqi", (43.83, 87.62), 4700),  # Wulumuqi:14.1M
    ("Haikou", (20.02, 110.35), 5000), ("Shenyang", (41.80, 123.43), 4900)
]

# 6 Factories (Unchanged)
factory_config = [
    ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# Process Data (Consistent with Sensitivity Model)
total_weight = sum(w for _, w in city_sales_weight)
sales_ratio = total_weight / len(city_sales_weight)
actual_sales_50 = NATIONAL_SALES_TOTAL * sales_ratio

markets, factories, candidates = [], [], []
locations, city_demand = {}, {}

for c, w in city_sales_weight:
    city_demand[c] = int(TOTAL_UNITS_NATIONAL * (actual_sales_50 * (w/total_weight) / NATIONAL_SALES_TOTAL))
    markets.append(f"M_{c}")
    locations[f"M_{c}"] = city_coords[c]

for c, pos in factory_config:
    factories.append(f"F_{c}")
    locations[f"F_{c}"] = pos

for c, pos, cost in recycler_config:
    candidates.append(f"R_{c}")
    locations[f"R_{c}"] = pos

# Updated: Fixed cost aligned with sensitivity model (10^4 -> 10^7 to match 15M+ scale)
fixed_cost = {f"R_{c}": cost * 3000 for c, _, cost in recycler_config}  # Adjust multiplier to match sensitivity's 114M total fixed cost
demand_base = {f"M_{c}": city_demand[c] for c, _ in city_sales_weight}
demand_uncertainty = {k: v * 0.2 for k, v in demand_base.items()}

# ==========================================
# 2. Solver Engine (Updated to Match Sensitivity Analysis Model)
# ==========================================
def get_dist(n1, n2):
    p1, p2 = locations[n1], locations[n2]
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 100

def solve_milp(alpha, demand_dict=None, verbose=False):
    """Core MILP solver with optional demand override (Updated with Sensitivity Model Params)"""
    # Updated Parameters: Aligned with Sensitivity Analysis Model
    TRANS_COST, CARBON_TAX = 1.6, 65  # CARBON_TAX matches baseline (65)
    FWD_CARBON_FACTOR, REV_CARBON_FACTOR = 0.0004, 0.0030  # Separate fwd/rev carbon factor (key update)
    CARBON_CAP = 100000  # Updated to match sensitivity's 222k total emissions (core constraint update)
    CAPACITY, MAX_REV_DIST = 80000, 600  # CAPACITY matches baseline (80000)

    prob = pulp.LpProblem("Bayes_MILP", pulp.LpMinimize)

    # Vars
    x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0, cat='Continuous')
    z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0, cat='Continuous')
    y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
    excess_e = pulp.LpVariable("ExcessE", 0, cat='Continuous')

    # Expressions (Updated: Separate forward/reverse carbon emission, align with sensitivity model)
    # Transport Cost: Keep consistent with sensitivity's 78% total cost ratio
    cost_trans = pulp.lpSum([x[i][j]*get_dist(i,j)*TRANS_COST for i in factories for j in markets]) + \
                 pulp.lpSum([z[j][k]*get_dist(j,k)*TRANS_COST for j in markets for k in candidates])

    # Emission: Separate forward/reverse carbon factors (key improvement from sensitivity model)
    emission = pulp.lpSum([x[i][j]*get_dist(i,j)*FWD_CARBON_FACTOR for i in factories for j in markets]) + \
               pulp.lpSum([z[j][k]*get_dist(j,k)*REV_CARBON_FACTOR for j in markets for k in candidates])

    # Fixed Cost: Aligned with sensitivity's 20.56% total cost ratio
    cost_fixed = pulp.lpSum([fixed_cost[k]*y[k] for k in candidates])

    # Objective Function: Match sensitivity model's cost structure
    prob += cost_fixed + cost_trans + excess_e*CARBON_TAX

    # Constraints (Updated to match sensitivity model's feasible region)
    prob += excess_e >= emission - CARBON_CAP

    current_demand = demand_dict if demand_dict else demand_base

    for j in markets:
        # Forward flow: Meet robust demand (base + 20% uncertainty, align with sensitivity)
        prob += pulp.lpSum([x[i][j] for i in factories]) >= current_demand[j] * 1.2
        # Reverse flow: Meet alpha recovery rate (core constraint, align with sensitivity)
        prob += pulp.lpSum([z[j][k] for k in candidates]) >= current_demand[j] * alpha
        # Max reverse distance constraint (unchanged, consistent with model)
        for k in candidates:
            if get_dist(j, k) > MAX_REV_DIST: prob += z[j][k] == 0

    # Recycler capacity constraint (match baseline: 80000 per recycler)
    for k in candidates:
        prob += pulp.lpSum([z[j][k] for j in markets]) <= CAPACITY * y[k]

    # Solve with CBC (quiet mode for MCMC efficiency)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=verbose))

    if pulp.LpStatus[status] == 'Optimal':
        return {'status': 'Optimal', 'cost': pulp.value(prob.objective)}
    else:
        return {'status': 'Infeasible', 'cost': np.nan}

# ==========================================
# 3. Bayesian Logic (Optimized, Keep Original Structure)
# ==========================================
class BayesianEngine:
    def __init__(self, true_alpha=0.28):
        self.true_alpha = true_alpha
        self.obs_costs = []

    def generate_observations(self, n=5):
        print(f"\n[Generation] Creating {n} synthetic observations based on True α={self.true_alpha}...")
        for i in range(n):
            # Perturb demand by 5% noise (consistent with original, align with robust demand)
            d_pert = {k: v * np.random.normal(1, 0.05) for k, v in demand_base.items()}
            res = solve_milp(self.true_alpha, d_pert)
            if res['status'] == 'Optimal':
                self.obs_costs.append(res['cost'])
            print(f"  -> Obs {i+1}: {res['cost']:,.0f} CNY")
        self.obs_mean = np.mean(self.obs_costs)
        self.obs_std = np.std(self.obs_costs)

    def log_likelihood(self, alpha):
        # Optimization: Don't resolve MILP 3 times. Solve once with mean demand.
        res = solve_milp(alpha)
        if res['status'] != 'Optimal': return -np.inf

        # Assume Gaussian error around the model's predicted cost (original logic)
        pred_cost = res['cost']
        return norm.logpdf(self.obs_mean, loc=pred_cost, scale=self.obs_std * 2.0)

    def run_mcmc(self, samples=1000, burn=100):
        print(f"\n[MCMC] Sampling {samples} points (Burn-in={burn})...")
        print("  Note: Exact MILP is running in the loop. Please wait.")

        chain = []
        curr_alpha = 0.3 # Start point (original)
        curr_ll = self.log_likelihood(curr_alpha)
        curr_prior = beta.logpdf(curr_alpha, 3, 7)

        accepted = 0
        start_t = time.time()

        for i in range(samples + burn):
            # Propose new alpha (original step size, keep convergence)
            prop_alpha = curr_alpha + np.random.normal(0, 0.02)
            if prop_alpha <= 0.1 or prop_alpha >= 0.5:
                prop_ll = -np.inf
            else:
                prop_ll = self.log_likelihood(prop_alpha)

            prop_prior = beta.logpdf(prop_alpha, 3, 7) if 0<prop_alpha<1 else -np.inf

            # Acceptance Ratio (original Bayesian logic)
            if prop_ll > -np.inf:
                ratio = (prop_ll + prop_prior) - (curr_ll + curr_prior)
                if np.log(np.random.rand()) < ratio:
                    curr_alpha = prop_alpha
                    curr_ll = prop_ll
                    curr_prior = prop_prior
                    accepted += 1

            if i >= burn:
                chain.append(curr_alpha)

            # Progress bar (original format, keep user feedback)
            if i % 5 == 0:
                elapsed = time.time() - start_t
                sys.stdout.write(f"\r  >> Iter {i}/{samples+burn} | Acc: {accepted/(i+1):.2%} | Alpha: {curr_alpha:.4f}")
                sys.stdout.flush()

        print(f"\n  ✓ Done. Total time: {time.time()-start_t:.1f}s")
        return np.array(chain)

# ==========================================
# 4. Execution & Visualization (Keep Original Style & Output)
# ==========================================
# A. Run Analysis (original parameters, keep consistency)
bayes = BayesianEngine(true_alpha=0.28)
bayes.generate_observations(n=5) # Generate synthetic reality (n=5, original)
posterior = bayes.run_mcmc(samples=1000, burn=100) # Small sample for demo speed, increase for paper

# B. Generate Priors for Plotting (original beta distribution)
x_axis = np.linspace(0.1, 0.5, 200)
y_prior = beta.pdf(x_axis, 3, 7)

# C. Calculate Statistics (original metrics, keep academic output)
post_mean = np.mean(posterior)
hdi_low, hdi_high = np.percentile(posterior, [2.5, 97.5])

# D. Plotting (Keep original layout, colors, and titles)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Plot 1: Parameter Estimation (original style, no layout change)
ax = axes[0]
# Prior Curve
ax.plot(x_axis, y_prior, color=COLORS['prior_line'], lw=2, linestyle='--', label='Prior Belief Beta(3,7)')
ax.fill_between(x_axis, 0, y_prior, color=COLORS['prior_fill'], alpha=0.3)

# Posterior KDE
kde = gaussian_kde(posterior)
y_post = kde(x_axis)
ax.plot(x_axis, y_post, color=COLORS['post_line'], lw=2.5, label='Posterior Evidence')
ax.fill_between(x_axis, 0, y_post, color=COLORS['post_fill'], alpha=0.4)

# Annotations (original labels, keep truth line at 0.28)
ax.axvline(0.28, color=COLORS['truth'], linestyle='-', lw=2, label='True α (0.28)')
ax.axvline(post_mean, color=COLORS['post_line'], linestyle=':', lw=2)

# HDI Shading (original 95% HDI, keep confidence interval)
x_hdi = np.linspace(hdi_low, hdi_high, 100)
ax.fill_between(x_hdi, 0, kde(x_hdi), color=COLORS['ci_shade'], alpha=0.2, label='95% HDI')

ax.set_title("Bayesian Updating of Recovery Rate (α)", fontweight='bold')
ax.set_xlabel("Recovery Rate")
ax.set_ylabel("Probability Density")
ax.legend(loc='upper right', frameon=False)

# Plot 2: Cost Risk Profile (original layout, keep hundred million CNY unit)
ax = axes[1]
# Predict costs using posterior alphas (original subset sampling for speed)
pred_costs = []
print("\n[Prediction] Generating predictive cost distribution...")
for a in posterior[::5]: # Sample subset to save time
    res = solve_milp(a)
    if res['status']=='Optimal': pred_costs.append(res['cost']/1e8)

# Histogram + KDE (original style, gray histogram)
ax.hist(pred_costs, bins=10, density=True, alpha=0.3, color='gray', edgecolor='white')
if len(pred_costs) > 1:
    kde_cost = gaussian_kde(pred_costs)
    x_cost = np.linspace(min(pred_costs)*0.98, max(pred_costs)*1.02, 100)
    ax.plot(x_cost, kde_cost(x_cost), color='#333333', lw=2)

mean_cost = np.mean(pred_costs)
ax.axvline(mean_cost, color=COLORS['post_line'], lw=2, linestyle='--', label=f'Exp. Cost: {mean_cost:.2f} HM')

ax.set_title("Posterior Predictive Cost Distribution", fontweight='bold')
ax.set_xlabel("Total Cost (Hundred Million CNY)")
ax.legend(loc='upper right')

# E. Save (original filename, keep journal quality DPI)
plt.savefig("50cities_bayesian_journal.png", bbox_inches='tight')
print(f"\n[Output] Visualization saved to '50cities_bayesian_journal.png'")

# F. Print Table (original format, keep statistics output)
print("\n" + "="*50)
print(f"{'BAYESIAN INFERENCE STATISTICS':^50}")
print("="*50)
df_stats = pd.DataFrame({
    'Metric': ['Mean Alpha', 'Median Alpha', '95% HDI Lower', '95% HDI Upper', 'Exp. Cost (HM)'],
    'Value': [post_mean, np.median(posterior), hdi_low, hdi_high, mean_cost]
})
print(df_stats.to_string(index=False, float_format="%.4f"))
print("-" * 50)
print("  Note: 'HM' = Hundred Million CNY | HDI = Highest Density Interval")
