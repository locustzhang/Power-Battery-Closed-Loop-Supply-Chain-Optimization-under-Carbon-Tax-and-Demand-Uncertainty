import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import time
import sys
import random
from scipy.stats import beta, norm, gaussian_kde
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ==========================================
# 【1. Core Config - Keep Original CLSC Logic (Unified City Spelling)】
# ==========================================
# Plot config for original CLSC network (reserved for network plot)
PLOT_CONFIG = {
    'font_family': 'Arial',
    'font_size': 10,
    'axes_titlesize': 14,
    'axes_labelsize': 12,
    'xtick_labelsize': 10,
    'ytick_labelsize': 10,
    'legend_fontsize': 10,
    'figure_dpi': 150,
    'savefig_dpi': 600,
    'style_colors': {
        'factory_fill': '#E64B35',
        'factory_edge': '#3C5488',
        'recycler_fill': '#00A087',
        'recycler_halo': '#00A087',
        'nw_recycler_halo': '#FFA500',
        'market_fill': '#4DBBD5',
        'fwd_line': '#E64B35',
        'rev_line': '#00A087',
        'text': '#333333'
    }
}

# Region constraint config (Urumqi/Xi'an unified)
REGION_CONFIG = {
    'nw_cities': ["Urumqi", "Xi'an"],
    'max_rev_dist': 600,
    'max_rev_dist_nw': 1200,
    'nw_radius_visual': 9.0,
    'normal_radius_visual': 6.0,
    'earth_radius': 6371
}

# Core economic/policy parameters (original calibrated values)
PAPER_PARAMS = {
    'trans_cost': 1.6,
    'carbon_tax': 65,
    'carbon_factor_fwd': 0.00005,
    'carbon_factor_rev': 0.00010,
    'carbon_cap': 16000,
    'single_recycler_capacity': 80000,
    'demand_uncertainty_rate': 0.2,
    'gamma': 1.0,
    'alpha': 0.28,  # Bayesian analysis core parameter
    'cost_unit_convert': 10000
}

# City demand base
CITY_DEMAND = {
    "Chengdu": 39500, "Hangzhou": 39200, "Shenzhen": 38400, "Shanghai": 37900,
    "Beijing": 37100, "Guangzhou": 35300, "Zhengzhou": 30300, "Chongqing": 28900,
    "Xi'an": 28800, "Tianjin": 28700, "Wuhan": 28100, "Suzhou": 28000,
    "Hefei": 21200, "Wuxi": 19500, "Ningbo": 19500, "Dongguan": 18400,
    "Nanjing": 18300, "Changsha": 17600, "Wenzhou": 17300, "Shijiazhuang": 15700,
    "Jinan": 15500, "Foshan": 15300, "Qingdao": 15200, "Changchun": 14800,
    "Shenyang": 14400, "Nanning": 13400, "Taiyuan": 12500, "Kunming": 12300,
    "Linyi": 12100, "Taizhou": 11700, "Jinhua": 11500, "Xuzhou": 11200,
    "Haikou": 10900, "Jining": 10600, "Xiamen": 10300, "Baoding": 10200,
    "Nanchang": 9700, "Changzhou": 9600, "Guiyang": 9300, "Luoyang": 9200,
    "Tangshan": 8700, "Nantong": 8700, "Harbin": 8600, "Handan": 8500,
    "Weifang": 8500, "Urumqi": 8200, "Quanzhou": 8200, "Fuzhou": 8100,
    "Zhongshan": 7800, "Jiaxing": 7800
}

# City coordinates
CITY_COORDS = {
    "Chengdu": (30.67, 104.06), "Hangzhou": (30.27, 120.15), "Shenzhen": (22.54, 114.05),
    "Shanghai": (31.23, 121.47), "Beijing": (39.90, 116.40), "Guangzhou": (23.13, 113.26),
    "Zhengzhou": (34.76, 113.65), "Chongqing": (29.56, 106.55), "Xi'an": (34.34, 108.94),
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
    "Harbin": (45.80, 126.53), "Handan": (36.61, 114.49), "Weifang": (36.71, 119.16),
    "Urumqi": (43.83, 87.62), "Quanzhou": (24.87, 118.68), "Fuzhou": (26.08, 119.30),
    "Zhongshan": (22.52, 113.39), "Jiaxing": (30.75, 120.75)
}

# Recycler config
RECYCLER_CONFIG = [
    ("Hefei", (31.82, 117.22), 5800), ("Zhengzhou", (34.76, 113.65), 5300),
    ("Guiyang", (26.64, 106.63), 5000), ("Changsha", (28.23, 112.94), 6200),
    ("Wuhan", (30.59, 114.30), 5800), ("Yibin", (28.77, 104.63), 7000),
    ("Nanchang", (28.68, 115.86), 5500), ("Xi'an", (34.34, 108.94), 5600),
    ("Tianjin", (39.13, 117.20), 5700), ("Nanjing", (32.05, 118.78), 5900),
    ("Hangzhou", (30.27, 120.15), 6000), ("Changchun", (43.88, 125.32), 4800),
    ("Nanning", (22.82, 108.32), 5200), ("Shenzhen", (22.54, 114.05), 6500),
    ("Qingdao", (36.07, 120.38), 5400), ("Harbin", (45.80, 126.53), 4600),
    ("Fuzhou", (26.08, 119.30), 5100), ("Xiamen", (24.48, 118.08), 5300),
    ("Kunming", (25.04, 102.71), 4900), ("Urumqi", (43.83, 87.62), 4700),
    ("Haikou", (20.02, 110.35), 5000), ("Shenyang", (41.80, 123.43), 4900)
]

# Factory config
FACTORY_CONFIG = [
    ("Xi'an", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# ==========================================
# 【2. Bayesian Plot Config - EXACT as Reference Code (Nature/Science Style)】
# ==========================================
# Professional color palette (EXACT reference)
COLORS = {
    'prior_line': '#00468B',    # Science Blue
    'prior_fill': '#A6CEE3',    # Light Blue
    'post_line': '#ED0000',     # Science Red
    'post_fill': '#FB9A99',     # Light Red
    'truth': '#333333',         # Dark Gray (Baseline)
    'text': '#2C3E50'           # Dark Slate
}

# Global plot config (EXACT reference - journal style)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'figure.constrained_layout.use': True,
})

# ==========================================
# 【3. Helper Functions - Keep Original CLSC Logic】
# ==========================================
def is_nw_city(city_name):
    city = city_name.replace("M_", "").replace("R_", "").replace("F_", "")
    return city in REGION_CONFIG['nw_cities']

def haversine_dist(n1, n2, locations):
    lat1, lon1 = locations[n1]
    lat2, lon2 = locations[n2]
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return REGION_CONFIG['earth_radius'] * c

def add_smart_label(pos, text, color, size=9, weight='bold', dy=0):
    txt = plt.text(pos[1], pos[0] + dy, text,
                   fontsize=size, fontweight=weight, color=color,
                   ha='center', va='center', zorder=50)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.9)])

# ==========================================
# 【4. MILP Solver - Modified for Bayesian (Alpha as Input)】
# ==========================================
def solve_clsc_milp(alpha=PAPER_PARAMS['alpha'], demand_dict=None, verbose=False):
    """CLSC MILP Solver with alpha as input - keep all original constraints"""
    # Build location dict
    locations = {}
    for c, pos in FACTORY_CONFIG: locations[f"F_{c}"] = pos
    for c in CITY_DEMAND.keys(): locations[f"M_{c}"] = CITY_COORDS[c]
    for c, pos, _ in RECYCLER_CONFIG: locations[f"R_{c}"] = pos

    # Basic params assignment
    fixed_cost = {f"R_{c}": cost * PAPER_PARAMS['cost_unit_convert'] for c, _, cost in RECYCLER_CONFIG}
    demand_base = {f"M_{c}": CITY_DEMAND[c] for c in CITY_DEMAND.keys()}
    current_demand = demand_dict if demand_dict is not None else demand_base
    demand_uncertainty = {k: v * PAPER_PARAMS['demand_uncertainty_rate'] for k, v in demand_base.items()}
    TRANS_COST = PAPER_PARAMS['trans_cost']
    CARBON_TAX = PAPER_PARAMS['carbon_tax']
    CARBON_FACTOR_FWD = PAPER_PARAMS['carbon_factor_fwd']
    CARBON_FACTOR_REV = PAPER_PARAMS['carbon_factor_rev']
    CARBON_CAP = PAPER_PARAMS['carbon_cap']
    CAPACITY = PAPER_PARAMS['single_recycler_capacity']
    GAMMA = PAPER_PARAMS['gamma']
    get_dist = lambda n1, n2: haversine_dist(n1, n2, locations)

    # Define problem and variables
    prob = pulp.LpProblem("CLSC_Bayesian_Analysis", pulp.LpMinimize)
    factories = [f"F_{c}" for c, _ in FACTORY_CONFIG]
    markets = [f"M_{c}" for c in CITY_DEMAND.keys()]
    candidates = [f"R_{c}" for c, _, _ in RECYCLER_CONFIG]
    x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0)
    z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0)
    y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
    excess_e = pulp.LpVariable("ExcessE", 0)

    # Objective function
    cost_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * TRANS_COST for i in factories for j in markets])
    cost_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * TRANS_COST for j in markets for k in candidates])
    cost_fix = pulp.lpSum([fixed_cost[k] * y[k] for k in candidates])
    emit_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * CARBON_FACTOR_FWD for i in factories for j in markets])
    emit_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * CARBON_FACTOR_REV for j in markets for k in candidates])
    prob += cost_fix + cost_fwd + cost_rev + excess_e * CARBON_TAX

    # Constraints (all original CLSC constraints)
    prob += excess_e >= (emit_fwd + emit_rev) - CARBON_CAP
    for j in markets:
        prob += pulp.lpSum([x[i][j] for i in factories]) >= current_demand[j] + GAMMA * demand_uncertainty[j]
        prob += pulp.lpSum([z[j][k] for k in candidates]) >= current_demand[j] * alpha
        for k in candidates:
            dist = get_dist(j, k)
            if is_nw_city(j) or is_nw_city(k):
                if dist > REGION_CONFIG['max_rev_dist_nw']:
                    prob += z[j][k] == 0
            else:
                if dist > REGION_CONFIG['max_rev_dist']:
                    prob += z[j][k] == 0
    for k in candidates:
        prob += pulp.lpSum([z[j][k] for j in markets]) <= CAPACITY * y[k]
    nw_recyclers = [k for k in candidates if is_nw_city(k)]
    if nw_recyclers:
        prob += pulp.lpSum([y[k] for k in nw_recyclers]) >= 1, "NW_Recycler_Min_Constraint"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=verbose))

    # Extract results
    if pulp.LpStatus[prob.status] == 'Optimal':
        total_cost = pulp.value(prob.objective)
        recycler_count = len([k for k in candidates if pulp.value(y[k]) > 0.5])
        total_emission = pulp.value(emit_fwd + emit_rev)
        excess_emission = pulp.value(excess_e)
        return {
            'status': 'Optimal',
            'cost': total_cost,
            'recycler_count': recycler_count,
            'total_emission': total_emission,
            'excess_emission': excess_emission
        }
    else:
        return {
            'status': 'Infeasible',
            'cost': np.nan,
            'recycler_count': np.nan,
            'total_emission': np.nan,
            'excess_emission': np.nan
        }

# ==========================================
# 【5. Bayesian Engine - EXACT as Reference Code (Metropolis-Hastings MCMC)】
# ==========================================
class BayesianEngine:
    def __init__(self, true_alpha=0.28):
        self.true_alpha = true_alpha
        self.obs_costs = []

    def generate_observations(self, n=5):
        print(f"\n[Generation] Creating {n} synthetic observations based on True α={self.true_alpha}...")
        demand_base = {f"M_{c}": CITY_DEMAND[c] for c in CITY_DEMAND.keys()}
        for i in range(n):
            d_pert = {k: v * np.random.normal(1, 0.05) for k, v in demand_base.items()}
            res = solve_clsc_milp(self.true_alpha, d_pert)
            if res['status'] == 'Optimal':
                self.obs_costs.append(res['cost'])
            print(f"  -> Obs {i+1}: {res['cost']:,.0f} CNY")
        self.obs_mean = np.mean(self.obs_costs)
        self.obs_std = np.std(self.obs_costs)

    def log_likelihood(self, alpha):
        res = solve_clsc_milp(alpha)
        if res['status'] != 'Optimal': return -np.inf
        pred_cost = res['cost']
        return norm.logpdf(self.obs_mean, loc=pred_cost, scale=self.obs_std * 2.0)

    def run_mcmc(self, samples=1000, burn=100):
        print(f"\n[MCMC] Sampling {samples} points (Burn-in={burn})...")
        print("  Note: Exact MILP is running in the loop. Please wait.")
        chain = []
        curr_alpha = 0.3
        curr_ll = self.log_likelihood(curr_alpha)
        curr_prior = beta.logpdf(curr_alpha, 3, 7)
        accepted = 0
        start_t = time.time()
        for i in range(samples + burn):
            prop_alpha = curr_alpha + np.random.normal(0, 0.02)
            if prop_alpha <= 0.1 or prop_alpha >= 0.5:
                prop_ll = -np.inf
            else:
                prop_ll = self.log_likelihood(prop_alpha)
            prop_prior = beta.logpdf(prop_alpha, 3, 7) if 0<prop_alpha<1 else -np.inf
            if prop_ll > -np.inf:
                ratio = (prop_ll + prop_prior) - (curr_ll + curr_prior)
                if np.log(np.random.rand()) < ratio:
                    curr_alpha = prop_alpha
                    curr_ll = prop_ll
                    curr_prior = prop_prior
                    accepted += 1
            if i >= burn: chain.append(curr_alpha)
            if i % 5 == 0:
                elapsed = time.time() - start_t
                sys.stdout.write(f"\r  >> Iter {i}/{samples+burn} | Acc: {accepted/(i+1):.2%} | Alpha: {curr_alpha:.4f}")
                sys.stdout.flush()
        print(f"\n  ✓ Done. Total time: {time.time()-start_t:.1f}s")
        return np.array(chain)

# ==========================================
# 【6. Visualization - EXACT Code as Reference (No Modification)】
# ==========================================
def plot_bayesian_results(posterior, true_alpha=0.28):
    """EXACT plotting code from reference - no any modification"""
    # Generate prediction data
    print("\n[Prediction] Generating predictive cost distribution...")
    pred_costs = []
    for a in posterior[::5]:
        res = solve_clsc_milp(a)
        if res['status']=='Optimal': pred_costs.append(res['cost']/1e8)

    # Calculate Statistics
    x_axis = np.linspace(0.1, 0.5, 200)
    y_prior = beta.pdf(x_axis, 3, 7)
    post_mean = np.mean(posterior)
    hdi_low, hdi_high = np.percentile(posterior, [2.5, 97.5])
    kde_post = gaussian_kde(posterior)

    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle('Bayesian Recovery Rate Updating and Predictive Cost Analysis',
                 fontsize=16, fontweight='bold', y=1.04, color=COLORS['text'])

    # ----------------------------
    # Plot 1: Parameter Updating (Left)
    # ----------------------------
    ax = axes[0]
    ax.plot(x_axis, y_prior, color=COLORS['prior_line'], lw=2, linestyle='--', label='Prior Belief')
    ax.fill_between(x_axis, 0, y_prior, color=COLORS['prior_fill'], alpha=0.3)
    y_post = kde_post(x_axis)
    ax.plot(x_axis, y_post, color=COLORS['post_line'], lw=2.5, label='Posterior Evidence')
    ax.fill_between(x_axis, 0, y_post, color=COLORS['post_fill'], alpha=0.4)
    x_hdi = np.linspace(hdi_low, hdi_high, 100)
    ax.fill_between(x_hdi, 0, kde_post(x_hdi), color=COLORS['post_line'], alpha=0.15)
    ax.axvline(true_alpha, color=COLORS['truth'], linestyle='-', lw=1.5, alpha=0.6)
    ax.annotate('True $\\alpha=0.28$', xy=(0.28, max(y_post)*0.6), xytext=(0.35, max(y_post)*0.6),
                arrowprops=dict(arrowstyle="->", color=COLORS['truth']), fontsize=10, color=COLORS['truth'])
    stats_text = (
        r"$\bf{Posterior\ Statistics}$" + "\n"
        r"-----------------------" + "\n"
        fr"Mean ($\mu$): {post_mean:.3f}" + "\n"
        fr"95% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
    ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color=COLORS['text'])
    ax.set_title('(a) Bayesian Updating of Recovery Rate ($\\alpha$)', loc='left', fontweight='bold', pad=15)
    ax.set_xlabel(r'Recovery Rate ($\alpha$)')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    # ----------------------------
    # Plot 2: Cost Risk Profile (Right)
    # ----------------------------
    ax = axes[1]
    ax.hist(pred_costs, bins=12, density=True, alpha=0.2, color='gray', edgecolor='white', label='Simulated Cost')
    if len(pred_costs) > 1:
        kde_cost = gaussian_kde(pred_costs)
        x_cost = np.linspace(min(pred_costs)*0.95, max(pred_costs)*1.05, 200)
        ax.plot(x_cost, kde_cost(x_cost), color=COLORS['text'], lw=2, label='Density Fit')
        ax.fill_between(x_cost, 0, kde_cost(x_cost), color='gray', alpha=0.1)
    mean_cost = np.mean(pred_costs)
    ax.axvline(mean_cost, color=COLORS['post_line'], lw=2, linestyle='--', label='Expected Cost')
    cost_std = np.std(pred_costs)
    cost_text = (
        r"$\bf{Cost\ Projection}$" + "\n"
        r"-----------------------" + "\n"
        fr"Exp. Cost: {mean_cost:.3f} HM" + "\n"
        fr"Std. Dev:  {cost_std:.3f} HM"
    )
    ax.text(0.97, 0.95, cost_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, color=COLORS['text'])
    ax.set_title('(b) Posterior Predictive Cost Distribution', loc='left', fontweight='bold', pad=15)
    ax.set_xlabel('Total Network Cost ($10^8$ CNY)')
    ax.set_ylabel('Probability Density')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper left', frameon=False, fontsize=10)

    # Save
    plt.savefig("50cities_bayesian_journal.png", dpi=600, bbox_inches='tight')
    print(f"[Output] High-Res Plot saved: '50cities_bayesian_journal.png'")

    # Print Final Table (EXACT as reference)
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
    return mean_cost

# ==========================================
# 【7. Baseline CLSC Run + Bayesian Execution】
# ==========================================
if __name__ == "__main__":
    # Print header
    print("="*70)
    print("  50 CITIES EV BATTERY CLSC: BAYESIAN UNCERTAINTY ANALYSIS")
    print("="*70)

    # 1. Run baseline CLSC model (original alpha=0.28)
    print("\n[Baseline] Running original CLSC optimization (α=0.28)...")
    baseline_res = solve_clsc_milp(alpha=0.28)
    if baseline_res['status'] == 'Optimal':
        print(f"  Baseline Total Cost: {baseline_res['cost']/1e8:.3f} HM CNY")
        print(f"  Baseline Recycler Count: {baseline_res['recycler_count']}")
        print(f"  Baseline Total Emission: {baseline_res['total_emission']:.2f} tCO2")

    # 2. Run Bayesian analysis (EXACT reference code flow)
    bayes = BayesianEngine(true_alpha=0.28)
    bayes.generate_observations(n=5)
    posterior = bayes.run_mcmc(samples=1000, burn=100)
    plot_bayesian_results(posterior, true_alpha=0.28)

    # Show plot
    plt.show()

    print("\n[Complete] Bayesian analysis finished - results saved and printed!")
