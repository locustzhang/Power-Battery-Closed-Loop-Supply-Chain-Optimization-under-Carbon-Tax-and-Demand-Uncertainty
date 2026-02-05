"""
50 Cities EV Battery CLSC: Bayesian Uncertainty Analysis (Nature/Science Style)
Method: Exact Mixed-Integer Linear Programming (MILP) embedded in Metropolis-Hastings MCMC
Output: High-Resolution Journal Plots
"""

import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import time
from scipy.stats import beta, norm, gaussian_kde
import warnings
import sys
import random

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ==========================================
# 0. Global Configuration (Journal Style Optimization)
# ==========================================
# 固定随机种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- 核心优化：顶刊绘图风格设置 ---
plt.rcParams.update({
    # 字体与渲染
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',  # 数学公式使用衬线体，与正文区分
    'font.size': 12,
    
    # 坐标轴与边框
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    
    # 布局与清晰度
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'figure.constrained_layout.use': True,
})

# 定义专业配色 (Nature/Science Palette)
COLORS = {
    'prior_line': '#00468B',    # Science Blue
    'prior_fill': '#A6CEE3',    # Light Blue
    'post_line': '#ED0000',     # Science Red
    'post_fill': '#FB9A99',     # Light Red
    'truth': '#333333',         # Dark Gray (Baseline)
    'text': '#2C3E50'           # Dark Slate
}

# ==========================================
# 1. Data Preparation (Aligned with Paper Data & Model)
# ==========================================
print("="*70)
print("  50 CITIES CLSC NETWORK: BAYESIAN INFERENCE (EXACT MILP)")
print("="*70)

# 论文核心校准参数（直接提取，避免冗余计算）
TOTAL_UNITS_NATIONAL = 820000 / 0.5  # 1,640,000 单位（82万吨 ÷ 0.5吨/单位）
PAPER_CITY_DEMAND = {
    "Chengdu": 39500, "Hangzhou": 39200, "Shenzhen": 38400, "Shanghai": 37900,
    "Beijing": 37100, "Guangzhou": 35300, "Zhengzhou": 30300, "Chongqing": 28900,
    "XiAn": 28800, "Tianjin": 28700, "Wuhan": 28100, "Suzhou": 28000,
    "Hefei": 21200, "Wuxi": 19500, "Ningbo": 19500, "Dongguan": 18400,
    "Nanjing": 18300, "Changsha": 17600, "Wenzhou": 17300, "Shijiazhuang": 15700,
    "Jinan": 15500, "Foshan": 15300, "Qingdao": 15200, "Changchun": 14800,
    "Shenyang": 14400, "Nanning": 13400, "Taiyuan": 12500, "Kunming": 12300,
    "Linyi": 12100, "Taizhou": 11700, "Jinhua": 11500, "Xuzhou": 11200,
    "Haikou": 10900, "Jining": 10600, "Xiamen": 10300, "Baoding": 10200,
    "Nanchang": 9700, "Changzhou": 9600, "Guiyang": 9300, "Luoyang": 9200,
    "Tangshan": 8700, "Nantong": 8700, "Haerbin": 8600, "Handan": 8500,
    "Weifang": 8500, "Wulumuqi": 8200, "Quanzhou": 8200, "Fuzhou": 8100,
    "Zhongshan": 7800, "Jiaxing": 7800
}

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

# 回收中心配置（对齐论文表~\ref{tab:fixed_cost_calibration}，年度分摊成本：万元→元）
recycler_config = [
    ("Hefei", (31.82, 117.22), 5800 * 10000), ("Zhengzhou", (34.76, 113.65), 5300 * 10000),
    ("Guiyang", (26.64, 106.63), 5000 * 10000), ("Changsha", (28.23, 112.94), 6200 * 10000),
    ("Wuhan", (30.59, 114.30), 5800 * 10000), ("Yibin", (28.77, 104.63), 7000 * 10000),
    ("Nanchang", (28.68, 115.86), 5500 * 10000), ("XiAn", (34.34, 108.94), 5600 * 10000),
    ("Tianjin", (39.13, 117.20), 5700 * 10000), ("Nanjing", (32.05, 118.78), 5900 * 10000),
    ("Hangzhou", (30.27, 120.15), 6000 * 10000), ("Changchun", (43.88, 125.32), 4800 * 10000),
    ("Nanning", (22.82, 108.32), 5200 * 10000), ("Shenzhen", (22.54, 114.05), 6500 * 10000),
    ("Qingdao", (36.07, 120.38), 5400 * 10000), ("Haerbin", (45.80, 126.53), 4600 * 10000),
    ("Fuzhou", (26.08, 119.30), 5100 * 10000), ("Xiamen", (24.48, 118.08), 5300 * 10000),
    ("Kunming", (25.04, 102.71), 4900 * 10000), ("Wulumuqi", (43.83, 87.62), 4700 * 10000),
    ("Haikou", (20.02, 110.35), 5000 * 10000), ("Shenyang", (41.80, 123.43), 4900 * 10000)
]

factory_config = [
    ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# 初始化集合与字典
markets, factories, candidates = [], [], []
locations, city_demand = {}, {}

# 市场数据（对齐论文需求表）
for c, demand in PAPER_CITY_DEMAND.items():
    city_demand[c] = demand
    markets.append(f"M_{c}")
    locations[f"M_{c}"] = city_coords[c]

# 工厂数据
for c, pos in factory_config:
    factories.append(f"F_{c}")
    locations[f"F_{c}"] = pos

# 回收中心数据
for c, pos, cost in recycler_config:
    candidates.append(f"R_{c}")
    locations[f"R_{c}"] = pos

# 固定成本（直接使用论文校准值，无额外放大）
fixed_cost = {f"R_{c}": cost for c, _, cost in recycler_config}
demand_base = {f"M_{c}": city_demand[c] for c in PAPER_CITY_DEMAND.keys()}
demand_uncertainty = {k: v * 0.2 for k, v in demand_base.items()}  # 20% 需求波动（论文假设）

# ==========================================
# 2. Solver Engine (Aligned with Paper MILP Model)
# ==========================================
def get_dist(n1, n2):
    """计算两点间地理距离（km），对齐论文空间约束"""
    p1, p2 = locations[n1], locations[n2]
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 100

def solve_milp(alpha, demand_dict=None, verbose=False):
    """
    修正后的MILP求解器，对齐论文数学模型参数
    返回：最优状态、总成本、回收中心数量、总碳排放、超额碳排放
    """
    # 论文校准核心参数（替换原硬编码偏差值）
    TRANS_COST = 1.6  # 元/单位·km
    CARBON_TAX = 65   # 元/吨CO₂
    FWD_CARBON_FACTOR = 0.004  # 吨CO₂/单位·km（论文正向因子）
    REV_CARBON_FACTOR = 0.025  # 吨CO₂/单位·km（论文逆向因子）
    CARBON_CAP = 1500000       # 150万吨CO₂（论文碳配额）
    CAPACITY = 80000           # 单厂最大处理能力（单位/年，对齐论文）
    MAX_REV_DIST = 600         # 逆向物流最大半径（km）

    # 初始化MILP问题
    prob = pulp.LpProblem("Bayes_MILP", pulp.LpMinimize)

    # 决策变量（对齐论文定义）
    x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0, cat='Continuous')  # 正向流量
    z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0, cat='Continuous')  # 逆向流量
    y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')                   # 回收中心建设决策
    excess_e = pulp.LpVariable("ExcessE", 0, cat='Continuous')                    # 超额碳排放

    # 成本项（对齐论文目标函数）
    cost_trans = pulp.lpSum([x[i][j]*get_dist(i,j)*TRANS_COST for i in factories for j in markets]) + \
                 pulp.lpSum([z[j][k]*get_dist(j,k)*TRANS_COST for j in markets for k in candidates])
    emission = pulp.lpSum([x[i][j]*get_dist(i,j)*FWD_CARBON_FACTOR for i in factories for j in markets]) + \
               pulp.lpSum([z[j][k]*get_dist(j,k)*REV_CARBON_FACTOR for j in markets for k in candidates])
    cost_fixed = pulp.lpSum([fixed_cost[k]*y[k] for k in candidates])

    # 目标函数：最小化总成本（固定成本+运输成本+碳税成本）
    prob += cost_fixed + cost_trans + excess_e*CARBON_TAX

    # 约束条件（对齐论文模型）
    prob += excess_e >= emission - CARBON_CAP  # 碳配额约束
    current_demand = demand_dict if demand_dict else demand_base

    for j in markets:
        # 1. 鲁棒需求覆盖约束（1.2倍基础需求 = 基础+20%波动，对齐论文γ=1.0）
        prob += pulp.lpSum([x[i][j] for i in factories]) >= current_demand[j] * 1.2
        # 2. 回收政策约束（最低回收率α）
        prob += pulp.lpSum([z[j][k] for k in candidates]) >= current_demand[j] * alpha
        # 3. 地理辐射约束（超600km禁止逆向物流）
        for k in candidates:
            if get_dist(j, k) > MAX_REV_DIST:
                prob += z[j][k] == 0

    # 4. 回收中心容量约束
    for k in candidates:
        prob += pulp.lpSum([z[j][k] for j in markets]) <= CAPACITY * y[k]

    # 求解模型
    status = prob.solve(pulp.PULP_CBC_CMD(msg=verbose))

    # 提取结果（扩展关键指标，保留原格式兼容性）
    if pulp.LpStatus[status] == 'Optimal':
        recycler_count = sum(1 for k in candidates if pulp.value(y[k]) == 1.0)
        total_emission = pulp.value(emission)
        total_excess_e = pulp.value(excess_e)
        return {
            'status': 'Optimal',
            'cost': pulp.value(prob.objective),
            'recycler_count': recycler_count,
            'total_emission': total_emission,
            'excess_emission': total_excess_e
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
# 3. Bayesian Logic (Unchanged, Keep Original Inference Logic)
# ==========================================
class BayesianEngine:
    def __init__(self, true_alpha=0.28):
        self.true_alpha = true_alpha
        self.obs_costs = []

    def generate_observations(self, n=5):
        print(f"\n[Generation] Creating {n} synthetic observations based on True α={self.true_alpha}...")
        for i in range(n):
            d_pert = {k: v * np.random.normal(1, 0.05) for k, v in demand_base.items()}
            res = solve_milp(self.true_alpha, d_pert)
            if res['status'] == 'Optimal':
                self.obs_costs.append(res['cost'])
            print(f"  -> Obs {i+1}: {res['cost']:,.0f} CNY")
        self.obs_mean = np.mean(self.obs_costs)
        self.obs_std = np.std(self.obs_costs)

    def log_likelihood(self, alpha):
        res = solve_milp(alpha)
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
# 4. Execution & Visualization (HEAVILY OPTIMIZED, Keep Original Style)
# ==========================================
# A. Run Analysis (Unchanged logic)
bayes = BayesianEngine(true_alpha=0.28)
bayes.generate_observations(n=5)
posterior = bayes.run_mcmc(samples=1000, burn=100)

# B. Generate Prediction Data (Unchanged logic)
print("\n[Prediction] Generating predictive cost distribution...")
pred_costs = []
for a in posterior[::5]: # Sample subset to save time
    res = solve_milp(a)
    if res['status']=='Optimal': pred_costs.append(res['cost']/1e8)

# --- START OF OPTIMIZED PLOTTING ---
print("\n[Plotting] Generating publication-quality figures...")

# Calculate Statistics
x_axis = np.linspace(0.1, 0.5, 200)
y_prior = beta.pdf(x_axis, 3, 7)
post_mean = np.mean(posterior)
hdi_low, hdi_high = np.percentile(posterior, [2.5, 97.5])
kde_post = gaussian_kde(posterior)

# Setup Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# ----------------------------
# Plot 1: Parameter Updating (Left)
# ----------------------------
ax = axes[0]

# 1. Prior (Blue, Dashed)
ax.plot(x_axis, y_prior, color=COLORS['prior_line'], lw=2, linestyle='--', label='Prior Belief')
ax.fill_between(x_axis, 0, y_prior, color=COLORS['prior_fill'], alpha=0.3)

# 2. Posterior (Red, Solid)
y_post = kde_post(x_axis)
ax.plot(x_axis, y_post, color=COLORS['post_line'], lw=2.5, label='Posterior Evidence')
ax.fill_between(x_axis, 0, y_post, color=COLORS['post_fill'], alpha=0.4)

# 3. 95% HDI Shading (Darker Red Area)
x_hdi = np.linspace(hdi_low, hdi_high, 100)
ax.fill_between(x_hdi, 0, kde_post(x_hdi), color=COLORS['post_line'], alpha=0.15)

# 4. Truth Line & Annotation
ax.axvline(0.28, color=COLORS['truth'], linestyle='-', lw=1.5, alpha=0.6)
ax.annotate('True $\\alpha=0.28$', xy=(0.28, max(y_post)*0.6), xytext=(0.35, max(y_post)*0.6),
            arrowprops=dict(arrowstyle="->", color=COLORS['truth']), fontsize=10, color=COLORS['truth'])

# 5. Statistical Stats Box (The "Journal" Touch)
stats_text = (
    r"$\bf{Posterior\ Statistics}$" + "\n"
    r"-----------------------" + "\n"
    fr"Mean ($\mu$): {post_mean:.3f}" + "\n"
    fr"95% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]"
)
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, color=COLORS['text'])

# Styling
ax.set_title('(a) Bayesian Updating of Recovery Rate ($\\alpha$)', loc='left', fontweight='bold', pad=15)
ax.set_xlabel(r'Recovery Rate ($\alpha$)')
ax.set_ylabel('Probability Density')
ax.legend(loc='upper right', frameon=False, fontsize=10)

# ----------------------------
# Plot 2: Cost Risk Profile (Right)
# ----------------------------
ax = axes[1]

# 1. Histogram (Clean Gray)
ax.hist(pred_costs, bins=12, density=True, alpha=0.2, color='gray', edgecolor='white', label='Simulated Cost')

# 2. KDE Curve (Smooth Line)
if len(pred_costs) > 1:
    kde_cost = gaussian_kde(pred_costs)
    x_cost = np.linspace(min(pred_costs)*0.95, max(pred_costs)*1.05, 200)
    ax.plot(x_cost, kde_cost(x_cost), color=COLORS['text'], lw=2, label='Density Fit')
    ax.fill_between(x_cost, 0, kde_cost(x_cost), color='gray', alpha=0.1)

# 3. Mean Cost Line
mean_cost = np.mean(pred_costs)
ax.axvline(mean_cost, color=COLORS['post_line'], lw=2, linestyle='--', label='Expected Cost')

# 4. Cost Stats Box
cost_std = np.std(pred_costs)
cost_text = (
    r"$\bf{Cost\ Projection}$" + "\n"
    r"-----------------------" + "\n"
    fr"Exp. Cost: {mean_cost:.3f} HM" + "\n"
    fr"Std. Dev:  {cost_std:.3f} HM"
)
ax.text(0.97, 0.95, cost_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props, color=COLORS['text'])

# Styling
ax.set_title('(b) Posterior Predictive Cost Distribution', loc='left', fontweight='bold', pad=15)
ax.set_xlabel('Total Network Cost ($10^8$ CNY)')
ax.set_ylabel('Probability Density')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.legend(loc='upper left', frameon=False, fontsize=10)

# Save
plt.savefig("50cities_bayesian_journal.png", dpi=600, bbox_inches='tight')
print(f"[Output] High-Res Plot saved: '50cities_bayesian_journal.png'")

# --- Print Final Table ---
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
plt.show()
