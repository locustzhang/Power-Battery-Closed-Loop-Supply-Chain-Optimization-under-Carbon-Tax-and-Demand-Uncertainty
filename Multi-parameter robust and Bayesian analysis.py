"""
50 Cities EV Battery CLSC: Robust vs. Bayesian Risk Analysis (Publication Ready)
Model: Exact MILP (50 Markets, 6 Factories, 22 Recyclers)
Output: Nature/Science Style Plots
"""

import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from scipy.stats import beta, norm, gaussian_kde
import pandas as pd
import numpy as np
import time
import sys
import warnings

# ==========================================
# 0. Global Configuration (Journal Style Optimization)
# ==========================================
warnings.filterwarnings('ignore')
np.random.seed(2025)

# --- 核心优化：顶刊绘图风格设置 ---
plt.rcParams.update({
    # 字体设置
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # 优先使用Arial
    'mathtext.fontset': 'stix', # 数学公式字体类似Times，与正文区分
    'font.size': 12,
    
    # 线条与轴
    'axes.linewidth': 1.0,
    'axes.spines.top': False,   # 去除顶部边框
    'axes.spines.right': False, # 去除右侧边框
    'lines.linewidth': 2.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    
    # 颜色与布局
    'figure.dpi': 150,          # 屏幕显示清晰度
    'savefig.dpi': 600,         # 导出图片印刷级清晰度
    'figure.autolayout': False, # 使用constrained_layout替代
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': False,    # 图例无边框
})

# 定义专业配色 (Nature/Science Palette)
COLORS = {
    'bayes_line': '#00468B',    # 这种深蓝用于主线
    'bayes_fill': '#42B540',    # 这种绿色用于贝叶斯填充（形成蓝绿对比）
    'bayes_fill_sub': '#0099B4',# 备用青色
    'robust_line': '#ED0000',   # 这种深红用于鲁棒主线
    'robust_fill': '#FD8D3C',   # 这种橙色用于鲁棒填充
    'baseline': '#333333',      # 深灰用于基准线
    'text': '#2C3E50'           # 文本颜色
}

# ==========================================
# 1. Data Generation (Unchanged)
# ==========================================
def prepare_50cities_data():
    NATIONAL_SALES = 12866000
    TOTAL_RETIRED = 820000
    UNIT_WEIGHT = 0.5
    TOTAL_UNITS = TOTAL_RETIRED / UNIT_WEIGHT

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

    fixed_cost = {f"R_{c}": cost * 3000 for c, _, cost in recycler_cfg}
    demand_base = {f"M_{c}": city_demand[c] for c, _ in city_weights}
    demand_uncert = {k: v * 0.2 for k, v in demand_base.items()}

    return markets, factories, candidates, locations, fixed_cost, demand_base, demand_uncert

markets, factories, candidates, locations, fixed_cost, demand_base, demand_uncert = prepare_50cities_data()

# ==========================================
# 2. Exact MILP Model (Unchanged)
# ==========================================
def get_dist(n1, n2):
    p1, p2 = locations[n1], locations[n2]
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 100

def solve_exact_milp(params):
    alpha = params['alpha']
    tax = params['carbon_tax']
    cap = params['carbon_cap']
    capacity = params['capacity']

    TRANS_COST = 1.6
    FWD_CARBON_FACTOR = 0.0004
    REV_CARBON_FACTOR = 0.0030
    MAX_DIST = 600
    GAMMA = 1.0

    prob = pulp.LpProblem("CLSC_Exact_Revised", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0, cat='Continuous')
    z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0, cat='Continuous')
    y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
    excess_e = pulp.LpVariable("ExcE", 0, cat='Continuous')

    cost_trans = pulp.lpSum([x[i][j]*get_dist(i,j)*TRANS_COST for i in factories for j in markets]) + \
                 pulp.lpSum([z[j][k]*get_dist(j,k)*TRANS_COST for j in markets for k in candidates])

    emission = pulp.lpSum([x[i][j]*get_dist(i,j)*FWD_CARBON_FACTOR for i in factories for j in markets]) + \
               pulp.lpSum([z[j][k]*get_dist(j,k)*REV_CARBON_FACTOR for j in markets for k in candidates])

    cost_fixed = pulp.lpSum([fixed_cost[k]*y[k] for k in candidates])

    prob += cost_fixed + cost_trans + excess_e * tax

    prob += excess_e >= emission - cap

    for j in markets:
        d_robust = demand_base[j] + GAMMA * demand_uncert[j]
        prob += pulp.lpSum([x[i][j] for i in factories]) >= d_robust
        prob += pulp.lpSum([z[j][k] for k in candidates]) >= demand_base[j] * alpha
        
        for k in candidates:
            if get_dist(j, k) > MAX_DIST:
                prob += z[j][k] == 0

    for k in candidates:
        prob += pulp.lpSum([z[j][k] for j in markets]) <= capacity * y[k]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == 'Optimal':
        return pulp.value(prob.objective)
    else:
        return np.nan

# ==========================================
# 3. Simulation Logic (Unchanged)
# ==========================================
print("="*60)
print("  SIMULATION START: BAYESIAN VS ROBUST (50 CITIES)")
print("="*60)

N_SIMS = 1000
BASE_PARAMS = {
    'alpha': 0.28,
    'carbon_tax': 65,
    'carbon_cap': 100000,
    'capacity': 80000
}

results_log = []

print(f"\n[1/2] Robust Sampling ({N_SIMS} runs) [Interval +/- 30%]...")
robust_costs = []
for i in range(N_SIMS):
    p = BASE_PARAMS.copy()
    p['alpha'] = np.random.uniform(0.196, 0.364)
    p['carbon_tax'] = np.random.uniform(45.5, 84.5)
    p['carbon_cap'] = np.random.uniform(70000, 130000)

    cost = solve_exact_milp(p)
    if not np.isnan(cost):
        robust_costs.append(cost)
        results_log.append({'Method': 'Robust', 'Cost': cost/1e8, 'Alpha': p['alpha'], 'Tax': p['carbon_tax']})
    if i % 50 == 0: sys.stdout.write(f"\r  >> Progress: {i/N_SIMS:.0%}")
sys.stdout.write("\n")

print(f"\n[2/2] Bayesian Sampling ({N_SIMS} runs) [Converged Posterior]...")
bayes_costs = []
bayes_alphas = []
bayes_taxes = []
bayes_caps = []
np.random.seed(999)

for i in range(N_SIMS):
    p = BASE_PARAMS.copy()
    p['alpha'] = beta.rvs(100, 257)
    p['carbon_tax'] = np.random.normal(65, 2.0)
    p['carbon_cap'] = np.random.normal(100000, 5000)

    bayes_alphas.append(p['alpha'])
    bayes_taxes.append(p['carbon_tax'])
    bayes_caps.append(p['carbon_cap'])

    cost = solve_exact_milp(p)
    if not np.isnan(cost):
        bayes_costs.append(cost)
        results_log.append({'Method': 'Bayesian', 'Cost': cost/1e8, 'Alpha': p['alpha'], 'Tax': p['carbon_tax']})
    if i % 50 == 0: sys.stdout.write(f"\r  >> Progress: {i/N_SIMS:.0%}")
sys.stdout.write("\n")

# ==========================================
# 4. Visualization (HEAVILY OPTIMIZED)
# ==========================================
print("\n[3/3] Generating High-Resolution Plots...")

rob_data = np.array(robust_costs) / 1e8
bay_data = np.array(bayes_costs) / 1e8

# --- Figure 1: Cost Risk Profile (Nature Style) ---
fig1, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

# 计算KDE
x_eval = np.linspace(min(rob_data.min(), bay_data.min())*0.95, 
                     max(rob_data.max(), bay_data.max())*1.05, 500)
kde_rob = gaussian_kde(rob_data)(x_eval)
kde_bay = gaussian_kde(bay_data)(x_eval)

# 1. 绘制鲁棒分布 (红色系)
ax.fill_between(x_eval, 0, kde_rob, color=COLORS['robust_fill'], alpha=0.3, label='_nolegend_')
ax.plot(x_eval, kde_rob, color=COLORS['robust_line'], lw=2, linestyle='-', label='Robust Optimization')
# 鲁棒均值线
rob_mean = np.mean(rob_data)
ax.axvline(rob_mean, color=COLORS['robust_line'], linestyle=':', lw=1.5, alpha=0.8)

# 2. 绘制贝叶斯分布 (蓝色系)
ax.fill_between(x_eval, 0, kde_bay, color=COLORS['bayes_fill_sub'], alpha=0.5, label='_nolegend_')
ax.plot(x_eval, kde_bay, color=COLORS['bayes_line'], lw=2.5, label='Bayesian Inference')
# 贝叶斯均值线
bay_mean = np.mean(bay_data)
ax.axvline(bay_mean, color=COLORS['bayes_line'], linestyle='--', lw=1.5, alpha=0.8)

# 3. 基准线 (Baseline)
baseline_cost = solve_exact_milp(BASE_PARAMS) / 1e8
ax.axvline(baseline_cost, color=COLORS['baseline'], linestyle='-', lw=1.5, label='Deterministic Baseline')

# 4. 统计信息文本框 (专业排版)
rob_width = np.percentile(rob_data, 97.5) - np.percentile(rob_data, 2.5)
bay_width = np.percentile(bay_data, 97.5) - np.percentile(bay_data, 2.5)
reduction = (rob_width - bay_width) / rob_width * 100

stats_text = (
    r"$\bf{Uncertainty\ Reduction}$" + "\n"
    r"------------------------" + "\n"
    fr"Robust 95% CI:  {rob_width:.2f} HM" + "\n"
    fr"Bayes 95% HDI:  {bay_width:.2f} HM" + "\n"
    fr"$\bf{{\Delta Precision: +{reduction:.1f}\%}}$"
)
# 使用bbox增加边框和圆角
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, color=COLORS['text'])

# 5. 轴和标题美化
ax.set_xlabel('Total Network Cost ($10^8$ CNY)', fontweight='bold', labelpad=10)
ax.set_ylabel('Probability Density', fontweight='bold', labelpad=10)
ax.set_title('Cost Risk Profile: Robust vs. Bayesian',
             loc='center', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize=10, frameon=False, ncol=1)

# 添加箭头标注 (Insight)
ax.annotate('Baseline', xy=(baseline_cost, max(kde_bay)*0.8), xytext=(baseline_cost+1.0, max(kde_bay)*0.9),
            arrowprops=dict(arrowstyle="->", color=COLORS['baseline']), color=COLORS['baseline'])

plt.savefig("Fig1_Risk_Profile_Publication.png", dpi=600, bbox_inches='tight')
plt.show()

# --- Figure 2: Posterior Parameters (Nature Style) ---
fig2, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

params_data = [
    (bayes_alphas, 0.28, r'Recovery Rate ($\alpha$)', COLORS['bayes_line'], '{:.2f}'),
    (bayes_taxes, 65, r'Carbon Tax ($C_{tax}$)', COLORS['robust_line'], '{:.0f}'),
    (np.array(bayes_caps)/1000, 100, r'Carbon Cap ($10^3$ t)', '#8E44AD', '{:.0f}') # 紫色
]
titles = ['(a) Recovery Rate', '(b) Carbon Tax', '(c) Carbon Cap']

for i, (data, base, label, col, fmt) in enumerate(params_data):
    ax = axes[i]
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)
    y = kde(x)
    
    # 填充与线条
    ax.plot(x, y, color=col, lw=2)
    ax.fill_between(x, 0, y, color=col, alpha=0.15)
    
    # 95% HDI 区域高亮
    low, high = np.percentile(data, [2.5, 97.5])
    mask = (x >= low) & (x <= high)
    ax.fill_between(x, 0, y, where=mask, color=col, alpha=0.4, label='95% HDI')
    
    # 基准线
    ax.axvline(base, color='#555555', linestyle='--', lw=1.2, label='Baseline')
    
    # 标注文本 (Mean & CI)
    mean_val = np.mean(data)
    info_text = fr"$\mu={fmt.format(mean_val)}$" + "\n" + fr"$CI_{{95\%}}=[{fmt.format(low)}, {fmt.format(high)}]$"
    ax.text(0.96, 0.96, info_text, transform=ax.transAxes, ha='right', va='top', 
            fontsize=9, bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.8))

    # 轴标签与标题
    ax.set_xlabel(label, fontweight='bold')
    if i == 0: 
        ax.set_ylabel('Posterior Density', fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.set_ylabel('')
        
    ax.set_title(titles[i], loc='left', fontsize=11, fontweight='bold')
    
    # 格式化X轴
    if i == 0:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

fig2.suptitle('Posterior Distributions of Key Uncertain Parameters',
              ha='center',  # 显式指定水平居中
              fontsize=14,
              fontweight='bold',
              y=1.06)

plt.savefig("Fig2_Parameters_Publication.png", dpi=600, bbox_inches='tight')
plt.show()

# ==========================================
# 5. Summary Text (Keep Logic)
# ==========================================
print("\n" + "="*60)
print(f"{'RESULTS SUMMARY':^60}")
print("="*60)
print(f"Baseline Cost: {baseline_cost:.3f} HM")
print(f"Robust Mean:   {np.mean(rob_data):.3f} HM (Width: {rob_width:.3f})")
print(f"Bayes Mean:    {np.mean(bay_data):.3f} HM (Width: {bay_width:.3f})")
print(f"Improvement:   {reduction:.2f}% reduction in uncertainty interval.")
print("="*60)
