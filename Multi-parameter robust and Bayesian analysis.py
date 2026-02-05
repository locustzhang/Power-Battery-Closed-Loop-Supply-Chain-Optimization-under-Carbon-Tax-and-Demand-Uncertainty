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
# 1. Data Generation (修正：贴合论文数据，移除无效计算)
# ==========================================
def prepare_50cities_data():
    # 论文表\ref{tab:demand} 直接录入校准后的需求数据（避免计算偏差）
    city_demand_direct = {
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

    # 论文表\ref{tab:fixed_cost_calibration} 回收中心固定成本（万元，直接录入，无额外放大）
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

    # 论文定义的6个工厂
    factory_cfg = [
        ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
        ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
        ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
    ]

    # 构造集合与映射
    markets = [f"M_{c}" for c in city_demand_direct.keys()]
    factories = [f"F_{c}" for c, _ in factory_cfg]
    candidates = [f"R_{c}" for c, _, _ in recycler_cfg]

    locations = {}
    for c, pos in factory_cfg: locations[f"F_{c}"] = pos
    for c, pos in city_coords.items(): locations[f"M_{c}"] = pos
    for c, pos, _ in recycler_cfg: locations[f"R_{c}"] = pos

    # 修正1：移除无依据的 *3000，固定成本直接转换为元（万元→元）
    fixed_cost = {f"R_{c}": cost * 10000 for c, _, cost in recycler_cfg}
    # 修正2：需求基数直接取自论文表，波动项为20%（论文假设）
    demand_base = {f"M_{c}": city_demand_direct[c] for c in city_demand_direct.keys()}
    demand_uncert = {k: v * 0.2 for k, v in demand_base.items()}

    return markets, factories, candidates, locations, fixed_cost, demand_base, demand_uncert

markets, factories, candidates, locations, fixed_cost, demand_base, demand_uncert = prepare_50cities_data()

# ==========================================
# 2. Exact MILP Model (修正：贴合论文数学模型，核心参数/约束对齐)
# ==========================================
def get_dist(n1, n2):
    """修正：地理距离计算（经纬度转实际公里数，确保600km约束有效）"""
    p1, p2 = locations[n1], locations[n2]
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    # 哈弗辛公式（计算地球表面两点间距离，更贴合实际）
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    R = 6371  # 地球平均半径（公里）
    return R * c

def solve_exact_milp(params):
    alpha = params['alpha']
    tax = params['carbon_tax']
    cap = params['carbon_cap']
    capacity = params['capacity']

    # 修正3：核心参数对齐论文表\ref{tab:transport_policy}
    TRANS_COST = 1.6  # 元/单位·km（论文取值）
    FWD_CARBON_FACTOR = 0.004  # 吨CO2/单位·km（正向，论文取值，移除多余0）
    REV_CARBON_FACTOR = 0.025  # 吨CO2/单位·km（逆向，论文取值，移除多余0）
    MAX_DIST = 600  # 逆向物流最大半径（公里，论文取值）
    GAMMA = 1.0  # 鲁棒系数（论文设定，最坏情形）

    prob = pulp.LpProblem("CLSC_Exact_Revised", pulp.LpMinimize)

    # 论文定义的决策变量
    x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0, cat='Continuous')
    z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0, cat='Continuous')
    y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
    excess_e = pulp.LpVariable("ExcE", 0, cat='Continuous')

    # 修正4：运输成本计算贴合论文公式（c_ij = dist_ij × c_trans）
    cost_trans = pulp.lpSum([x[i][j]*get_dist(i,j)*TRANS_COST for i in factories for j in markets]) + \
                 pulp.lpSum([z[j][k]*get_dist(j,k)*TRANS_COST for j in markets for k in candidates])

    # 修正5：碳排放计算贴合论文公式（正/逆向差异化因子）
    emission = pulp.lpSum([x[i][j]*get_dist(i,j)*FWD_CARBON_FACTOR for i in factories for j in markets]) + \
               pulp.lpSum([z[j][k]*get_dist(j,k)*REV_CARBON_FACTOR for j in markets for k in candidates])

    # 固定成本（已修正，万元→元）
    cost_fixed = pulp.lpSum([fixed_cost[k]*y[k] for k in candidates])

    # 论文目标函数：最小化总成本（固定+运输+碳税）
    prob += cost_fixed + cost_trans + excess_e * tax

    # 约束1：碳配额约束（论文公式4）
    prob += excess_e >= emission - cap

    # 约束2：鲁棒需求覆盖 + 约束3：回收政策约束（论文公式2、3）
    for j in markets:
        d_robust = demand_base[j] + GAMMA * demand_uncert[j]  # 论文γ=1.0，最坏情形
        prob += pulp.lpSum([x[i][j] for i in factories]) >= d_robust  # 正向供应充足
        prob += pulp.lpSum([z[j][k] for k in candidates]) >= demand_base[j] * alpha  # 最低回收率

        # 约束4：地理辐射约束（论文公式5，超过600km禁止运输）
        for k in candidates:
            if get_dist(j, k) > MAX_DIST:
                prob += z[j][k] == 0

    # 约束5：回收中心容量约束（论文公式3）
    for k in candidates:
        prob += pulp.lpSum([z[j][k] for j in markets]) <= capacity * y[k]

    # 求解模型（静默模式，不输出冗余日志）
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == 'Optimal':
        return pulp.value(prob.objective)
    else:
        return np.nan

# ==========================================
# 3. Simulation Logic (修正：碳配额参数贴合论文，保留输出格式)
# ==========================================
print("="*60)
print("  SIMULATION START: BAYESIAN VS ROBUST (50 CITIES)")
print("="*60)

N_SIMS = 1000
# 修正6：BASE_PARAMS 对齐论文表\ref{tab:transport_policy}
BASE_PARAMS = {
    'alpha': 0.28,               # 法定回收率（论文取值）
    'carbon_tax': 65,            # 碳税（元/吨CO2，论文取值）
    'carbon_cap': 150000,        # 年度碳配额（万吨→吨，论文取值150万吨）
    'capacity': 80000            # 单回收中心处理能力（单位/年，论文取值）
}

results_log = []

print(f"\n[1/2] Robust Sampling ({N_SIMS} runs) [Interval +/- 30%]...")
robust_costs = []
for i in range(N_SIMS):
    p = BASE_PARAMS.copy()
    p['alpha'] = np.random.uniform(0.196, 0.364)
    p['carbon_tax'] = np.random.uniform(45.5, 84.5)
    p['carbon_cap'] = np.random.uniform(105000, 195000)  # 碳配额波动对应±30%

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
    p['carbon_cap'] = np.random.normal(150000, 7500)  # 碳配额正态分布贴合论文

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
# 4. Visualization (HEAVILY OPTIMIZED) - 完全保留原风格，无修改
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
    (np.array(bayes_caps)/1000, 150, r'Carbon Cap ($10^3$ t)', '#8E44AD', '{:.0f}') # 修正：碳配额基准值150（贴合论文）
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
# 5. Summary Text (Keep Logic) - 完全保留原输出格式，无修改
# ==========================================
print("\n" + "="*60)
print(f"{'RESULTS SUMMARY':^60}")
print("="*60)
print(f"Baseline Cost: {baseline_cost:.3f} HM")
print(f"Robust Mean:   {np.mean(rob_data):.3f} HM (Width: {rob_width:.3f})")
print(f"Bayes Mean:    {np.mean(bay_data):.3f} HM (Width: {bay_width:.3f})")
print(f"Improvement:   {reduction:.2f}% reduction in uncertainty interval.")
print("="*60)
