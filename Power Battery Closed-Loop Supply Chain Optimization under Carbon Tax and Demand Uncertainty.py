import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

# ==========================================
# 【核心修改：统一城市英文拼写（国际通用格式）】
# 乌鲁木齐：Wulumuqi → Urumqi
# 西安：XiAn/Xi'an → Xi'an（带撇号的国际通用拼写）
# ==========================================
# ---------------------- 1. 绘图样式固定参数（论文图表格式） ----------------------
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
        'nw_recycler_halo': '#FFA500',  # 黄色（西北1200km圈）
        'market_fill': '#4DBBD5',
        'fwd_line': '#E64B35',
        'rev_line': '#00A087',
        'text': '#333333'
    }
}

# ---------------------- 2. 区域约束参数（统一拼写） ----------------------
REGION_CONFIG = {
    'nw_cities': ["Urumqi", "Xi'an"],  # 核心修改：Wulumuqi→Urumqi，XiAn→Xi'an
    'max_rev_dist': 600,  # 常规区域逆向物流最大半径（km）
    'max_rev_dist_nw': 1200,  # 西北区域逆向物流最大半径（km）
    'nw_radius_visual': 9.0,  # 西北圈视觉尺寸
    'normal_radius_visual': 6.0,  # 常规区域辐射范围视觉尺寸（绘图用）
    'earth_radius': 6371  # 地球平均半径（km，Haversine公式用）
}

# ---------------------- 3. 核心经济/政策参数（完全不变） ----------------------
PAPER_PARAMS = {
    'trans_cost': 1.6,  # 单位运输成本（元/单位·km）
    'carbon_tax': 65,  # 碳税/碳价（元/吨CO2）
    'carbon_factor_fwd': 0.00005,  # 正向物流碳排放因子（吨CO2/单位·km）
    'carbon_factor_rev': 0.00010,  # 逆向物流碳排放因子（吨CO2/单位·km）
    'carbon_cap': 16000,  # 碳配额（吨CO2，1.6万吨）
    'single_recycler_capacity': 80000,  # 单回收中心年处理能力（单位/年）
    'demand_uncertainty_rate': 0.2,  # 需求波动比例（20%）
    'gamma': 1.0,  # 需求不确定性系数（模型鲁棒性假设）
    'alpha': 0.28,  # 法定回收率目标（28%）
    'cost_unit_convert': 10000  # 万元 → 元 转换系数
}

# ---------------------- 4. 节点基础数据参数（统一拼写） ----------------------
# 4.1 城市需求基数（无修改，已为Urumqi/Xi'an）
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

# 4.2 城市经纬度（无修改，已为Urumqi/Xi'an）
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

# 4.3 回收中心配置（无修改，已为Urumqi/Xi'an）
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

# 4.4 工厂配置（核心修改：XiAn → Xi'an，统一拼写）
FACTORY_CONFIG = [
    ("Xi'an", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# ==========================================
# 【后续代码：逻辑完全不变，仅依赖统一后的拼写】
# ==========================================
# ---------------------- 第一步：全局配置初始化 ----------------------
plt.rcParams.update({
    'font.family': PLOT_CONFIG['font_family'],
    'font.size': PLOT_CONFIG['font_size'],
    'axes.titlesize': PLOT_CONFIG['axes_titlesize'],
    'axes.labelsize': PLOT_CONFIG['axes_labelsize'],
    'xtick.labelsize': PLOT_CONFIG['xtick_labelsize'],
    'ytick.labelsize': PLOT_CONFIG['ytick_labelsize'],
    'legend.fontsize': PLOT_CONFIG['legend_fontsize'],
    'figure.dpi': PLOT_CONFIG['figure_dpi'],
    'savefig.dpi': PLOT_CONFIG['savefig_dpi'],
})

# ---------------------- 第二步：辅助函数 ----------------------
def is_nw_city(city_name):
    """判断是否为西北城市（匹配统一后的Urumqi、Xi'an）"""
    city = city_name.replace("M_", "").replace("R_", "").replace("F_", "")
    return city in REGION_CONFIG['nw_cities']

def haversine_dist(n1, n2, locations):
    """Haversine公式计算实际距离（km）"""
    lat1, lon1 = locations[n1]
    lat2, lon2 = locations[n2]

    # 转弧度制
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算球面距离
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return REGION_CONFIG['earth_radius'] * c

def add_smart_label(pos, text, color, size=9, weight='bold', dy=0):
    """绘图标签辅助函数"""
    txt = plt.text(pos[1], pos[0] + dy, text,
                   fontsize=size, fontweight=weight, color=color,
                   ha='center', va='center', zorder=50)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.9)])

# ---------------------- 第三步：数据构建 ----------------------
# 构建位置字典（自动使用统一后的城市名）
locations = {}
for c, pos in FACTORY_CONFIG: locations[f"F_{c}"] = pos
for c in CITY_DEMAND.keys(): locations[f"M_{c}"] = CITY_COORDS[c]
for c, pos, _ in RECYCLER_CONFIG: locations[f"R_{c}"] = pos

# 基础参数赋值
fixed_cost = {f"R_{c}": cost * PAPER_PARAMS['cost_unit_convert'] for c, _, cost in RECYCLER_CONFIG}
demand_base = {f"M_{c}": CITY_DEMAND[c] for c in CITY_DEMAND.keys()}
demand_uncertainty = {k: v * PAPER_PARAMS['demand_uncertainty_rate'] for k, v in demand_base.items()}
TRANS_COST = PAPER_PARAMS['trans_cost']
CARBON_TAX = PAPER_PARAMS['carbon_tax']
CARBON_FACTOR_FWD = PAPER_PARAMS['carbon_factor_fwd']
CARBON_FACTOR_REV = PAPER_PARAMS['carbon_factor_rev']
CARBON_CAP = PAPER_PARAMS['carbon_cap']
CAPACITY = PAPER_PARAMS['single_recycler_capacity']
GAMMA = PAPER_PARAMS['gamma']
ALPHA = PAPER_PARAMS['alpha']

# 距离计算函数赋值
get_dist = lambda n1, n2: haversine_dist(n1, n2, locations)

# ---------------------- 第四步：模型构建 ----------------------
# 定义问题与变量
prob = pulp.LpProblem("CLSC_Deep_Analysis", pulp.LpMinimize)
factories = [f"F_{c}" for c, _ in FACTORY_CONFIG]
markets = [f"M_{c}" for c in CITY_DEMAND.keys()]
candidates = [f"R_{c}" for c, _, _ in RECYCLER_CONFIG]

x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0)
z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0)
y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
excess_e = pulp.LpVariable("ExcessE", 0)

# 目标函数
cost_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * TRANS_COST for i in factories for j in markets])
cost_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * TRANS_COST for j in markets for k in candidates])
cost_fix = pulp.lpSum([fixed_cost[k] * y[k] for k in candidates])
emit_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * CARBON_FACTOR_FWD for i in factories for j in markets])
emit_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * CARBON_FACTOR_REV for j in markets for k in candidates])
prob += cost_fix + cost_fwd + cost_rev + excess_e * CARBON_TAX

# 约束条件
prob += excess_e >= (emit_fwd + emit_rev) - CARBON_CAP

for j in markets:
    prob += pulp.lpSum([x[i][j] for i in factories]) >= demand_base[j] + GAMMA * demand_uncertainty[j]
    prob += pulp.lpSum([z[j][k] for k in candidates]) >= demand_base[j] * ALPHA

    # 区域距离约束（现在判定逻辑完全一致）
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

# 西北回收中心优先约束（拼写统一后，判定更精准）
nw_recyclers = [k for k in candidates if is_nw_city(k)]
if nw_recyclers:
    prob += pulp.lpSum([y[k] for k in nw_recyclers]) >= 1, "NW_Recycler_Min_Constraint"

# 求解模型
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# ---------------------- 第五步：结果输出 ----------------------
if pulp.LpStatus[prob.status] == 'Optimal':
    tc = pulp.value(prob.objective)
    built_k = [k for k in candidates if pulp.value(y[k]) > 0.5]
    total_qty_fwd = sum(pulp.value(x[i][j]) for i in factories for j in markets)
    total_qty_rev = sum(pulp.value(z[j][k]) for j in markets for k in candidates)
    built_nw_k = [k for k in built_k if is_nw_city(k)]

    print("\n" + "=" * 90)
    print(f"{'SCIENTIFIC ANALYSIS: CLOSED-LOOP OPTIMIZATION RESULTS':^90}")
    print("=" * 90)

    print(f"[A] COST BREAKDOWN ANALYSIS")
    print(f" - Total Objective Cost   : {tc:,.2f} CNY")
    print(f" - Fixed Construction Cost: {pulp.value(cost_fix):,.2f} CNY ({pulp.value(cost_fix) / tc * 100:>.2f}%)")
    print(f" - Forward Logistics Cost : {pulp.value(cost_fwd):,.2f} CNY ({pulp.value(cost_fwd) / tc * 100:>.2f}%)")
    print(f" - Reverse Logistics Cost : {pulp.value(cost_rev):,.2f} CNY ({pulp.value(cost_rev) / tc * 100:>.2f}%)")
    print(f" - Carbon Tax Expense     : {pulp.value(excess_e * CARBON_TAX):,.2f} CNY")
    print("-" * 90)

    print(f"[B] EMISSION & CARBON FOOTPRINT")
    print(f" - Total Carbon Emissions : {pulp.value(emit_fwd + emit_rev):,.2f} tCO2")
    total_emission = pulp.value(emit_fwd + emit_rev)
    print(f" - Forward Emission Cont. : {pulp.value(emit_fwd):,.2f} tCO2 ({pulp.value(emit_fwd) / total_emission * 100:>.2f}%)")
    print(f" - Reverse Emission Cont. : {pulp.value(emit_rev):,.2f} tCO2 ({pulp.value(emit_rev) / total_emission * 100:>.2f}%)")
    used_quota = min(total_emission, CARBON_CAP)
    print(f" - Carbon Quota Used      : {used_quota:,.2f} tCO2 / {CARBON_CAP:,.2f} tCO2")
    print("-" * 90)

    avg_f_dist = sum(pulp.value(x[i][j]) * get_dist(i, j) for i in factories for j in markets) / total_qty_fwd
    avg_r_dist = sum(pulp.value(z[j][k]) * get_dist(j, k) for j in markets for k in candidates) / total_qty_rev
    print(f" - Avg Transport Distance (Fwd): {avg_f_dist:>.2f} km")
    print(f" - Avg Transport Distance (Rev): {avg_r_dist:>.2f} km")
    print("-" * 90)

    print(f"[D] HUB OPERATIONAL STATUS")
    if built_nw_k:
        print(f" - Northwest Recyclers Built: {[k.replace('R_', '') for k in built_nw_k]}")  # 输出统一后的名称
    print(f"{'Recycling Center':<20} | {'Processed Qty':<15} | {'Utilization (%)':<15} | {'Fixed Cost (10^4 CNY)'}")
    for k in built_k:
        load = sum(pulp.value(z[j][k]) for j in markets)
        print(f"{k.replace('R_', ''):<20} | {load:<15,.0f} | {load / CAPACITY * 100:<15.2f} | {fixed_cost[k] / 10000:<,.0f}")
    print("=" * 90)

# ---------------------- 第六步：绘图 ----------------------
print("\n正在生成出版级高分辨率网络图...")
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': PLOT_CONFIG['font_family'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_aspect(1.18)
STYLE_COLORS = PLOT_CONFIG['style_colors']

# 计算最大流量用于归一化
max_flow_fwd = max([pulp.value(x[i][j]) for i in factories for j in markets] + [1])
max_flow_rev = max([pulp.value(z[j][k]) for j in markets for k in candidates] + [1])
max_demand = max(demand_base.values())

# 绘制物流网络
# 正向物流
for i in factories:
    for j in markets:
        val = pulp.value(x[i][j])
        if val > 100:
            p1, p2 = locations[i], locations[j]
            norm_val = val / max_flow_fwd
            alpha = 0.1 + norm_val * 0.8
            lw = 0.5 + norm_val * 2.5
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                    c=STYLE_COLORS['fwd_line'], alpha=alpha, linewidth=lw,
                    zorder=2, solid_capstyle='round')

# 逆向物流
for j in markets:
    for k in built_k:
        val = pulp.value(z[j][k])
        if val > 10:
            p1, p2 = locations[j], locations[k]
            norm_val = val / max_flow_rev
            alpha = 0.2 + norm_val * 0.7
            lw = 0.8 + norm_val * 2.5
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                    c=STYLE_COLORS['rev_line'], alpha=alpha, linewidth=lw,
                    linestyle=(0, (1, 1.5)), zorder=3)

# 绘制节点
built_cities = [k.replace('R_', '') for k in built_k]
factory_cities = [f.replace('F_', '') for f in factories]

# 消费市场
for m in markets:
    city_name = m.replace('M_', '')
    pos = locations[m]
    size = 20 + (demand_base[m] / max_demand) * 150
    if city_name not in built_cities and city_name not in factory_cities:
        ax.scatter(pos[1], pos[0], s=size, c=STYLE_COLORS['market_fill'],
                   alpha=0.6, edgecolors='white', lw=0.5, zorder=4)
        if demand_base[m] > sorted(demand_base.values(), reverse=True)[12]:
            add_smart_label(pos, city_name, STYLE_COLORS['text'], size=8, weight='normal', dy=-0.4)

# 回收中心（带辐射范围）
for k in built_k:
    pos = locations[k]
    city_name = k.replace('R_', '')
    if is_nw_city(k):
        # 西北回收中心（1200km黄色圈）
        circle = plt.Circle((pos[1], pos[0]), REGION_CONFIG['nw_radius_visual'],
                            color=STYLE_COLORS['nw_recycler_halo'], alpha=0.3,
                            zorder=1, linewidth=0)
        circle_edge = plt.Circle((pos[1], pos[0]), REGION_CONFIG['nw_radius_visual'],
                                 fill=False, edgecolor=STYLE_COLORS['nw_recycler_halo'],
                                 linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    else:
        # 常规回收中心（600km绿色圈）
        circle = plt.Circle((pos[1], pos[0]), REGION_CONFIG['normal_radius_visual'],
                            color=STYLE_COLORS['recycler_halo'], alpha=0.15, zorder=1, linewidth=0)
        circle_edge = plt.Circle((pos[1], pos[0]), REGION_CONFIG['normal_radius_visual'],
                                 fill=False, edgecolor=STYLE_COLORS['recycler_halo'],
                                 linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)

    ax.add_patch(circle)
    ax.add_patch(circle_edge)

    if is_nw_city(k):
        ax.scatter(pos[1], pos[0], c=STYLE_COLORS['recycler_fill'], marker='^', s=200,
                   edgecolors=STYLE_COLORS['nw_recycler_halo'], lw=2.0, zorder=20, label='_nolegend_')
    else:
        ax.scatter(pos[1], pos[0], c=STYLE_COLORS['recycler_fill'], marker='^', s=200,
                   edgecolors='white', lw=1.5, zorder=20, label='_nolegend_')
    add_smart_label(pos, city_name, '#006655', size=10, weight='bold', dy=-0.6)

# 电池工厂
for f in factories:
    pos = locations[f]
    ax.scatter(pos[1], pos[0], c=STYLE_COLORS['factory_fill'], marker='s', s=180,
               edgecolors='white', lw=1.5, zorder=15)
    add_smart_label(pos, f.replace('F_', ''), '#8B0000', size=10, weight='bold', dy=0.6)

# 图饰优化
legend_elements = [
    mlines.Line2D([], [], color=STYLE_COLORS['factory_fill'], marker='s', linestyle='None',
                  markersize=10, label='Battery Factory'),
    mlines.Line2D([], [], color=STYLE_COLORS['recycler_fill'], marker='^', linestyle='None',
                  markersize=10, label='Recycling Center'),
    mlines.Line2D([], [], color=STYLE_COLORS['recycler_fill'], marker='^', linestyle='None',
                  markersize=10, markeredgecolor=STYLE_COLORS['nw_recycler_halo'], markeredgewidth=2,
                  label='Northwest Recycler (1200km)'),
    mlines.Line2D([], [], color=STYLE_COLORS['market_fill'], marker='o', linestyle='None',
                  markersize=8, alpha=0.7, label='Market Demand'),
    mlines.Line2D([], [], color=STYLE_COLORS['recycler_halo'], marker='o', linestyle='None',
                  markersize=15, alpha=0.3, label='600km Service Radius'),
    mlines.Line2D([], [], color=STYLE_COLORS['nw_recycler_halo'], marker='o', linestyle='None',
                  markersize=30, alpha=0.3, label='1200km Northwest Service Radius'),
    mlines.Line2D([], [], color=STYLE_COLORS['fwd_line'], lw=2, label='Forward Flow (Proportional)'),
    mlines.Line2D([], [], color=STYLE_COLORS['rev_line'], lw=2, linestyle=':', label='Reverse Flow (Proportional)'),
]

leg = ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                frameon=True, fancybox=True, framealpha=0.9, edgecolor='#CCCCCC',
                bbox_to_anchor=(0.98, 0.02), title="Network Components", title_fontsize=11)
leg.get_title().set_fontweight('bold')

ax.set_title("Optimized Closed-Loop Supply Chain Network Structure",
             fontsize=16, fontweight='bold', pad=20, color='#333333')
ax.set_xlabel("Longitude (°E)", fontsize=12, labelpad=10)
ax.set_ylabel("Latitude (°N)", fontsize=12, labelpad=10)
ax.set_xlim(75, 135)
ax.set_ylim(15, 55)
ax.grid(True, linestyle=':', linewidth=0.5, color='#999999', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("50cities_optimized_vis_nw_unified.png", bbox_inches='tight', dpi=PLOT_CONFIG['savefig_dpi'])
plt.show()
