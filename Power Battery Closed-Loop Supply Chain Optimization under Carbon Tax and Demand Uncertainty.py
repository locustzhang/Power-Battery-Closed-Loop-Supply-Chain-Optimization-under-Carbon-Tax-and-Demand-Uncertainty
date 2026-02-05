import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

# ==========================================
# 0. 全局配置（完全保留原始样式）
# ==========================================
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 600,
})

COLORS = {
    'factory': '#D73027', 'market': '#4575B4', 'recycler': '#1A9850',
    'fwd_flow': '#D73027', 'rev_flow': '#1A9850', 'radius': '#1A9850', 'gray': '#BDBDBD'
}

# ==========================================
# 1. 数据准备（严格匹配论文表1-表3，核心修正：直接写入论文校准后的需求基数）
# ==========================================
# 论文表1 校准后的50城市需求基数（单位：标准化电池包单位）- 直接替换原自动计算逻辑
city_demand = {
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

# 城市经纬度（完全保留原始）
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

# 回收中心配置（论文表2：固定成本直接取校准值，单位：万元 → 转换为元）
recycler_config = [
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

# 工厂配置（完全保留原始）
factory_config = [
    ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# 构建位置字典（完全保留原始逻辑）
locations = {}
for c, pos in factory_config: locations[f"F_{c}"] = pos
for c in city_demand.keys(): locations[f"M_{c}"] = city_coords[c]
for c, pos, _ in recycler_config: locations[f"R_{c}"] = pos

# 基础参数赋值（严格匹配论文）
fixed_cost = {f"R_{c}": cost * 10000 for c, _, cost in recycler_config}  # 万元→元
demand_base = {f"M_{c}": city_demand[c] for c in city_demand.keys()}
demand_uncertainty = {k: v * 0.2 for k, v in demand_base.items()}  # 保留20%需求波动

# ==========================================
# 核心参数设定（100%匹配论文表3 运输成本、碳排放与政策参数）
# 修正点1：碳配额、回收中心处理能力 匹配论文数值
# 修正点2：保留所有原始格式，仅修改错误参数值
# ==========================================
TRANS_COST = 1.6  # 元/单位·km（论文中高位值）
CARBON_TAX = 65  # 元/吨CO2（论文校准值）
CARBON_FACTOR_FWD = 0.004  # 正向碳排放因子 吨CO2/单位·km（论文明确值）
CARBON_FACTOR_REV = 0.025  # 逆向碳排放因子 吨CO2/单位·km（论文明确值）
CARBON_CAP = 1500000  # 碳配额 150万吨CO2（修正：原150000→1500000，匹配论文）
CAPACITY = 80000  # 单厂处理能力 80000单位/年（修正：原8000→80000，匹配论文假设）
MAX_REV_DIST = 600  # 逆向物流半径 600km（论文明确值）
GAMMA = 1.0  # 需求不确定性系数（保留原始，匹配模型鲁棒性假设）
ALPHA = 0.28  # 回收率28%（保留模型基准，匹配论文）


# ==========================================
# 修正点3：完善经纬度→实际公里数换算（符合模型"运输距离决定成本/碳排放"的假设）
# 采用Haversine公式，准确计算两点间球面距离（公里）
# ==========================================
def haversine_dist(n1, n2):
    """
    输入两个节点名称，计算其经纬度对应的实际地面距离（公里）
    解决原直接差值*100的近似误差问题，匹配模型中距离相关约束
    """
    lat1, lon1 = locations[n1]
    lat2, lon2 = locations[n2]

    # 转弧度制
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371  # 地球平均半径（公里）
    return R * c


# 替换原简单距离计算函数，保持后续调用逻辑不变
get_dist = haversine_dist

# ==========================================
# 2. 模型构建（完全保留原始逻辑，仅调用修正后参数与距离函数）
# ==========================================
# 定义问题与变量
prob = pulp.LpProblem("CLSC_Deep_Analysis", pulp.LpMinimize)
factories = [f"F_{c}" for c, _ in factory_config]
markets = [f"M_{c}" for c in city_demand.keys()]
candidates = [f"R_{c}" for c, _, _ in recycler_config]

x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0)  # 正向物流
z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0)  # 逆向物流
y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')  # 回收中心开启
excess_e = pulp.LpVariable("ExcessE", 0)  # 超额碳排放

# 目标函数：总成本（固定成本+运输成本+碳税）
cost_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * TRANS_COST for i in factories for j in markets])
cost_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * TRANS_COST for j in markets for k in candidates])
cost_fix = pulp.lpSum([fixed_cost[k] * y[k] for k in candidates])
emit_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * CARBON_FACTOR_FWD for i in factories for j in markets])
emit_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * CARBON_FACTOR_REV for j in markets for k in candidates])
prob += cost_fix + cost_fwd + cost_rev + excess_e * CARBON_TAX

# 约束条件（完全匹配数学模型约束，仅调用修正后参数）
prob += excess_e >= (emit_fwd + emit_rev) - CARBON_CAP  # 碳配额约束
for j in markets:
    prob += pulp.lpSum([x[i][j] for i in factories]) >= demand_base[j] + GAMMA * demand_uncertainty[j]  # 需求满足
    prob += pulp.lpSum([z[j][k] for k in candidates]) >= demand_base[j] * ALPHA  # 回收率约束
    for k in candidates:
        if get_dist(j, k) > MAX_REV_DIST: prob += z[j][k] == 0  # 逆向半径约束（600km）
for k in candidates:
    prob += pulp.lpSum([z[j][k] for j in markets]) <= CAPACITY * y[k]  # 处理能力约束（80000单位）

# 求解模型
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# ==========================================
# 3. 论文数据输出（完全保留原始格式，仅输出修正后参数结果）
# ==========================================
if pulp.LpStatus[prob.status] == 'Optimal':
    tc = pulp.value(prob.objective)
    built_k = [k for k in candidates if pulp.value(y[k]) > 0.5]
    total_qty_fwd = sum(pulp.value(x[i][j]) for i in factories for j in markets)
    total_qty_rev = sum(pulp.value(z[j][k]) for j in markets for k in candidates)

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
    print(
        f" - Forward Emission Cont. : {pulp.value(emit_fwd):,.2f} tCO2 ({pulp.value(emit_fwd) / pulp.value(emit_fwd + emit_rev) * 100:>.2f}%)")
    print(
        f" - Reverse Emission Cont. : {pulp.value(emit_rev):,.2f} tCO2 ({pulp.value(emit_rev) / pulp.value(emit_fwd + emit_rev) * 100:>.2f}%)")
    # 修正：先提取数值再比较，避免直接对LpAffineExpression使用min()
    total_emission = pulp.value(emit_fwd + emit_rev)
    used_quota = min(total_emission, CARBON_CAP)
    print(f" - Carbon Quota Used      : {used_quota:,.2f} tCO2 / {CARBON_CAP:,.2f} tCO2")
    print("-" * 90)

    avg_f_dist = sum(pulp.value(x[i][j]) * get_dist(i, j) for i in factories for j in markets) / total_qty_fwd
    avg_r_dist = sum(pulp.value(z[j][k]) * get_dist(j, k) for j in markets for k in candidates) / total_qty_rev
    print(f" - Avg Transport Distance (Fwd): {avg_f_dist:>.2f} km")
    print(f" - Avg Transport Distance (Rev): {avg_r_dist:>.2f} km")
    print("-" * 90)

    print(f"[D] HUB OPERATIONAL STATUS")
    print(f"{'Recycling Center':<20} | {'Processed Qty':<15} | {'Utilization (%)':<15} | {'Fixed Cost (10^4 CNY)'}")
    for k in built_k:
        load = sum(pulp.value(z[j][k]) for j in markets)
        print(
            f"{k.replace('R_', ''):<20} | {load:<15,.0f} | {load / CAPACITY * 100:<15.2f} | {fixed_cost[k] / 10000:<,.0f}")
    print("=" * 90)

# ==========================================
# 4. 高水平期刊绘图（完全保留原始样式、配色、布局，无任何修改）
# ==========================================
print("\n正在生成出版级高分辨率网络图...")

# 设置全局绘图风格
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

# 定义高级配色 (Science Style)
STYLE_COLORS = {
    'factory_fill': '#E64B35',  # 砖红
    'factory_edge': '#3C5488',  # 深蓝边框
    'recycler_fill': '#00A087',  # 翡翠绿
    'recycler_halo': '#00A087',  # 浅绿光晕
    'market_fill': '#4DBBD5',  # 天蓝
    'fwd_line': '#E64B35',  # 正向物流色
    'rev_line': '#00A087',  # 逆向物流色
    'text': '#333333'  # 深灰字体
}

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_aspect(1.18)

# 计算最大流量用于归一化线宽
max_flow_fwd = max([pulp.value(x[i][j]) for i in factories for j in markets] + [1])
max_flow_rev = max([pulp.value(z[j][k]) for j in markets for k in candidates] + [1])
max_demand = max(demand_base.values())

# --- 1. 绘制物流网络 (线条) ---
# 1.1 正向物流 (实线)
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

# 1.2 逆向物流 (虚线)
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


# --- 2. 绘制节点 (气泡与图标) ---
# 标签辅助函数 (带白色描边，防重叠)
def add_smart_label(pos, text, color, size=9, weight='bold', dy=0):
    txt = ax.text(pos[1], pos[0] + dy, text,
                  fontsize=size, fontweight=weight, color=color,
                  ha='center', va='center', zorder=50)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.9)])


# 2.1 消费市场 (气泡图)
built_cities = [k.replace('R_', '') for k in built_k]
factory_cities = [f.replace('F_', '') for f in factories]
for m in markets:
    city_name = m.replace('M_', '')
    pos = locations[m]
    size = 20 + (demand_base[m] / max_demand) * 150
    if city_name not in built_cities and city_name not in factory_cities:
        ax.scatter(pos[1], pos[0], s=size, c=STYLE_COLORS['market_fill'],
                   alpha=0.6, edgecolors='white', lw=0.5, zorder=4)
        if demand_base[m] > sorted(demand_base.values(), reverse=True)[12]:
            add_smart_label(pos, city_name, STYLE_COLORS['text'], size=8, weight='normal', dy=-0.4)

# 2.2 回收中心 (带辐射范围)
for k in built_k:
    pos = locations[k]
    # 辐射半径 (对应600km物流半径，绘图视觉比例保留原始)
    circle = plt.Circle((pos[1], pos[0]), 6.0, color=STYLE_COLORS['recycler_halo'],
                        alpha=0.15, zorder=1, linewidth=0)
    ax.add_patch(circle)
    circle_edge = plt.Circle((pos[1], pos[0]), 6.0, fill=False, edgecolor=STYLE_COLORS['recycler_halo'],
                             linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
    ax.add_patch(circle_edge)
    # 节点图标
    ax.scatter(pos[1], pos[0], c=STYLE_COLORS['recycler_fill'], marker='^', s=200,
               edgecolors='white', lw=1.5, zorder=20, label='_nolegend_')
    add_smart_label(pos, k.replace('R_', ''), '#006655', size=10, weight='bold', dy=-0.6)

# 2.3 电池工厂 (方块)
for f in factories:
    pos = locations[f]
    ax.scatter(pos[1], pos[0], c=STYLE_COLORS['factory_fill'], marker='s', s=180,
               edgecolors='white', lw=1.5, zorder=15)
    add_smart_label(pos, f.replace('F_', ''), '#8B0000', size=10, weight='bold', dy=0.6)

# --- 3. 图饰与布局优化 ---
# 自定义图例
legend_elements = [
    mlines.Line2D([], [], color=STYLE_COLORS['factory_fill'], marker='s', linestyle='None',
                  markersize=10, label='Battery Factory'),
    mlines.Line2D([], [], color=STYLE_COLORS['recycler_fill'], marker='^', linestyle='None',
                  markersize=10, label='Recycling Center'),
    mlines.Line2D([], [], color=STYLE_COLORS['market_fill'], marker='o', linestyle='None',
                  markersize=8, alpha=0.7, label='Market Demand'),
    mlines.Line2D([], [], color=STYLE_COLORS['recycler_halo'], marker='o', linestyle='None',
                  markersize=15, alpha=0.3, label='600km Service Radius'),
    mlines.Line2D([], [], color=STYLE_COLORS['fwd_line'], lw=2, label='Forward Flow (Proportional)'),
    mlines.Line2D([], [], color=STYLE_COLORS['rev_line'], lw=2, linestyle=':', label='Reverse Flow (Proportional)'),
]

leg = ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                frameon=True, fancybox=True, framealpha=0.9, edgecolor='#CCCCCC',
                bbox_to_anchor=(0.98, 0.02), title="Network Components", title_fontsize=11)
leg.get_title().set_fontweight('bold')

# 标题与坐标轴
ax.set_title("Optimized Closed-Loop Supply Chain Network Structure (50 Cities)",
             fontsize=16, fontweight='bold', pad=20, color='#333333')
ax.set_xlabel("Longitude (°E)", fontsize=12, labelpad=10)
ax.set_ylabel("Latitude (°N)", fontsize=12, labelpad=10)

# 显示范围
ax.set_xlim(80, 135)
ax.set_ylim(15, 55)

# 网格与边框
ax.grid(True, linestyle=':', linewidth=0.5, color='#999999', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("50cities_optimized_vis.png", bbox_inches='tight', dpi=600)
plt.show()
