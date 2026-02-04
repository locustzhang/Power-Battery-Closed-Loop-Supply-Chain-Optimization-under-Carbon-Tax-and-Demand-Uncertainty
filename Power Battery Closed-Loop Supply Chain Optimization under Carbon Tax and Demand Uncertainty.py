import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

# ==========================================
# 0. 全局配置
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
# 1. 数据准备
# ==========================================
NATIONAL_SALES_TOTAL = 12866000
TOTAL_RETIRED_BATTERY = 820000
UNIT_BATTERY_WEIGHT = 0.5
TOTAL_UNITS_NATIONAL = TOTAL_RETIRED_BATTERY / UNIT_BATTERY_WEIGHT

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

recycler_config = [
    ("Hefei", (31.82, 117.22), 5800), ("Zhengzhou", (34.76, 113.65), 5300),
    ("Guiyang", (26.64, 106.63), 5000), ("Changsha", (28.23, 112.94), 6200),
    ("Wuhan", (30.59, 114.30), 5800), ("Yibin", (28.77, 104.63), 7000),
    ("Nanchang", (28.68, 115.86), 5500), ("Xian", (34.34, 108.94), 5600),
    ("Tianjin", (39.13, 117.20), 5700), ("Nanjing", (32.05, 118.78), 5900),
    ("Hangzhou", (30.27, 120.15), 6000), ("Changchun", (43.88, 125.32), 4800),
    ("Nanning", (22.82, 108.32), 5200), ("Shenzhen", (22.54, 114.05), 6500),
    ("Qingdao", (36.07, 120.38), 5400), ("Haerbin", (45.80, 126.53), 4600),
    ("Fuzhou", (26.08, 119.30), 5100), ("Xiamen", (24.48, 118.08), 5300),
    ("Kunming", (25.04, 102.71), 4900), ("Wulumuqi", (43.83, 87.62), 4700),
    ("Haikou", (20.02, 110.35), 5000), ("Shenyang", (41.80, 123.43), 4900)
]

factory_config = [
    ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# --- 自动计算 ---
total_city_weight = sum([w for _, w in city_sales_weight])
actual_50_sales_total = NATIONAL_SALES_TOTAL * (total_city_weight / len(city_sales_weight))
city_demand = {}
for city, weight in city_sales_weight:
    single_city_sales = actual_50_sales_total * (weight / total_city_weight)
    city_demand[city] = int(TOTAL_UNITS_NATIONAL * (single_city_sales / NATIONAL_SALES_TOTAL))

locations = {}
for c, pos in factory_config: locations[f"F_{c}"] = pos
for c, _ in city_sales_weight: locations[f"M_{c}"] = city_coords[c]
for c, pos, _ in recycler_config: locations[f"R_{c}"] = pos

# --- 参数设定 (维持原始逻辑) ---
fixed_cost = {f"R_{c}": cost * 3000 for c, _, cost in recycler_config}
demand_base = {f"M_{c}": city_demand[c] for c, _ in city_sales_weight}
demand_uncertainty = {k: v * 0.2 for k, v in demand_base.items()}


# ==========================================
# 2. 模型构建
# ==========================================
def get_dist(n1, n2):
    p1, p2 = locations[n1], locations[n2]
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * 100


prob = pulp.LpProblem("CLSC_Deep_Analysis", pulp.LpMinimize)
factories, markets, candidates = [f"F_{c}" for c, _ in factory_config], [f"M_{c}" for c, _ in city_sales_weight], [
    f"R_{c}" for c, _, _ in recycler_config]
x, z, y, excess_e = pulp.LpVariable.dicts("Fwd", (factories, markets), 0), pulp.LpVariable.dicts("Rev",
                                                                                                 (markets, candidates),
                                                                                                 0), pulp.LpVariable.dicts(
    "Open", candidates, cat='Binary'), pulp.LpVariable("ExcessE", 0)

TRANS_COST, CARBON_TAX, CARBON_FACTOR_FWD, CARBON_FACTOR_REV, CARBON_CAP, CAPACITY, MAX_REV_DIST, GAMMA, ALPHA = 1.6, 65, 0.0004, 0.0030, 100000, 80000, 600, 1.0, 0.28

cost_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * TRANS_COST for i in factories for j in markets])
cost_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * TRANS_COST for j in markets for k in candidates])
cost_fix = pulp.lpSum([fixed_cost[k] * y[k] for k in candidates])
emit_fwd = pulp.lpSum([x[i][j] * get_dist(i, j) * CARBON_FACTOR_FWD for i in factories for j in markets])
emit_rev = pulp.lpSum([z[j][k] * get_dist(j, k) * CARBON_FACTOR_REV for j in markets for k in candidates])
prob += cost_fix + cost_fwd + cost_rev + excess_e * CARBON_TAX
prob += excess_e >= (emit_fwd + emit_rev) - CARBON_CAP

for j in markets:
    prob += pulp.lpSum([x[i][j] for i in factories]) >= demand_base[j] + GAMMA * demand_uncertainty[j]
    prob += pulp.lpSum([z[j][k] for k in candidates]) >= demand_base[j] * ALPHA
    for k in candidates:
        if get_dist(j, k) > MAX_REV_DIST: prob += z[j][k] == 0
for k in candidates:
    prob += pulp.lpSum([z[j][k] for j in markets]) <= CAPACITY * y[k]

prob.solve(pulp.PULP_CBC_CMD(msg=False))

# ==========================================
# 3. 论文数据输出 (Deep Dashboard)
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
    print("-" * 90)

    print(f"[C] NETWORK EFFICIENCY")
    avg_f_dist = sum(pulp.value(x[i][j]) * get_dist(i, j) for i in factories for j in markets) / total_qty_fwd
    avg_r_dist = sum(pulp.value(z[j][k]) * get_dist(j, k) for j in markets for k in candidates) / total_qty_rev
    print(f" - Avg Transport Distance (Fwd): {avg_f_dist:>.2f} km")
    print(f" - Avg Transport Distance (Rev): {avg_r_dist:>.2f} km")
    print("-" * 90)

    print(f"[D] HUB OPERATIONAL STATUS")
    print(f"{'Recycling Center':<20} | {'Processed Qty':<15} | {'Utilization (%)':<15} | {'Fixed Cost'}")
    for k in built_k:
        load = sum(pulp.value(z[j][k]) for j in markets)
        print(f"{k.replace('R_', ''):<20} | {load:<15,.0f} | {load / CAPACITY * 100:<15.2f} | {fixed_cost[k]:,.0f}")
    print("=" * 90)

# ==========================================
# 4. 高水平期刊绘图 (严格维持原始逻辑)
# ==========================================
print("\n正在生成高分辨率网络图...")
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_aspect(1.15)
ax.set_facecolor('white')
ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.5, zorder=0)

max_flow_fwd = max([pulp.value(x[i][j]) for i in factories for j in markets] + [1])
max_flow_rev = max([pulp.value(z[j][k]) for j in markets for k in candidates] + [1])

# 绘制正向物流 (实线)
for i in factories:
    for j in markets:
        val = pulp.value(x[i][j])
        if val > 100:
            p1, p2 = locations[i], locations[j]
            alpha = 0.2 + (val / max_flow_fwd) * 0.8
            lw = 0.5 + (val / max_flow_fwd) * 3.0
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], c=COLORS['fwd_flow'], alpha=alpha, linewidth=lw, zorder=2)

# 绘制逆向物流 (虚线)
for j in markets:
    for k in built_k:
        val = pulp.value(z[j][k])
        if val > 10:
            p1, p2 = locations[j], locations[k]
            alpha = 0.3 + (val / max_flow_rev) * 0.7
            lw = 0.5 + (val / max_flow_rev) * 3.0
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], c=COLORS['rev_flow'], alpha=alpha, linewidth=lw, linestyle=':',
                    dashes=(1, 1), zorder=3)


# 节点绘制函数
def add_label(x, y, text, color, y_offset=0, size=9, weight='normal'):
    txt = ax.text(x, y + y_offset, text, fontsize=size, fontweight=weight, color=color, ha='center', va='center',
                  zorder=20)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])


# 绘制工厂
for f in factories:
    pos = locations[f]
    ax.scatter(pos[1], pos[0], c=COLORS['factory'], marker='s', s=180, edgecolors='k', lw=1, zorder=10)
    add_label(pos[1], pos[0], f.replace('F_', ''), COLORS['factory'], 0.7, 10, 'bold')

# 绘制回收中心及辐射半径
for k in built_k:
    pos = locations[k]
    ax.scatter(pos[1], pos[0], c=COLORS['recycler'], marker='^', s=220, edgecolors='k', lw=1, zorder=15)
    add_label(pos[1], pos[0], k.replace('R_', ''), '#004d26', -0.7, 10, 'bold')
    circle = plt.Circle((pos[1], pos[0]), 6.0, fill=True, color=COLORS['radius'], alpha=0.05, linestyle='--',
                        linewidth=0.5, ec=COLORS['radius'], zorder=1)
    ax.add_patch(circle)

# 绘制市场 (Top 12)
sorted_markets = sorted(demand_base.items(), key=lambda item: item[1], reverse=True)
top_markets = [k for k, v in sorted_markets[:12]]
for m in markets:
    pos = locations[m]
    size = 30 + (demand_base[m] / max(demand_base.values())) * 120
    if m in top_markets:
        ax.scatter(pos[1], pos[0], c=COLORS['market'], marker='o', s=size, edgecolors='white', lw=0.5, zorder=5)
        if m.replace('M_', '') not in [k.replace('R_', '') for k in built_k] and m.replace('M_', '') not in [
            f.replace('F_', '') for f in factories]:
            add_label(pos[1], pos[0], m.replace('M_', ''), COLORS['market'], -0.5, 8)
    else:
        ax.scatter(pos[1], pos[0], c=COLORS['market'], marker='o', s=size, alpha=0.5, edgecolors='none', zorder=4)

# 图例与装饰
legend_handles = [
    mlines.Line2D([], [], color=COLORS['factory'], marker='s', linestyle='None', markersize=10,
                  label='Battery Factory'),
    mlines.Line2D([], [], color=COLORS['market'], marker='o', linestyle='None', markersize=8, label='Consumer Market'),
    mlines.Line2D([], [], color=COLORS['recycler'], marker='^', linestyle='None', markersize=10,
                  label='Established Recycler'),
    mlines.Line2D([], [], color=COLORS['radius'], marker='o', linestyle='None', markersize=15, alpha=0.2,
                  label='600km Service Radius'),
    mlines.Line2D([], [], color=COLORS['fwd_flow'], lw=2, label='Forward Logistics'),
    mlines.Line2D([], [], color=COLORS['rev_flow'], lw=2, linestyle=':', label='Reverse Logistics'),
]
ax.legend(handles=legend_handles, loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=10,
          bbox_to_anchor=(0.98, 0.02))

ax.set_title(f"Optimized Closed-Loop Supply Chain Network (50 Cities)\nTotal Cost Analysis with Carbon Tax Mechanism",
             pad=20, fontweight='bold')
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
plt.tight_layout()
plt.savefig("50cities_high_impact_plot.png", bbox_inches='tight')
plt.show()
