import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

# ==========================================
# 0. 全局配置 (学术风格设置)
# ==========================================
plt.rcParams.update({
    'font.family': 'Arial',  # 英文期刊首选无衬线字体
    'font.size': 10,  # 基础字号
    'axes.titlesize': 14,  # 标题字号
    'axes.labelsize': 12,  # 轴标签字号
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,  # 屏幕预览分辨率
    'savefig.dpi': 600,  # 出版级导出分辨率
    'mathtext.fontset': 'stix',  # 数学公式使用 Times 风格
})

# 学术配色方案 (Colorblind Safe & High Contrast)
COLORS = {
    'factory': '#D73027',  # 砖红
    'market': '#4575B4',  # 钢蓝
    'recycler': '#1A9850',  # 森林绿
    'fwd_flow': '#D73027',  # 正向流颜色
    'rev_flow': '#1A9850',  # 逆向流颜色
    'radius': '#1A9850',  # 半径圈颜色
    'gray': '#BDBDBD'  # 未选中/背景色
}

# ==========================================
# 1. 数据准备 (保持原逻辑，增强结构)
# ==========================================

# --- 基准参数 ---
NATIONAL_SALES_TOTAL = 12866000
TOTAL_RETIRED_BATTERY = 820000
UNIT_BATTERY_WEIGHT = 0.5
TOTAL_UNITS_NATIONAL = TOTAL_RETIRED_BATTERY / UNIT_BATTERY_WEIGHT

# 50城配置
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

# 50城坐标
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

# 回收中心候选
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

# 工厂
factory_config = [
    ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# --- 自动计算逻辑 ---
total_city_weight = sum([w for _, w in city_sales_weight])
city_sales_ratio = total_city_weight / len(city_sales_weight)
actual_50_sales_total = NATIONAL_SALES_TOTAL * city_sales_ratio

city_demand = {}
for city, weight in city_sales_weight:
    single_city_sales = actual_50_sales_total * (weight / total_city_weight)
    city_demand[city] = int(TOTAL_UNITS_NATIONAL * (single_city_sales / NATIONAL_SALES_TOTAL))

markets = [f"M_{city}" for city, _ in city_sales_weight]
factories = [f"F_{city}" for city, _ in factory_config]
candidates = [f"R_{city}" for city, _, _ in recycler_config]

locations = {}
for c, pos in factory_config: locations[f"F_{c}"] = pos
for c, _ in city_sales_weight: locations[f"M_{c}"] = city_coords[c]
for c, pos, _ in recycler_config: locations[f"R_{c}"] = pos

fixed_cost = {f"R_{c}": cost * 10000 for c, _, cost in recycler_config}
demand_base = {f"M_{c}": city_demand[c] for c, _ in city_sales_weight}
demand_uncertainty = {k: v * 0.2 for k, v in demand_base.items()}


# ==========================================
# 2. 模型构建与求解
# ==========================================
def get_dist(n1, n2):
    p1, p2 = locations[n1], locations[n2]
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * 100


prob = pulp.LpProblem("CLSC_50_Optimization", pulp.LpMinimize)

# 变量
x = pulp.LpVariable.dicts("Fwd", (factories, markets), 0, cat='Continuous')
z = pulp.LpVariable.dicts("Rev", (markets, candidates), 0, cat='Continuous')
y = pulp.LpVariable.dicts("Open", candidates, cat='Binary')
excess_e = pulp.LpVariable("ExcessE", 0, cat='Continuous')

# 参数
TRANS_COST = 1.6
CARBON_TAX = 65
CARBON_FACTOR = 0.0004
CARBON_CAP = 1500000
CAPACITY = 80000
MAX_REV_DIST = 600
GAMMA = 1.0
ALPHA = 0.28

# 表达式
cost_trans = pulp.lpSum([x[i][j] * get_dist(i, j) * TRANS_COST for i in factories for j in markets]) + \
             pulp.lpSum([z[j][k] * get_dist(j, k) * TRANS_COST for j in markets for k in candidates])

emission = pulp.lpSum([x[i][j] * get_dist(i, j) * CARBON_FACTOR for i in factories for j in markets]) + \
           pulp.lpSum([z[j][k] * get_dist(j, k) * CARBON_FACTOR for j in markets for k in candidates])

cost_fixed = pulp.lpSum([fixed_cost[k] * y[k] for k in candidates])
prob += excess_e >= emission - CARBON_CAP
prob += cost_fixed + cost_trans + excess_e * CARBON_TAX

# 约束
for j in markets:
    d_j = demand_base[j] + GAMMA * demand_uncertainty[j]
    prob += pulp.lpSum([x[i][j] for i in factories]) >= d_j
    prob += pulp.lpSum([z[j][k] for k in candidates]) >= demand_base[j] * ALPHA
    for k in candidates:
        if get_dist(j, k) > MAX_REV_DIST:
            prob += z[j][k] == 0

for k in candidates:
    prob += pulp.lpSum([z[j][k] for j in markets]) <= CAPACITY * y[k]

# 求解
print("正在使用 CBC 求解器进行优化...")
prob.solve(pulp.PULP_CBC_CMD(msg=False))
print(f"求解状态: {pulp.LpStatus[prob.status]}")

# ==========================================
# 3. 丰富的输出 (Rich Output)
# ==========================================
if pulp.LpStatus[prob.status] == 'Optimal':

    # --- KPI 计算 ---
    total_c = pulp.value(prob.objective)
    total_e = pulp.value(emission)
    built_k = [k for k in candidates if pulp.value(y[k]) > 0.5]

    # 计算平均运输距离 (加权)
    total_fwd_units = sum(pulp.value(x[i][j]) for i in factories for j in markets)
    total_fwd_km = sum(pulp.value(x[i][j]) * get_dist(i, j) for i in factories for j in markets)
    avg_fwd_dist = total_fwd_km / total_fwd_units if total_fwd_units else 0

    total_rev_units = sum(pulp.value(z[j][k]) for j in markets for k in candidates)
    total_rev_km = sum(pulp.value(z[j][k]) * get_dist(j, k) for j in markets for k in candidates)
    avg_rev_dist = total_rev_km / total_rev_units if total_rev_units else 0

    # --- 仪表盘输出 ---
    print("\n" + "=" * 60)
    print(f"{'OPTIMIZATION RESULTS DASHBOARD':^60}")
    print("=" * 60)

    kpi_data = {
        "Metric": ["Total Cost (CNY)", "Total Emissions (tCO2)", "Recyclers Built",
                   "Forward Flow (Units)", "Reverse Flow (Units)",
                   "Avg Fwd Dist (km)", "Avg Rev Dist (km)"],
        "Value": [f"{total_c:,.0f}", f"{total_e:,.2f}", f"{len(built_k)} / {len(candidates)}",
                  f"{total_fwd_units:,.0f}", f"{total_rev_units:,.0f}",
                  f"{avg_fwd_dist:.1f}", f"{avg_rev_dist:.1f}"]
    }
    df_kpi = pd.DataFrame(kpi_data)
    print(df_kpi.to_string(index=False))
    print("-" * 60)

    # --- 选址详情 ---
    print(f"\n[Selected Recyclers & Utilization]")
    recycler_stats = []
    for k in built_k:
        load = sum(pulp.value(z[j][k]) for j in markets)
        utilization = load / CAPACITY * 100
        city_name = k.replace('R_', '')
        fixed_c = fixed_cost[k]
        recycler_stats.append([city_name, f"{load:,.0f}", f"{utilization:.1f}%", f"{fixed_c / 10000:.0f}万"])

    df_rec = pd.DataFrame(recycler_stats, columns=["Location", "Load (Units)", "Util %", "Fixed Cost"])
    print(df_rec.to_string(index=False))

# ==========================================
# 4. 高水平期刊绘图 (Journal Quality Visualization)
# ==========================================
print("\n正在生成高分辨率网络图...")

fig, ax = plt.subplots(figsize=(14, 12))  # 黄金比例附近

# 4.1 背景与地图投影模拟
ax.set_aspect(1.15)  # 调整中国地图的纵横比视觉
ax.set_facecolor('white')  # 纯白背景适合论文
# 绘制经纬度网格（极淡）
ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.5, zorder=0)

# 4.2 数据归一化准备
max_flow_fwd = max([pulp.value(x[i][j]) for i in factories for j in markets] + [1])
max_flow_rev = max([pulp.value(z[j][k]) for j in markets for k in candidates] + [1])

# 4.3 绘制物流连线 (使用透明度表现层级)
# 正向物流 (F -> M)
for i in factories:
    for j in markets:
        val = pulp.value(x[i][j])
        if val > 100:  # 过滤极小流量
            p1, p2 = locations[i], locations[j]
            # 线宽与透明度非线性映射，突出主干道
            alpha = 0.2 + (val / max_flow_fwd) * 0.8
            lw = 0.5 + (val / max_flow_fwd) * 3.0
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], c=COLORS['fwd_flow'],
                    alpha=alpha, linewidth=lw, zorder=2)

# 逆向物流 (M -> R)
for j in markets:
    for k in built_k:
        val = pulp.value(z[j][k])
        if val > 10:
            p1, p2 = locations[j], locations[k]
            alpha = 0.3 + (val / max_flow_rev) * 0.7
            lw = 0.5 + (val / max_flow_rev) * 3.0
            # 使用虚线表示逆向
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], c=COLORS['rev_flow'],
                    alpha=alpha, linewidth=lw, linestyle=':', dashes=(1, 1), zorder=3)


# 4.4 绘制节点 (智能筛选标签)
def add_label(x, y, text, color, y_offset=0, size=9, weight='normal'):
    txt = ax.text(x, y + y_offset, text, fontsize=size, fontweight=weight, color=color,
                  ha='center', va='center', zorder=20)
    txt.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])


# 工厂 (方块)
for f in factories:
    pos = locations[f]
    ax.scatter(pos[1], pos[0], c=COLORS['factory'], marker='s', s=180, edgecolors='k', lw=1, zorder=10)
    add_label(pos[1], pos[0], f.replace('F_', ''), COLORS['factory'], 0.7, 10, 'bold')

# 回收中心 (三角 + 600km 覆盖圈)
for k in built_k:
    pos = locations[k]
    ax.scatter(pos[1], pos[0], c=COLORS['recycler'], marker='^', s=220, edgecolors='k', lw=1, zorder=15)
    add_label(pos[1], pos[0], k.replace('R_', ''), '#004d26', -0.7, 10, 'bold')
    # 绘制覆盖半径 (近似: 1度经纬度 ≈ 100km, 600km ≈ 6度)
    # 注意：这只是视觉示意，实际计算使用的是欧氏距离
    circle = plt.Circle((pos[1], pos[0]), 6.0, fill=True, color=COLORS['radius'],
                        alpha=0.05, linestyle='--', linewidth=0.5, ec=COLORS['radius'], zorder=1)
    ax.add_patch(circle)

# 市场 (圆形 - 智能显示Top市场)
# 找出销量前10的市场
sorted_markets = sorted(demand_base.items(), key=lambda item: item[1], reverse=True)
top_markets = [k for k, v in sorted_markets[:12]]  # 标注前12个

for m in markets:
    pos = locations[m]
    size = 30 + (demand_base[m] / max(demand_base.values())) * 120  # 大小随需求变化

    if m in top_markets:
        # 核心市场：深色，带标签
        ax.scatter(pos[1], pos[0], c=COLORS['market'], marker='o', s=size, edgecolors='white', lw=0.5, zorder=5)
        # 避免与工厂/回收站重叠的标签逻辑
        if m.replace('M_', '') not in [k.replace('R_', '') for k in built_k] and \
                m.replace('M_', '') not in [f.replace('F_', '') for f in factories]:
            add_label(pos[1], pos[0], m.replace('M_', ''), COLORS['market'], -0.5, 8)
    else:
        # 小市场：浅色，无标签，减少视觉噪音
        ax.scatter(pos[1], pos[0], c=COLORS['market'], marker='o', s=size, alpha=0.5, edgecolors='none', zorder=4)

# 4.5 专业的图例与装饰
# 自定义图例句柄
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

ax.legend(handles=legend_handles, loc='lower right', frameon=True, fancybox=False,
          edgecolor='black', fontsize=10, bbox_to_anchor=(0.98, 0.02))

# 标题与轴
total_cost_m = total_c / 1e8
ax.set_title(
    f"Optimized Closed-Loop Supply Chain Network (50 Cities)\nTotal Cost: {total_cost_m:.2f} Hundred Million CNY",
    pad=20, fontweight='bold')
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

# 坐标范围微调
x_vals = [pos[1] for pos in locations.values()]
y_vals = [pos[0] for pos in locations.values()]
ax.set_xlim(min(x_vals) - 2, max(x_vals) + 2)
ax.set_ylim(min(y_vals) - 2, max(y_vals) + 2)

# 去除顶部和右侧边框 (Despine)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("50cities_high_impact_plot.png", bbox_inches='tight')
print(f"✅ 高清图表已保存: 50cities_high_impact_plot.png (600 DPI)")
plt.show()
