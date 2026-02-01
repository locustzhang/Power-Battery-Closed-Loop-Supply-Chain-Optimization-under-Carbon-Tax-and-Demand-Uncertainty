#0125新增多城市数据的代码
import pulp
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据准备 (最终版：贴合2025工业事实+物流现实)
# ==========================================

locations = {
    # 工厂 (核心3个，贴合动力电池生产基地布局)
    'F_XiAn': (34.34, 108.94),
    'F_Changsha': (28.23, 112.94),
    'F_Shenzhen': (22.54, 114.05),

    # 市场 (Top12核心消费城市，2024销量占比≈50%)
    'M_Chengdu': (30.67, 104.06),
    'M_Hangzhou': (30.27, 120.15),
    'M_Shenzhen': (22.54, 114.05),
    'M_Shanghai': (31.23, 121.47),
    'M_Beijing': (39.90, 116.40),
    'M_Guangzhou': (23.13, 113.26),
    'M_Zhengzhou': (34.76, 113.65),
    'M_XiAn': (34.34, 108.94),
    'M_Chongqing': (29.56, 106.55),
    'M_Tianjin': (39.13, 117.20),
    'M_Wuhan': (30.59, 114.30),
    'M_Changsha': (28.23, 112.94),

    # 回收中心候选 (7个核心枢纽，贴合物流能级)
    'R_Hefei': (31.82, 117.22),
    'R_Zhengzhou': (34.76, 113.65),
    'R_Guiyang': (26.64, 106.63),
    'R_Changsha': (28.23, 112.94),
    'R_Wuhan': (30.59, 114.30),
    'R_Yibin': (28.77, 104.63),
    'R_Nanchang': (28.68, 115.86)
}

# 分类定义
factories = ['F_XiAn', 'F_Changsha', 'F_Shenzhen']
markets = ['M_Chengdu', 'M_Hangzhou', 'M_Shenzhen', 'M_Shanghai', 'M_Beijing',
           'M_Guangzhou', 'M_Zhengzhou', 'M_XiAn', 'M_Chongqing', 'M_Tianjin',
           'M_Wuhan', 'M_Changsha']
candidates = ['R_Hefei', 'R_Zhengzhou', 'R_Guiyang', 'R_Changsha',
              'R_Wuhan', 'R_Yibin', 'R_Nanchang']


# 经纬度转实际公里数（欧氏距离×100，贴合中国地理尺度）
def get_dist(n1, n2):
    pos1 = locations[n1]
    pos2 = locations[n2]
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * 100


# 需求基数（基于2024 NEV销量比例缩放，贴合Top12城市占比）
demand_base = {
    'M_Chengdu': 51600,
    'M_Hangzhou': 51200,
    'M_Shenzhen': 50100,
    'M_Shanghai': 49500,
    'M_Beijing': 48500,
    'M_Guangzhou': 46200,
    'M_Zhengzhou': 39600,
    'M_XiAn': 37600,
    'M_Chongqing': 36700,
    'M_Tianjin': 33300,
    'M_Wuhan': 35000,
    'M_Changsha': 31700
}
# 需求不确定性（20%波动，贴合市场实际）
demand_uncertainty = {k: v * 0.20 for k, v in demand_base.items()}

# 回收中心固定成本（贴合枢纽能级：核心枢纽微降，非核心微涨）
fixed_cost = {
    'R_Hefei': 58000000,  # 合肥（华东枢纽）：5800万（规模效应）
    'R_Zhengzhou': 53000000,  # 郑州（中原枢纽）：5300万（物流成本优势）
    'R_Guiyang': 50000000,  # 贵阳（西南枢纽）：5000万（西南低成本）
    'R_Changsha': 62000000,  # 长沙（湘粤枢纽）：6200万（低于南昌，贴合物流能级）
    'R_Wuhan': 58000000,  # 武汉（华中）：5800万（基准值）
    'R_Yibin': 70000000,  # 宜宾：7000万（成本最高，无优势）
    'R_Nanchang': 55000000  # 南昌（非核心）：5500万（微涨，体现能级差异）
}

# 回收中心容量（4万吨/年，贴合行业典型规模）
capacity = 40000

# ============ 核心参数（完全校准2025工业/物流现实） ============
trans_cost_per_km = 1.6  # 动力电池公路运输实际单价：1.2-1.8元/单位·公里
carbon_tax = 65  # 2025全国碳市场均价：62.36-68元/吨
carbon_factor = 0.0004  # 重卡碳排放×电池包重量折算：0.00029吨/吨·公里 → 0.0004吨/单位·公里
carbon_cap = 1500000  # 企业碳配额：150万吨（充足，贴合头部企业实际）
max_rev_dist = 600  # 逆向物流最优半径：600km（动力电池回收核心辐射圈）
# ==============================================================

# 鲁棒优化参数
GAMMA = 1.0  # 需求不确定性权重
alpha = 0.28  # 回收率：28%（贴合2025政策目标）

# ==========================================
# 2. 模型构建（最终版：添加物流半径约束）
# ==========================================

# 定义问题：最小化总成本
prob = pulp.LpProblem("EV_Battery_CLSC_Optimization_2025_Industrial_Reality", pulp.LpMinimize)

# 决策变量
x_vars = pulp.LpVariable.dicts("Flow_Fwd", (factories, markets), lowBound=0, cat='Continuous')  # 正向物流：工厂→市场
z_vars = pulp.LpVariable.dicts("Flow_Rev", (markets, candidates), lowBound=0, cat='Continuous')  # 逆向物流：市场→回收中心
y_vars = pulp.LpVariable.dicts("Open_Recycler", candidates, cat='Binary')  # 回收中心建设决策

# 成本/排放计算
total_transport_cost = pulp.LpAffineExpression()  # 总物流成本
total_emission = pulp.LpAffineExpression()  # 总碳排放量

# 正向物流成本&碳排放
for i in factories:
    for j in markets:
        dist = get_dist(i, j)
        total_transport_cost += x_vars[i][j] * dist * trans_cost_per_km
        total_emission += x_vars[i][j] * dist * carbon_factor

# 逆向物流成本&碳排放
for j in markets:
    for k in candidates:
        dist = get_dist(j, k)
        total_transport_cost += z_vars[j][k] * dist * trans_cost_per_km
        total_emission += z_vars[j][k] * dist * carbon_factor

# 固定成本
total_fixed_cost = pulp.lpSum([fixed_cost[k] * y_vars[k] for k in candidates])

# 碳税成本（超额排放惩罚）
excess_emission = pulp.LpVariable("Excess_Emission", lowBound=0, cat='Continuous')
prob += excess_emission >= total_emission - carbon_cap, "Carbon_Cap_Constraint"
carbon_cost = excess_emission * carbon_tax

# 目标函数：总成本=固定成本+物流成本+碳税成本
prob += total_fixed_cost + total_transport_cost + carbon_cost, "Total_Cost"

# ============ 约束条件（贴合工业现实） ============
# 1. 需求约束（鲁棒优化：覆盖需求+20%波动）
for j in markets:
    robust_demand = demand_base[j] + GAMMA * demand_uncertainty[j]
    prob += pulp.lpSum([x_vars[i][j] for i in factories]) >= robust_demand, f"Demand_{j}"

# 2. 回收约束（回收率≥28%，贴合政策目标）
for j in markets:
    prob += pulp.lpSum([z_vars[j][k] for k in candidates]) >= demand_base[j] * alpha, f"Recycle_{j}"

# 3. 容量约束（回收中心容量上限）
for k in candidates:
    prob += pulp.lpSum([z_vars[j][k] for j in markets]) <= capacity * y_vars[k], f"Cap_{k}"

# 4. 逆向物流半径约束（核心：贴合600km最优辐射圈）
for j in markets:
    for k in candidates:
        dist = get_dist(j, k)
        if dist > max_rev_dist:
            prob += z_vars[j][k] == 0, f"Rev_Dist_Limit_{j}_{k}"
# ==================================================

# ==========================================
# 3. 求解与结果分析（最终版）
# ==========================================

print("开始求解贴合工业现实的动力电池闭环供应链模型...")
# 求解器配置：关闭日志，加速求解
status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

# 输出求解状态
print(f"求解状态: {pulp.LpStatus[status]}")

if pulp.LpStatus[status] == 'Optimal':
    # 总成本
    total_cost = pulp.value(prob.objective)
    print(f"最小总成本: {total_cost:,.2f} 元")

    # 选址决策
    print("\n--- 选址决策（贴合物流枢纽能级） ---")
    built_recyclers = []
    for k in candidates:
        if pulp.value(y_vars[k]) > 0.5:
            built_recyclers.append(k)
            print(f" [√] 建设回收中心: {k} (固定成本: {fixed_cost[k]:,.0f} 元)")
        else:
            print(f" [x] 不建设: {k}")

    # 正向物流路径（流量>0）
    print("\n--- 正向物流路径 (工厂→市场，流量 > 0) ---")
    for i in factories:
        for j in markets:
            val = pulp.value(x_vars[i][j])
            if val > 0:
                print(f"  {i} → {j}: {val:.1f} 单位")

    # 逆向物流路径（流量>0）
    print("\n--- 逆向物流路径 (市场→回收中心，流量 > 0) ---")
    for j in markets:
        for k in candidates:
            val = pulp.value(z_vars[j][k])
            if val > 0:
                dist = get_dist(j, k)
                print(f"  {j} → {k}: {val:.1f} 单位 (距离: {dist:.0f}km)")

    # 碳排放分析
    print("\n--- 碳排放分析（贴合2025碳市场实际） ---")
    emission = pulp.value(total_emission)
    excess = pulp.value(excess_emission)
    print(f"总碳排放量: {emission:.2f} 吨")
    print(f"碳配额: {carbon_cap} 吨")
    if excess > 0:
        print(f"超额碳排放: {excess:.2f} 吨，需缴纳碳税: {excess * carbon_tax:,.2f} 元")
    else:
        print(f"未超额碳排放，剩余配额: {carbon_cap - emission:.2f} 吨，无需缴纳碳税")

    # 成本构成分析
    print("\n--- 成本构成分析 ---")
    transport_cost = pulp.value(total_transport_cost)
    fixed_cost_total = pulp.value(total_fixed_cost)
    carbon_cost_total = pulp.value(carbon_cost)
    print(f"固定成本: {fixed_cost_total:,.2f} 元 (占比: {fixed_cost_total / total_cost * 100:.1f}%)")
    print(f"物流成本: {transport_cost:,.2f} 元 (占比: {transport_cost / total_cost * 100:.1f}%)")
    print(f"碳税成本: {carbon_cost_total:,.2f} 元 (占比: {carbon_cost_total / total_cost * 100:.1f}%)")

else:
    print("模型未找到最优解（可能约束过紧），建议调整max_rev_dist或carbon_cap")

# ==========================================
# 4. 结果可视化（Science风格，贴合学术/工业展示）
# ==========================================
# ... (保留上面所有的 Import, 数据准备, 模型构建, 求解部分) ...

# ==========================================
# 4. 结果可视化（期刊发表级优化版）
# ==========================================
import matplotlib.patheffects as pe  # 用于文字描边，提升可读性
import matplotlib.lines as mlines  # 用于自定义图例

print("\n正在绘制期刊级网络图...")

# --- 1. 样式配置 (学术风格) ---
plt.style.use('default')  # 重置为默认，避免seaborn样式干扰
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # 优先使用Arial
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'figure.dpi': 300,
    'mathtext.fontset': 'stix',  # 数学公式字体类似Times
})

# 创建画布
fig, ax = plt.subplots(figsize=(14, 10))

# --- 2. 绘制基础地理框架 ---
# 设置近似Mercator投影的纵横比 (中国维度约为1.1-1.2)
ax.set_aspect(1.1)
# 淡化网格，仅作为参考
ax.grid(True, linestyle=':', color='gray', alpha=0.3, zorder=0)
# 设置背景色为极淡的灰色，突出前景
ax.set_facecolor('#FAFAFA')

# --- 3. 绘制物流连线 (Edges) ---
# 定义最大流量以便归一化线宽
max_flow_fwd = max([pulp.value(x_vars[i][j]) for i in factories for j in markets] + [1])
max_flow_rev = max([pulp.value(z_vars[j][k]) for j in markets for k in candidates] + [1])

# 正向物流 (F -> M): 实线，渐变色或纯色
for i in factories:
    for j in markets:
        val = pulp.value(x_vars[i][j])
        if val > 1e-3:  # 忽略极小值
            p1, p2 = locations[i], locations[j]
            # 线宽映射：最小1.0，最大4.5
            lw = 1.0 + (val / max_flow_fwd) * 3.5
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                    c='#B22222', alpha=0.5, linewidth=lw, solid_capstyle='round', zorder=2)

# 逆向物流 (M -> R): 虚线
for j in markets:
    for k in candidates:
        val = pulp.value(z_vars[j][k])
        if val > 1e-3:
            p1, p2 = locations[j], locations[k]
            lw = 1.0 + (val / max_flow_rev) * 3.5
            # 使用 dense dotted line
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                    c='#228B22', alpha=0.6, linewidth=lw, linestyle=(0, (3, 1.5)), zorder=3)


# --- 4. 绘制节点 (Nodes) ---
# 辅助函数：绘制带描边的文字
def plot_text(x, y, text, color, offset_y=0):
    txt = ax.text(x, y + offset_y, text, fontsize=10, fontweight='bold', color=color,
                  ha='center', va='center', zorder=20)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])


# 4.1 工厂 (Factories) - 红色方块
for f in factories:
    pos = locations[f]
    ax.scatter(pos[1], pos[0], c='#D73027', marker='s', s=250, edgecolors='k', linewidth=1.5, zorder=10)
    plot_text(pos[1], pos[0], f.replace('F_', ''), '#8B0000', offset_y=0.6)

# 4.2 市场 (Markets) - 蓝色圆圈
for m in markets:
    pos = locations[m]
    # 市场大小略微区别，或者保持一致
    ax.scatter(pos[1], pos[0], c='#4575B4', marker='o', s=180, edgecolors='white', linewidth=1.5, zorder=10)
    plot_text(pos[1], pos[0], m.replace('M_', ''), '#2C5BB4', offset_y=-0.5)

# 4.3 回收中心 (Recyclers) - 绿色三角 (选中) / 灰色X (未选)
built_count = 0
for k in candidates:
    pos = locations[k]
    is_built = pulp.value(y_vars[k]) > 0.5
    if is_built:
        built_count += 1
        ax.scatter(pos[1], pos[0], c='#1A9850', marker='^', s=300, edgecolors='k', linewidth=1.5, zorder=15)
        plot_text(pos[1], pos[0], k.replace('R_', ''), '#006400', offset_y=0.7)
    else:
        ax.scatter(pos[1], pos[0], c='#BDBDBD', marker='X', s=120, linewidth=2, zorder=5)
        # 未选中的仅显示较小的灰色文字
        txt = ax.text(pos[1], pos[0] - 0.4, k.replace('R_', ''), fontsize=8, color='gray', ha='center', zorder=5)
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

# --- 5. 图例与装饰 (关键优化) ---

# 构建自定义图例句柄 (更加专业)
legend_elements = [
    # 节点分类
    mlines.Line2D([], [], color='#D73027', marker='s', linestyle='None', markersize=10, label='Battery Factory'),
    mlines.Line2D([], [], color='#4575B4', marker='o', linestyle='None', markersize=9, label='Consumer Market'),
    mlines.Line2D([], [], color='#1A9850', marker='^', linestyle='None', markersize=11, label='Selected Recycler'),
    mlines.Line2D([], [], color='#BDBDBD', marker='X', linestyle='None', markersize=8, label='Candidate (Not Built)'),

    # 空行分割
    mlines.Line2D([], [], color='none', label=''),

    # 连线含义
    mlines.Line2D([], [], color='#B22222', linewidth=2, label='Forward Flow (Products)'),
    mlines.Line2D([], [], color='#228B22', linewidth=2, linestyle='--', label='Reverse Flow (Used Batteries)'),

    # 流量粗细说明
    mlines.Line2D([], [], color='gray', linewidth=1, label='Low Volume'),
    mlines.Line2D([], [], color='gray', linewidth=4, label='High Volume'),
]

ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=False,
          edgecolor='black', shadow=False, fontsize=10, bbox_to_anchor=(1.0, 0.02))

# 标题与标签
total_cost_m = pulp.value(prob.objective) / 1e6 if pulp.LpStatus[status] == 'Optimal' else 0
title_str = (f"Optimized CLSC Network for EV Batteries (2025 Scenario)\n"
             f"Total Cost: ¥{total_cost_m:.2f} Million | Recyclers Established: {built_count}/7")
ax.set_title(title_str, pad=20, fontweight='bold')
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

# 坐标轴范围微调 (留出边距)
x_vals = [pos[1] for pos in locations.values()]
y_vals = [pos[0] for pos in locations.values()]
ax.set_xlim(min(x_vals) - 2, max(x_vals) + 2)
ax.set_ylim(min(y_vals) - 2, max(y_vals) + 2)

# 移除上方和右侧的边框线 (Despine)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("paper_ready_network_plot.png", dpi=600, bbox_inches='tight')
print("图片已保存: paper_ready_network_plot.png (600 DPI - 期刊高标准)")
plt.show()