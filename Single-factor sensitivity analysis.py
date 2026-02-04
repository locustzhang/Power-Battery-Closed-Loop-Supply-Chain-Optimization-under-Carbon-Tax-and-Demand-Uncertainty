import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import warnings

# 忽略无关警告，保持输出整洁
warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局配置 (期刊级格式标准，优化绘图样式)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif', # 保持Times New Roman字体
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'mathtext.fontset': 'stix', # LaTeX风格数学字体
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False, # 移除顶部边框
    'axes.spines.right': False, # 移除右侧边框
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'figure.constrained_layout.use': True, # 自动调整子图布局
})

# 提前定义 LaTex 格式常量（解决 f-string 反斜杠问题）
LATEX_BF_START = r"$\bf{"
LATEX_BF_END = r"}$"
LATEX_CN = r"CNY"
LATEX_CO2 = r"CO$_2$"

# 期刊级配色方案 (更新为更具对比度和专业性的颜色)
COLORS = {
    'primary_line': '#00468B',   # 主线蓝色
    'primary_fill': '#A6CEE3',   # 浅蓝填充
    'secondary_line': '#ED0000', # 次线红色
    'secondary_fill': '#FB9A99', # 浅红填充
    'tertiary_line': '#009944',  # 第三线绿色
    'tertiary_fill': '#C7E9C0',  # 浅绿填充
    'quaternary_line': '#7C007C',# 第四线紫色
    'quaternary_fill': '#DFC5DA',# 浅紫填充
    'bar_fill': '#B0C4DE',       # 柱状图填充色 (LightSteelBlue)
    'bar_edge': '#6A809A',       # 柱状图边框色
    'grid': '#E0E0E0',           # 网格线颜色
    'text_color': '#333333',     # 文本颜色
    'baseline_marker': '#FFD700' # 基准点星形标记颜色 (Gold)
}


# 单位转换因子 (便于期刊级标注，保留旧代码定义)
COST_FACTOR = 1e4  # 总成本转换为 10^4 CNY
CAPACITY_FACTOR = 1e4  # 回收中心容量转换为 10^4 units/yr
CARBON_CAP_FACTOR = 1e4  # 碳容量转换为 10^4 tons CO2 (调整以匹配绘图x轴单位)

# ==========================================
# 1. 数据准备 (50城市全量数据，对齐数学模型+保留旧代码结构)
# ==========================================
def prepare_50cities_data():
    """准备50城市CLSC模型全量数据，返回结构化结果，对齐数学模型参数"""
    # 基础配置参数 (数学模型基准值)
    NATIONAL_SALES_TOTAL = 12866000
    TOTAL_RETIRED_BATTERY = 820000
    UNIT_BATTERY_WEIGHT = 0.5
    TOTAL_UNITS_NATIONAL = TOTAL_RETIRED_BATTERY / UNIT_BATTERY_WEIGHT

    # 50城市销量权重 (完整保留)
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

    # 50城市坐标 (修正字典语法：所有键值对用 : 分隔，修复错误格式)
    city_coords = {
        "Chengdu": (30.67, 104.06),
        "Hangzhou": (30.27, 120.15),
        "Shenzhen": (22.54, 114.05),
        "Shanghai": (31.23, 121.47),
        "Beijing": (39.90, 116.40),
        "Guangzhou": (23.13, 113.26),
        "Zhengzhou": (34.76, 113.65),
        "Chongqing": (29.56, 106.55),
        "XiAn": (34.34, 108.94),
        "Tianjin": (39.13, 117.20),
        "Wuhan": (30.59, 114.30),
        "Suzhou": (31.30, 120.58),
        "Hefei": (31.82, 117.22),
        "Wuxi": (31.57, 120.30),
        "Ningbo": (29.82, 121.55),
        "Dongguan": (23.05, 113.75),
        "Nanjing": (32.05, 118.78),
        "Changsha": (28.23, 112.94),
        "Wenzhou": (28.00, 120.70),
        "Shijiazhuang": (38.04, 114.51),
        "Jinan": (36.65, 117.12),
        "Foshan": (23.02, 113.12),
        "Qingdao": (36.07, 120.38),
        "Changchun": (43.88, 125.32),
        "Shenyang": (41.80, 123.43),
        "Nanning": (22.82, 108.32),
        "Taiyuan": (37.87, 112.55),
        "Kunming": (25.04, 102.71),
        "Linyi": (35.05, 118.35),
        "Taizhou": (28.66, 121.42),
        "Jinhua": (29.08, 119.65),
        "Xuzhou": (34.26, 117.28),
        "Haikou": (20.02, 110.35),
        "Jining": (35.42, 116.59),
        "Xiamen": (24.48, 118.08),
        "Baoding": (38.87, 115.48),
        "Nanchang": (28.68, 115.86),
        "Changzhou": (31.78, 119.95),
        "Guiyang": (26.64, 106.63),
        "Luoyang": (34.62, 112.45),
        "Tangshan": (39.63, 118.18),
        "Nantong": (32.01, 120.86),
        "Haerbin": (45.80, 126.53),
        "Handan": (36.61, 114.49),
        "Weifang": (36.71, 119.16),
        "Wulumuqi": (43.83, 87.62),
        "Quanzhou": (24.87, 118.68),
        "Fuzhou": (26.08, 119.30),
        "Zhongshan": (22.52, 113.39),
        "Jiaxing": (30.75, 120.75)
    }

    # 22个回收中心候选 (完整保留，对齐数学模型固定成本计算)
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

    # 6个工厂配置 (完整保留，对齐数学模型)
    factory_config = [
        ("XiAn", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
        ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
        ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
    ]

    # 自动计算衍生数据 (对齐数学模型)
    total_city_weight = sum([w for _, w in city_sales_weight])
    city_sales_ratio = total_city_weight / len(city_sales_weight)
    actual_50_sales_total = NATIONAL_SALES_TOTAL * city_sales_ratio

    city_demand = {}
    for city, weight in city_sales_weight:
        single_city_sales = actual_50_sales_total * (weight / total_city_weight)
        city_demand[city] = int(TOTAL_UNITS_NATIONAL * (single_city_sales / NATIONAL_SALES_TOTAL))

    # 构建节点名称（对齐之前代码格式）
    markets = [f"M_{city}" for city, _ in city_sales_weight]
    factories = [f"F_{city}" for city, _ in factory_config]
    candidates = [f"R_{city}" for city, _, _ in recycler_config]

    # 构建位置字典
    locations = {}
    for c, pos in factory_config: locations[f"F_{c}"] = pos
    for c, _ in city_sales_weight: locations[f"M_{c}"] = city_coords[c]
    for c, pos, _ in recycler_config: locations[f"R_{c}"] = pos

    # 构建成本与需求字典（对齐数学模型：F_k = Cost × 3000）
    fixed_cost = {f"R_{c}": cost * 3000 for c, _, cost in recycler_config}
    demand_base = {f"M_{c}": city_demand[c] for c, _ in city_sales_weight}
    demand_uncertainty = {k: v * 0.2 for k, v in demand_base.items()}  # 数学模型：d_j^var = 20% d_j^base

    return {
        'locations': locations,
        'markets': markets,
        'factories': factories,
        'candidates': candidates,
        'fixed_cost': fixed_cost,
        'demand_base': demand_base,
        'demand_uncertainty': demand_uncertainty,
        'recycler_config': recycler_config,  # 新增：用于计算利用率
        'factory_config': factory_config     # 新增：用于完整输出
    }

# 加载50城市数据
DATA_50CITIES = prepare_50cities_data()

# ==========================================
# 2. 辅助函数与模型求解 (对齐数学模型+超丰富结果提取)
# ==========================================
def get_dist(n1, n2):
    """计算两点间距离 (单位: km，对齐数学模型大圆距离折算)"""
    p1, p2 = DATA_50CITIES['locations'][n1], DATA_50CITIES['locations'][n2]
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * 100

def solve_model(params, return_detailed=True):
    """
    求解50城市CLSC模型，返回期刊级严谨结果+超丰富明细
    return_detailed: 是否返回详细数据（成本细分、排放明细、利用率等）
    """
    # 提取参数（对齐数学模型，默认值为数学模型基准值）
    carbon_tax = params.get('carbon_tax', 65)
    alpha = params.get('alpha', 0.28)
    carbon_cap = params.get('carbon_cap', 100000)  # 数学模型基准：100,000 tCO2
    capacity = params.get('capacity', 80000)       # 数学模型基准：80,000 units/yr

    # 提取50城市数据
    locations = DATA_50CITIES['locations']
    factories = DATA_50CITIES['factories']
    markets = DATA_50CITIES['markets']
    candidates = DATA_50CITIES['candidates']
    fixed_cost = DATA_50CITIES['fixed_cost']
    demand_base = DATA_50CITIES['demand_base']
    demand_uncertainty = DATA_50CITIES['demand_uncertainty']

    # 模型固定参数（对齐数学模型双碳排放因子）
    trans_cost_per_km = 1.6  # 数学模型：c_trans = 1.6 CNY/unit·km
    carbon_factor_fwd = 0.0004  # 数学模型：β_fwd = 0.0004 tCO2/unit·km
    carbon_factor_rev = 0.0030  # 数学模型：β_rev = 0.0030 tCO2/unit·km
    GAMMA = 1.0  # 数学模型鲁棒系数：γ=1.0
    max_rev_dist = 600  # 数学模型：D_max=600 km

    # 2. 模型构建（期刊级规范，对齐数学模型MILP）
    prob = pulp.LpProblem("50Cities_CLSC_Sensitivity_Journal", pulp.LpMinimize)

    # 定义决策变量
    x_vars = pulp.LpVariable.dicts("Flow_Fwd", (factories, markets), lowBound=0, cat='Continuous')
    z_vars = pulp.LpVariable.dicts("Flow_Rev", (markets, candidates), lowBound=0, cat='Continuous')
    y_vars = pulp.LpVariable.dicts("Open_Recycler", candidates, cat='Binary')
    excess_emission = pulp.LpVariable("Excess_Emission", lowBound=0, cat='Continuous')

    # 构建成本与排放表达式（对齐数学模型，细分正向/逆向）
    cost_fwd = pulp.LpAffineExpression()
    cost_rev = pulp.LpAffineExpression()
    emit_fwd = pulp.LpAffineExpression()
    emit_rev = pulp.LpAffineExpression()

    for i in factories:
        for j in markets:
            dist = get_dist(i, j)
            cost_fwd += x_vars[i][j] * dist * trans_cost_per_km
            emit_fwd += x_vars[i][j] * dist * carbon_factor_fwd

    for j in markets:
        for k in candidates:
            dist = get_dist(j, k)
            cost_rev += z_vars[j][k] * dist * trans_cost_per_km
            emit_rev += z_vars[j][k] * dist * carbon_factor_rev

    total_fixed_cost = pulp.lpSum([fixed_cost[k] * y_vars[k] for k in candidates])
    total_transport_cost = cost_fwd + cost_rev
    total_emission = emit_fwd + emit_rev
    carbon_cost = excess_emission * carbon_tax

    # 目标函数（对齐数学模型：最小化总成本）
    prob += total_fixed_cost + total_transport_cost + carbon_cost, "Total_Cost_Objective"

    # 约束条件（对齐数学模型全部约束）
    prob += excess_emission >= total_emission - carbon_cap, "Carbon_Cap_Constraint"

    for j in markets:
        robust_demand = demand_base[j] + GAMMA * demand_uncertainty[j]
        prob += pulp.lpSum([x_vars[i][j] for i in factories]) >= robust_demand, f"Demand_Constraint_{j}"

    for j in markets:
        prob += pulp.lpSum([z_vars[j][k] for k in candidates]) >= demand_base[j] * alpha, f"Recycle_Constraint_{j}"

    for k in candidates:
        prob += pulp.lpSum([z_vars[j][k] for j in markets]) <= capacity * y_vars[k], f"Capacity_Constraint_{k}"

    for j in markets:
        for k in candidates:
            dist = get_dist(j, k)
            if dist > max_rev_dist:
                prob += z_vars[j][k] == 0, f"Reverse_Distance_Constraint_{j}_{k}"

    # 3. 模型求解（静默模式）
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 4. 基础结果提取与异常兜底
    base_results = {
        'status': pulp.LpStatus[status],
        'params': params.copy(),
        'total_cost': np.nan,
        'fixed_cost': np.nan,
        'transport_cost': np.nan,
        'cost_fwd': np.nan,
        'cost_rev': np.nan,
        'carbon_cost': np.nan,
        'total_emission': np.nan,
        'emit_fwd': np.nan,
        'emit_rev': np.nan,
        'excess_emission': np.nan,
        'recyclers_built': [],
        'recycler_count': 0,
        'total_flow_fwd': 0,
        'total_flow_rev': 0
    }

    if pulp.LpStatus[status] != 'Optimal':
        if not return_detailed:
            return base_results
        else:
            base_results['detailed'] = {'utilization': {}, 'top_markets': []}
            return base_results

    # 5. 最优解结果提取（基础指标）
    base_results['total_cost'] = pulp.value(prob.objective)
    base_results['fixed_cost'] = pulp.value(total_fixed_cost)
    base_results['transport_cost'] = pulp.value(total_transport_cost)
    base_results['cost_fwd'] = pulp.value(cost_fwd)
    base_results['cost_rev'] = pulp.value(cost_rev)
    base_results['carbon_cost'] = pulp.value(carbon_cost)
    base_results['total_emission'] = pulp.value(total_emission)
    base_results['emit_fwd'] = pulp.value(emit_fwd)
    base_results['emit_rev'] = pulp.value(emit_rev)
    base_results['excess_emission'] = pulp.value(excess_emission)
    base_results['recyclers_built'] = [k for k in candidates if pulp.value(y_vars[k]) > 0.5]
    base_results['recycler_count'] = len(base_results['recyclers_built'])

    # 计算总流量
    total_flow_fwd = sum([pulp.value(x_vars[i][j]) for i in factories for j in markets])
    total_flow_rev = sum([pulp.value(z_vars[j][k]) for j in markets for k in candidates])
    base_results['total_flow_fwd'] = total_flow_fwd
    base_results['total_flow_rev'] = total_flow_rev

    # 6. 超丰富详细数据提取（仅当return_detailed=True时）
    if not return_detailed:
        return base_results

    detailed_results = {}

    # 6.1 回收中心利用率明细
    utilization = {}
    for k in base_results['recyclers_built']:
        processed_qty = sum([pulp.value(z_vars[j][k]) for j in markets])
        util_rate = (processed_qty / capacity) * 100 if capacity > 0 else 0
        utilization[k] = {
            'processed_qty': processed_qty,
            'utilization_rate': util_rate,
            'fixed_cost': fixed_cost[k],
            'capacity': capacity
        }
    detailed_results['utilization'] = utilization

    # 6.2 前10大市场需求与回收明细
    top_markets = sorted(demand_base.items(), key=lambda x: x[1], reverse=True)[:10]
    market_details = []
    for market, base_demand in top_markets:
        fwd_flow = sum([pulp.value(x_vars[i][market]) for i in factories])
        rev_flow = sum([pulp.value(z_vars[market][k]) for k in candidates])
        market_details.append({
            'market': market,
            'base_demand': base_demand,
            'robust_demand': base_demand + GAMMA * demand_uncertainty[market],
            'fwd_flow': fwd_flow,
            'rev_flow': rev_flow,
            'recycle_rate_actual': (rev_flow / base_demand) * 100 if base_demand > 0 else 0
        })
    detailed_results['top_markets'] = market_details

    # 6.3 成本结构占比
    detailed_results['cost_share'] = {
        'fixed_cost_share': (base_results['fixed_cost'] / base_results['total_cost']) * 100 if base_results['total_cost'] > 0 else 0,
        'transport_cost_share': (base_results['transport_cost'] / base_results['total_cost']) * 100 if base_results['total_cost'] > 0 else 0,
        'carbon_cost_share': (base_results['carbon_cost'] / base_results['total_cost']) * 100 if base_results['total_cost'] > 0 else 0,
        'fwd_trans_share': (base_results['cost_fwd'] / base_results['transport_cost']) * 100 if base_results['transport_cost'] > 0 else 0,
        'rev_trans_share': (base_results['cost_rev'] / base_results['transport_cost']) * 100 if base_results['transport_cost'] > 0 else 0
    }

    # 6.4 排放结构占比
    detailed_results['emission_share'] = {
        'fwd_emission_share': (base_results['emit_fwd'] / base_results['total_emission']) * 100 if base_results['total_emission'] > 0 else 0,
        'rev_emission_share': (base_results['emit_rev'] / base_results['total_emission']) * 100 if base_results['total_emission'] > 0 else 0,
        'excess_emission_share': (base_results['excess_emission'] / base_results['total_emission']) * 100 if base_results['total_emission'] > 0 else 0
    }

    # 6.5 参数弹性系数（相对于基准值，提前预留）
    detailed_results['elasticity'] = {}

    # 合并详细结果到基础结果
    base_results['detailed'] = detailed_results

    return base_results

# ==========================================
# 3. 基准模型求解 (超丰富期刊级数据打印)
# ==========================================
print("=" * 100)
print(f"{LATEX_BF_START}50 Cities CLSC Baseline Solution (Journal Calibration - Full Details){LATEX_BF_END}")
print("=" * 100)

# 基准参数（对齐数学模型基准值）
base_params = {
    'carbon_tax': 65,
    'alpha': 0.28,
    'carbon_cap': 100000,
    'capacity': 80000
}

# 求解基准模型（返回超详细结果）
base_results = solve_model(base_params, return_detailed=True)

# 3.1 基础结果打印（期刊级格式化）
if base_results['status'] == 'Optimal':
    print("\n" + "-" * 100)
    print(f"{LATEX_BF_START}[A] Core Objective Results{LATEX_BF_END}")
    print("-" * 100)
    print(f"Baseline Total Cost:                {base_results['total_cost']:,.2f} {LATEX_CN} ({base_results['total_cost']/COST_FACTOR:,.2f} × 10⁴ {LATEX_CN})")
    print(f"  -> Fixed Construction Cost:       {base_results['fixed_cost']:,.2f} {LATEX_CN} ({base_results['fixed_cost']/base_results['total_cost']*100:>.2f}%)")
    print(f"  -> Total Transport Cost:          {base_results['transport_cost']:,.2f} {LATEX_CN} ({base_results['transport_cost']/base_results['total_cost']*100:>.2f}%)")
    print(f"     > Forward Transport Cost:      {base_results['cost_fwd']:,.2f} {LATEX_CN} ({base_results['cost_fwd']/base_results['transport_cost']*100:>.2f}%)")
    print(f"     > Reverse Transport Cost:      {base_results['cost_rev']:,.2f} {LATEX_CN} ({base_results['cost_rev']/base_results['transport_cost']*100:>.2f}%)")
    print(f"  -> Carbon Tax Expense:            {base_results['carbon_cost']:,.2f} {LATEX_CN} ({base_results['carbon_cost']/base_results['total_cost']*100:>.2f}%)")
    print(f"\nTotal Carbon Emissions:              {base_results['total_emission']:,.2f} t{LATEX_CO2}")
    print(f"  -> Forward Emissions (Factory→Market): {base_results['emit_fwd']:,.2f} t{LATEX_CO2} ({base_results['emit_fwd']/base_results['total_emission']*100:>.2f}%)")
    print(f"  -> Reverse Emissions (Market→Recycler): {base_results['emit_rev']:,.2f} t{LATEX_CO2} ({base_results['emit_rev']/base_results['total_emission']*100:>.2f}%)")
    print(f"  -> Excess Emissions (Taxable):    {base_results['excess_emission']:,.2f} t{LATEX_CO2} ({base_results['excess_emission']/base_results['total_emission']*100:>.2f}%)")
    print(f"\nNetwork Flow Statistics:")
    print(f"  -> Total Forward Flow (Products): {base_results['total_flow_fwd']:,.0f} units")
    print(f"  -> Total Reverse Flow (Recycled): {base_results['total_flow_rev']:,.0f} units")
    print(f"  -> Actual Overall Recycle Rate:   {base_results['total_flow_rev']/sum(DATA_50CITIES['demand_base'].values())*100:>.2f}% (Target: {base_params['alpha']*100}%)")
    print(f"\nFacility Status:")
    print(f"  -> Built Recycling Centers:       {base_results['recycler_count']} units (from 22 candidates)")
    print(f"  -> Key Facilities (Top 5):        {[rc.replace('R_', '') for rc in base_results['recyclers_built'][:5]]}...")

    # 3.2 回收中心利用率明细打印
    print("\n" + "-" * 100)
    print(f"{LATEX_BF_START}[B] Recycling Center Utilization Details{LATEX_BF_END}")
    print("-" * 100)
    print(f"{'Recycling Center':<20} | {'Processed Qty':<15} | {'Utilization (%)':<15} | {'Fixed Cost (CNY)':<20} | {'Capacity (Units)'}")
    print("-" * 100)
    for rc, details in base_results['detailed']['utilization'].items():
        rc_name = rc.replace('R_', '')
        processed = details['processed_qty']
        util = details['utilization_rate']
        fixed_c = details['fixed_cost']
        cap = details['capacity']
        print(f"{rc_name:<20} | {processed:<15,.0f} | {util:<15.2f} | {fixed_c:<20,.2f} | {cap:,.0f}")

    # 3.3 前10大市场明细打印
    print("\n" + "-" * 100)
    print(f"{LATEX_BF_START}[C] Top 10 Market Demand & Recycle Details{LATEX_BF_END}")
    print("-" * 100)
    print(f"{'Market':<15} | {'Base Demand':<15} | {'Robust Demand':<15} | {'Forward Flow':<15} | {'Reverse Flow':<15} | {'Actual Recycle Rate (%)'}")
    print("-" * 100)
    for market_detail in base_results['detailed']['top_markets']:
        market_name = market_detail['market'].replace('M_', '')
        base_d = market_detail['base_demand']
        robust_d = market_detail['robust_demand']
        fwd_f = market_detail['fwd_flow']
        rev_f = market_detail['rev_flow']
        recyc_r = market_detail['recycle_rate_actual']
        print(f"{market_name:<15} | {base_d:<15,.0f} | {robust_d:<15,.0f} | {fwd_f:<15,.0f} | {rev_f:<15,.0f} | {recyc_r:<15.2f}")

    # 3.4 成本与排放结构占比打印
    print("\n" + "-" * 100)
    print(f"{LATEX_BF_START}[D] Cost & Emission Structure Share{LATEX_BF_END}")
    print("-" * 100)
    cost_share = base_results['detailed']['cost_share']
    emission_share = base_results['detailed']['emission_share']
    print(f"Cost Structure:")
    print(f"  -> Fixed Cost Share:              {cost_share['fixed_cost_share']:>.2f}%")
    print(f"  -> Transport Cost Share:          {cost_share['transport_cost_share']:>.2f}%")
    print(f"  -> Carbon Cost Share:             {cost_share['carbon_cost_share']:>.2f}%")
    print(f"\nEmission Structure:")
    print(f"  -> Forward Emission Share:        {emission_share['fwd_emission_share']:>.2f}%")
    print(f"  -> Reverse Emission Share:        {emission_share['rev_emission_share']:>.2f}%")
    print(f"  -> Taxable Excess Emission Share: {emission_share['excess_emission_share']:>.2f}%")
else:
    print("Baseline Model Infeasible! Please check constraints and parameters.")

print("=" * 100)

# ==========================================
# 4. 灵敏度分析 (超丰富数据打印+保留旧代码风格)
# ==========================================
print("\n" + "=" * 100)
print(f"{LATEX_BF_START}50 Cities CLSC Sensitivity Analysis (Full Detailed Output){LATEX_BF_END}")
print("=" * 100)

# 分析配置（保持旧代码参数变化范围，对齐数学模型）
analysis_config = {
    'carbon_tax': [45.5, 55.25, 65, 74.75, 84.5],    # ±15%、±30% 变化（基准65）
    'alpha': [0.20, 0.24, 0.28, 0.32, 0.36],          # 回收率变化（基准0.28）
    'carbon_cap': [70000, 85000, 100000, 115000, 130000],  # 碳容量变化（基准100000）
    'capacity': [56000, 68000, 80000, 92000, 104000]  # 回收中心容量变化（基准80000）
}

# 存储灵敏度结果
sensitivity_results = {}
# 存储所有场景的详细结果（用于后续对比）
all_scenario_results = {}

for param_name, param_values in analysis_config.items():
    print(f"\n" + "=" * 100)
    param_title = param_name.replace('_', ' ').title()
    print(f"{LATEX_BF_START}Sensitivity to Parameter: {param_title}{LATEX_BF_END}")
    print("=" * 100)
    sensitivity_results[param_name] = []
    all_scenario_results[param_name] = []

    for idx, val in enumerate(param_values):
        test_params = base_params.copy()
        test_params[param_name] = val
        res = solve_model(test_params, return_detailed=True)

        # 计算成本变化率与弹性系数
        cost_change = np.nan
        elasticity = np.nan
        if res['status'] == 'Optimal' and not np.isnan(res['total_cost']) and not np.isnan(base_results['total_cost']):
            cost_change = (res['total_cost'] - base_results['total_cost']) / base_results['total_cost'] * 100
            # 计算弹性系数：成本变化率 / 参数变化率
            param_base = base_params[param_name]
            param_change_rate = (val - param_base) / param_base * 100 if param_base != 0 else 0
            elasticity = cost_change / param_change_rate if param_change_rate != 0 else np.nan

        # 存储简化结果（用于绘图）
        sensitivity_results[param_name].append({
            'value': val,
            'total_cost': res['total_cost'],
            'cost_change': cost_change,
            'recyclers_built': res['recyclers_built'],
            'recycler_count': res['recycler_count'],
            'carbon_cost': res['carbon_cost'],
            'elasticity': elasticity
        })

        # 存储完整详细结果
        all_scenario_results[param_name].append(res)

        # 3. 超丰富场景结果打印
        print(f"\n" + "-" * 100)
        print(f"{LATEX_BF_START}Scenario {idx+1}: {param_name} = {val}{LATEX_BF_END} (Base: {base_params[param_name]})")
        print("-" * 100)
        if res['status'] == 'Optimal':
            # 核心指标
            print(f"Scenario Status:                    Optimal")
            print(f"Total Cost:                         {res['total_cost']:,.2f} {LATEX_CN} ({res['total_cost']/COST_FACTOR:,.2f} × 10⁴ {LATEX_CN})")
            print(f"Cost Change vs Baseline:            {cost_change:+.2f}%")
            print(f"Elasticity Coefficient:             {elasticity:>.4f} (Cost % / Param %)")
            print(f"Built Recycling Centers:            {res['recycler_count']} units (Change: {res['recycler_count'] - base_results['recycler_count']:+d})")
            print(f"Carbon Tax Expense:                 {res['carbon_cost']:,.2f} {LATEX_CN} (Change: {(res['carbon_cost'] - base_results['carbon_cost']):+.2f} {LATEX_CN})")
            print(f"Total Emissions:                    {res['total_emission']:,.2f} t{LATEX_CO2} (Excess: {res['excess_emission']:,.2f} t{LATEX_CO2})")

            # 成本结构简况
            if res['total_cost'] > 0:
                fixed_share = (res['fixed_cost'] / res['total_cost']) * 100
                trans_share = (res['transport_cost'] / res['total_cost']) * 100
                carbon_share = (res['carbon_cost'] / res['total_cost']) * 100
                print(f"\nCost Structure Snapshot:")
                print(f"  -> Fixed Cost: {fixed_share:>.2f}% | Transport Cost: {trans_share:>.2f}% | Carbon Cost: {carbon_share:>.2f}%")

            # 回收中心利用率简况（Top 3）
            top_recyclers = sorted(res['detailed']['utilization'].items(), key=lambda x: x[1]['utilization_rate'], reverse=True)[:3]
            print(f"\nTop 3 Recycling Center Utilization:")
            for rc, util_details in top_recyclers:
                rc_name = rc.replace('R_', '')
                print(f"  -> {rc_name}: {util_details['processed_qty']:,.0f} units ({util_details['utilization_rate']:>.2f}% utilization)")
        else:
            print(f"Scenario Status:                    {res['status']} (Infeasible/Unbounded)")

# 4. 灵敏度分析汇总表打印
print("\n" + "=" * 100)
print(f"{LATEX_BF_START}Sensitivity Analysis Summary Table{LATEX_BF_END}")
print("=" * 100)
print(f"{'Parameter':<20} | {'Value':<15} | {'Total Cost (×10⁴ CNY)':<25} | {'Cost Change (%)':<20} | {'Recyclers Built':<20} | {'Elasticity'}")
print("-" * 100)
for param_name, results in sensitivity_results.items():
    for res in results:
        cost_str = f"{res['total_cost']/COST_FACTOR:,.2f}" if not np.isnan(res['total_cost']) else "nan"
        change_str = f"{res['cost_change']:+.2f}" if not np.isnan(res['cost_change']) else "nan"
        elastic_str = f"{res['elasticity']:>.4f}" if not np.isnan(res['elasticity']) else "nan"
        print(f"{param_name:<20} | {res['value']:<15} | {cost_str:<25} | {change_str:<20} | {res['recycler_count']:<20} | {elastic_str}")

print("=" * 100)

# ==========================================
# 5. 可视化 (HEAVILY OPTIMIZED)
# ==========================================
print("\n" + "=" * 100)
print(f"{LATEX_BF_START}Generating Journal-Quality Sensitivity Chart{LATEX_BF_END}")
print("=" * 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # constrained_layout=True in rcParams
fig.suptitle(f"{LATEX_BF_START}Sensitivity Analysis of 50 Cities EV Battery Closed-Loop Supply Chain{LATEX_BF_END}",
             y=1.03, fontsize=18, color=COLORS['text_color'])

# 参数标签（期刊级LaTex格式，适配数学模型参数及绘图x轴单位）
param_labels = {
    'carbon_tax': r"Carbon Tax ($C_{tax}$) (CNY/tCO$_2$)",
    'alpha': r'Recovery Rate ($\alpha$)',
    'carbon_cap': r"Carbon Cap ($E_{cap}$) ($10^4$ tCO$_2$)", # 10^4 匹配绘图中的 x 轴单位转换
    'capacity': r'Recycler Capacity ($Cap$) ($10^4$ units/yr)' # 10^4 匹配绘图中的 x 轴单位转换
}

# 线条颜色循环
line_colors_list = [COLORS['primary_line'], COLORS['secondary_line'], COLORS['tertiary_line'], COLORS['quaternary_line']]

for idx, (param_name, results) in enumerate(sensitivity_results.items()):
    ax1 = axes.flat[idx]

    # 筛选有效结果
    valid_results = [r for r in results if not np.isnan(r['total_cost'])]
    if not valid_results:
        continue

    # 提取数据
    values = [item['value'] for item in valid_results]
    costs = [item['total_cost'] / COST_FACTOR for item in valid_results]
    num_recyclers = [item['recycler_count'] for item in valid_results]

    # 单位转换（适配绘图x轴单位）
    display_values = list(values) # Ensure it's a list for modification
    if param_name == 'carbon_cap':
        display_values = [v / 1e4 for v in values]  # 转换为 10^4 tCO2
    elif param_name == 'capacity':
        display_values = [v / CAPACITY_FACTOR for v in values]  # 转换为 10^4 units/yr

    # 双轴图构建
    ax2 = ax1.twinx()

    # 柱状图（设施数量）
    bar_width = (max(display_values) - min(display_values)) / (len(display_values) - 1 + 1e-6) * 0.4 if len(display_values) > 1 else 0.1
    ax2.bar(display_values, num_recyclers, width=bar_width,
            color=COLORS['bar_fill'], edgecolor=COLORS['bar_edge'], linewidth=0.8, alpha=0.7,
            label='No. of Facilities', zorder=1)

    # 线图（总成本）
    current_line_color = line_colors_list[idx % len(line_colors_list)]
    ax1.plot(display_values, costs, marker='o', markersize=7, linewidth=2.5,
             color=current_line_color, markerfacecolor='white', markeredgewidth=1.5,
             label='Total Cost', zorder=10)

    # 坐标轴优化
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # 设施数量是整数
    ax2.set_ylim(bottom=0, top=max(num_recyclers) * 1.2 if max(num_recyclers) > 0 else 5) # 调整y轴上限

    ax1.set_xlabel(param_labels[param_name], fontweight='bold', color=COLORS['text_color'])
    ax1.set_ylabel(r"Total Cost ($10^4$ CNY)", color=current_line_color, fontweight='bold')
    ax2.set_ylabel("No. of Established Recyclers", color=COLORS['text_color'], rotation=270, labelpad=18)

    # 子图标题（期刊级字母标注）
    letters = ['(a)', '(b)', '(c)', '(d)']
    param_title_display = param_name.replace('_', ' ').title()
    ax1.set_title(f"{letters[idx]} Sensitivity to {param_title_display}",
                  loc='left', fontsize=14, fontweight='bold', pad=12, color=COLORS['text_color'])

    # 网格优化
    ax1.grid(True, which='major', linestyle=':', alpha=0.4, color=COLORS['grid'])

    # 刻度颜色匹配
    ax1.tick_params(axis='y', colors=current_line_color)
    ax2.tick_params(axis='y', colors=COLORS['text_color'])

    # 标注基准值
    base_val = base_params[param_name]
    if param_name == 'carbon_cap':
        disp_base = base_val / 1e4
    elif param_name == 'capacity':
        disp_base = base_val / CAPACITY_FACTOR
    else:
        disp_base = base_val

    try:
        # 提取基准值对应的成本
        base_cost_y = [c for v, c in zip(display_values, costs) if np.isclose(v, disp_base)][0]
        # 绘制基准线
        ax1.axvline(x=disp_base, color='gray', linestyle=':', linewidth=1.5, zorder=5)
        # 绘制基准标记（*号突出）
        ax1.scatter([disp_base], [base_cost_y], color=COLORS['baseline_marker'], s=200, marker='*',
                    zorder=20, label='Baseline', edgecolors='k', linewidth=1) # 增加edgecolors和linewidth

        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        # 避免重复的 'No. of Facilities' 和 'Total Cost' 标签
        combined_labels = []
        combined_handles = []
        if 'Total Cost' in labels_1:
            combined_labels.append('Total Cost')
            combined_handles.append(lines_1[labels_1.index('Total Cost')])
        if 'No. of Facilities' in labels_2:
            combined_labels.append('No. of Facilities')
            combined_handles.append(lines_2[labels_2.index('No. of Facilities')])
        # 添加基准标签（如果存在）
        if 'Baseline' in labels_1:
             combined_labels.append('Baseline')
             combined_handles.append(lines_1[labels_1.index('Baseline')])

        ax1.legend(combined_handles, combined_labels, loc='best',
                   frameon=False, fontsize=10, ncol=1, handletextpad=0.5, columnspacing=1.0)
    except IndexError:
        pass # If baseline value is not in `display_values`, skip this

    # 标注极值变化率（圆角框）
    for i in [0, -1]:
        # 确保索引有效
        if len(display_values) > i and i >= -len(display_values):
            val = display_values[i]
            cost = costs[i]
            orig_res = [r for r in valid_results if np.isclose(r['value'], values[i])][0]
            chg = orig_res['cost_change']

            if not np.isnan(chg) and abs(chg) > 0.05: # 仅标注显著变化
                box_props = dict(boxstyle="round,pad=0.3", fc="white", ec=current_line_color, alpha=0.9, lw=0.8)
                txt = f"{chg:+.1f}%"
                ax1.annotate(txt, xy=(val, cost), xytext=(0, 18 if chg > 0 else -18), # 调整ytext偏移
                             textcoords='offset points', ha='center', fontsize=9,
                             color=COLORS['text_color'], bbox=box_props, fontweight='bold', zorder=25)

# 保存期刊级图表
output_file = "50cities_sensitivity_analysis_full_details.png"
plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
print(f"\nChart saved successfully: {output_file} (600 DPI, Journal Quality)")
print("=" * 100)

# 显示图表
plt.show()
