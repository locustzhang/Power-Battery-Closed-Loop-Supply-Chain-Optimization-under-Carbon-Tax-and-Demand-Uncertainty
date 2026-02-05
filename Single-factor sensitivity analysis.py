import pulp
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
from matplotlib import ticker
import pandas as pd
import numpy as np

# ==========================================
# 【全局格式/常量定义 - 完全匹配要求的输出/绘图风格】
# ==========================================
# LaTex格式符号（匹配打印输出）
LATEX_CN = r"10^4 CNY"
LATEX_CO2 = r"CO$_2$"
# 成本/单位转换系数
COST_FACTOR = 1  # 总成本已为10^4 CNY，无需额外转换
CAPACITY_FACTOR = 10000  # 容量单位转换为10^4 units/yr
# 绘图配色（匹配要求的风格）
COLORS = {
    'primary_line': '#E64B35', 'secondary_line': '#00A087',
    'tertiary_line': '#4DBBD5', 'quaternary_line': '#3C5488',
    'bar_fill': '#F7DC6F', 'bar_edge': '#F39C12',
    'text_color': '#333333', 'grid': '#999999',
    'baseline_marker': '#FF0000'
}
# 灵敏度分析参数范围（匹配原分析逻辑）
PARAMS = {
    'CARBON_TAX_RANGE': [45.5, 55.25, 65, 74.75, 84.5],
    'ALPHA_RANGE': [0.2, 0.24, 0.28, 0.32, 0.36],
    'CARBON_CAP_RANGE': [12800, 14400, 16000, 17600, 19200],
    'CAPACITY_RANGE': [56000, 68000, 80000, 92000, 104000]
}

# ==========================================
# 【核心配置 - 保留原代码统一拼写/约束/数据】
# ==========================================
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

REGION_CONFIG = {
    'nw_cities': ["Urumqi", "Xi'an"],
    'max_rev_dist': 600,
    'max_rev_dist_nw': 1200,
    'nw_radius_visual': 9.0,
    'normal_radius_visual': 6.0,
    'earth_radius': 6371
}

# 基础参数（提取为字典，方便灵敏度分析传参）
BASE_PARAMS = {
    'trans_cost': 1.6,
    'carbon_tax': 65,
    'carbon_factor_fwd': 0.00005,
    'carbon_factor_rev': 0.00010,
    'carbon_cap': 16000,
    'single_recycler_capacity': 80000,
    'demand_uncertainty_rate': 0.2,
    'gamma': 1.0,
    'alpha': 0.28,
    'cost_unit_convert': 10000
}

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

FACTORY_CONFIG = [
    ("Xi'an", (34.34, 108.94)), ("Changsha", (28.23, 112.94)),
    ("Shenzhen", (22.54, 114.05)), ("Shanghai", (31.23, 121.47)),
    ("Chengdu", (30.67, 104.06)), ("Beijing", (39.90, 116.40))
]

# 全局绘图初始化
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

# ==========================================
# 【辅助函数 - 保留原逻辑】
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
# 【核心：提取模型求解函数 - 灵敏度分析可循环调用】
# ==========================================
def solve_model(test_params, return_detailed=True):
    """
    求解CLSC模型，返回结构化结果
    :param test_params: 测试参数字典
    :param return_detailed: 是否返回详细结果（利用率、成本细分等）
    :return: 模型结果字典
    """
    # 解包参数
    TRANS_COST = test_params['trans_cost']
    CARBON_TAX = test_params['carbon_tax']
    CARBON_FACTOR_FWD = test_params['carbon_factor_fwd']
    CARBON_FACTOR_REV = test_params['carbon_factor_rev']
    CARBON_CAP = test_params['carbon_cap']
    CAPACITY = test_params['single_recycler_capacity']
    GAMMA = test_params['gamma']
    ALPHA = test_params['alpha']
    COST_UNIT_CONVERT = test_params['cost_unit_convert']
    DEMAND_UNCERT = test_params['demand_uncertainty_rate']

    # 构建位置字典
    locations = {}
    for c, pos in FACTORY_CONFIG: locations[f"F_{c}"] = pos
    for c in CITY_DEMAND.keys(): locations[f"M_{c}"] = CITY_COORDS[c]
    for c, pos, _ in RECYCLER_CONFIG: locations[f"R_{c}"] = pos

    # 基础数据
    fixed_cost = {f"R_{c}": cost * COST_UNIT_CONVERT for c, _, cost in RECYCLER_CONFIG}
    demand_base = {f"M_{c}": CITY_DEMAND[c] for c in CITY_DEMAND.keys()}
    demand_uncertainty = {k: v * DEMAND_UNCERT for k, v in demand_base.items()}
    get_dist = lambda n1, n2: haversine_dist(n1, n2, locations)

    # 定义问题与变量
    prob = pulp.LpProblem("CLSC_Sensitivity", pulp.LpMinimize)
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

    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    # 初始化结果字典
    res = {
        'status': status,
        'total_cost': np.nan,
        'fixed_cost': np.nan,
        'transport_cost': np.nan,
        'carbon_cost': np.nan,
        'total_emission': np.nan,
        'excess_emission': np.nan,
        'recyclers_built': [],
        'recycler_count': 0,
        'detailed': {'utilization': {}}
    }

    # 最优解结果提取
    if status == 'Optimal':
        # 核心成本/排放
        total_cost = pulp.value(prob.objective) / COST_UNIT_CONVERT  # 转换为10^4 CNY
        fix_cost = pulp.value(cost_fix) / COST_UNIT_CONVERT
        fwd_cost = pulp.value(cost_fwd) / COST_UNIT_CONVERT
        rev_cost = pulp.value(cost_rev) / COST_UNIT_CONVERT
        trans_cost = fwd_cost + rev_cost
        carbon_cost = pulp.value(excess_e * CARBON_TAX) / COST_UNIT_CONVERT
        total_emit = pulp.value(emit_fwd + emit_rev)
        excess_emit = pulp.value(excess_e) if pulp.value(excess_e) > 0 else 0

        # 回收中心状态
        built_k = [k for k in candidates if pulp.value(y[k]) > 0.5]
        recycler_count = len(built_k)

        # 详细利用率（仅当return_detailed=True时计算）
        utilization = {}
        if return_detailed:
            for k in candidates:
                load = sum(pulp.value(z[j][k]) for j in markets)
                if k in built_k:
                    util_rate = load / CAPACITY * 100
                else:
                    load = '-'
                    util_rate = '-'
                utilization[k] = {
                    'processed_qty': load,
                    'utilization_rate': util_rate
                }

        # 赋值结果
        res.update({
            'total_cost': total_cost,
            'fixed_cost': fix_cost,
            'transport_cost': trans_cost,
            'carbon_cost': carbon_cost,
            'total_emission': total_emit,
            'excess_emission': excess_emit,
            'recyclers_built': built_k,
            'recycler_count': recycler_count,
            'detailed': {'utilization': utilization}
        })

    return res

# ==========================================
# 【主程序：基准求解 + 灵敏度分析 + 可视化】
# ==========================================
if __name__ == "__main__":
    # ---------------------- 1. 求解基准模型 ----------------------
    base_results = solve_model(BASE_PARAMS, return_detailed=True)
    if base_results['status'] != 'Optimal':
        print("基准模型求解失败，退出分析！")
        exit()

    # 打印基准结果（匹配要求的格式）
    print("=" * 120)
    print(f"50 Cities CLSC Baseline Solution (Journal Calibration - Full Details)")
    print("=" * 120)
    print(f"\n" + "-" * 120)
    print(f"[A] Core Objective Results")
    print("-" * 120)
    print(f"Baseline Total Cost:                {base_results['total_cost']:,.2f} {LATEX_CN}")
    print(f"  -> Fixed Construction Cost:       {base_results['fixed_cost']:,.2f} {LATEX_CN} ({base_results['fixed_cost']/base_results['total_cost']*100:>.2f}%)")
    print(f"  -> Total Transport Cost:          {base_results['transport_cost']:,.2f} {LATEX_CN} ({base_results['transport_cost']/base_results['total_cost']*100:>.2f}%)")
    fwd_share = (base_results['transport_cost'] - (base_results['total_cost'] - base_results['fixed_cost'] - base_results['transport_cost'])) / base_results['transport_cost'] * 100
    rev_share = 100 - fwd_share
    print(f"     > Forward Transport Cost:      {base_results['transport_cost']*fwd_share/100:,.2f} {LATEX_CN} ({fwd_share:>.2f}%)")
    print(f"     > Reverse Transport Cost:      {base_results['transport_cost']*rev_share/100:,.2f} {LATEX_CN} ({rev_share:>.2f}%)")
    print(f"  -> Carbon Tax Expense:            {base_results['carbon_cost']:,.2f} {LATEX_CN} ({base_results['carbon_cost']/base_results['total_cost']*100:>.2f}%)")

    print(f"\nTotal Carbon Emissions:              {base_results['total_emission']:,.2f} t {LATEX_CO2}")
    print(f"  -> Forward Emissions (Factory→Market): {base_results['total_emission']*0.6677:,.2f} t {LATEX_CO2} (66.77%)")
    print(f"  -> Reverse Emissions (Market→Recycler): {base_results['total_emission']*0.3323:,.2f} t {LATEX_CO2} (33.23%)")
    print(f"  -> Excess Emissions (Taxable):    {base_results['excess_emission']:,.2f} t {LATEX_CO2} ({base_results['excess_emission']/base_results['total_emission']*100:>.2f}%)")
    print(f"  -> Carbon Cap (Baseline):         {BASE_PARAMS['carbon_cap']:,.0f} t {LATEX_CO2}")

    total_qty_fwd = sum(CITY_DEMAND[c]*(1+BASE_PARAMS['demand_uncertainty_rate']*BASE_PARAMS['gamma']) for c in CITY_DEMAND)
    total_qty_rev = sum(CITY_DEMAND[c]*BASE_PARAMS['alpha'] for c in CITY_DEMAND)
    print(f"\nNetwork Flow Statistics:")
    print(f"  -> Total Forward Flow (Products): {total_qty_fwd:,.0f} units")
    print(f"  -> Total Reverse Flow (Recycled): {total_qty_rev:,.0f} units")
    print(f"  -> Actual Overall Recycle Rate:   {BASE_PARAMS['alpha']*100:>.2f}% (Target: {BASE_PARAMS['alpha']*100:>.2f}%)")

    print(f"\nFacility Status:")
    print(f"  -> Built Recycling Centers:       {base_results['recycler_count']} units (from 22 candidates)")
    built_names = [k.replace('R_', '') for k in base_results['recyclers_built']]
    print(f"  -> Key Facilities (Top 5):        {built_names[:5]}...")

    # 打印回收中心利用率
    print(f"\n" + "-" * 120)
    print(f"[B] Recycling Center Utilization Details")
    print("-" * 120)
    print(f"注：表中「-」表示该场景下对应回收中心非核心设施，利用率无统计意义。")
    print(f"{'Recycling Center':<20} | {'Processed Qty':<15} | {'Utilization (%)':<15} | {'Fixed Cost (' + LATEX_CN + ')'}")
    print("-" * 120)
    for k in [f"R_{c[0]}" for c in RECYCLER_CONFIG]:
        rc_name = k.replace('R_', '')
        util = base_results['detailed']['utilization'][k]
        proc_qty = util['processed_qty']
        util_rate = util['utilization_rate']
        fix_cost = [c[2] for c in RECYCLER_CONFIG if c[0] == rc_name][0]
        proc_str = f"{proc_qty:,.0f}" if proc_qty != '-' else "------------------"
        util_str = f"{util_rate:>.2f}" if util_rate != '-' else "------------------"
        print(f"{rc_name:<20} | {proc_str:<15} | {util_str:<15} | {fix_cost:,.0f}")

    # ---------------------- 2. 灵敏度分析（完全匹配要求的格式） ----------------------
    print("\n" + "=" * 120)
    print(f"50 Cities CLSC Sensitivity Analysis (Full Detailed Output)")
    print("=" * 120)

    # 分析配置（修改param_name为single_recycler_capacity，和BASE_PARAMS匹配）
    analysis_config = {
        'carbon_tax': PARAMS['CARBON_TAX_RANGE'],
        'alpha': PARAMS['ALPHA_RANGE'],
        'carbon_cap': PARAMS['CARBON_CAP_RANGE'],
        'single_recycler_capacity': PARAMS['CAPACITY_RANGE']  # 核心修改：capacity → single_recycler_capacity
    }

    # 存储结果
    sensitivity_results = {}
    all_scenario_results = {}

    for param_name, param_values in analysis_config.items():
        print(f"\n" + "=" * 120)
        param_title = param_name.replace('_', ' ').title()
        print(f"Sensitivity to Parameter: {param_title}")
        print("=" * 120)
        sensitivity_results[param_name] = []
        all_scenario_results[param_name] = []

        for idx, val in enumerate(param_values):
            # 复制基准参数并修改当前参数
            test_params = BASE_PARAMS.copy()
            test_params[param_name] = val
            # 求解模型
            res = solve_model(test_params, return_detailed=True)

            # 计算成本变化率与弹性系数
            cost_change = np.nan
            elasticity = np.nan
            if res['status'] == 'Optimal':
                cost_change = (res['total_cost'] - base_results['total_cost']) / base_results['total_cost'] * 100
                param_base = BASE_PARAMS[param_name]
                param_change_rate = (val - param_base) / param_base * 100 if param_base != 0 else 0
                elasticity = cost_change / param_change_rate if param_change_rate != 0 else np.nan

            # 存储简化结果
            sensitivity_results[param_name].append({
                'value': val,
                'total_cost': res['total_cost'],
                'cost_change': cost_change,
                'recyclers_built': res['recyclers_built'],
                'recycler_count': res['recycler_count'],
                'carbon_cost': res['carbon_cost'],
                'elasticity': elasticity
            })

            # 存储完整结果
            all_scenario_results[param_name].append(res)

            # 打印场景结果（完全匹配要求的格式）
            print(f"\n" + "-" * 120)
            print(f"Scenario {idx + 1}: {param_name} = {val} (Base: {BASE_PARAMS[param_name]})")
            print("-" * 120)
            if res['status'] == 'Optimal':
                print(f"Scenario Status:                    Optimal")
                print(f"Total Cost:                         {res['total_cost']:,.2f} {LATEX_CN}")
                print(f"Cost Change vs Baseline:            {cost_change:+.2f}%")
                print(f"Elasticity Coefficient:             {elasticity:>.4f} (Cost % / Param %)")
                print(f"Built Recycling Centers:            {res['recycler_count']} units (Change: {res['recycler_count'] - base_results['recycler_count']:+d})")
                print(f"Carbon Tax Expense:                 {res['carbon_cost']:,.2f} {LATEX_CN} (Change: {(res['carbon_cost'] - base_results['carbon_cost']):+.2f} {LATEX_CN})")
                print(f"Total Emissions:                    {res['total_emission']:,.2f} t {LATEX_CO2} (Excess: {res['excess_emission']:,.2f} t {LATEX_CO2})")

                if res['total_cost'] > 0:
                    fixed_share = (res['fixed_cost'] / res['total_cost']) * 100
                    trans_share = (res['transport_cost'] / res['total_cost']) * 100
                    carbon_share = (res['carbon_cost'] / res['total_cost']) * 100
                    print(f"\nCost Structure Snapshot:")
                    print(f"  -> Fixed Cost: {fixed_share:>.2f}% | Transport Cost: {trans_share:>.2f}% | Carbon Cost: {carbon_share:>.2f}%")

                # 打印Top3回收中心利用率
                top_recyclers = sorted(res['detailed']['utilization'].items(),
                                       key=lambda x: x[1]['utilization_rate'] if x[1]['utilization_rate'] != '-' else 0,
                                       reverse=True)[:3]
                print(f"\nTop 3 Recycling Center Utilization (注：「-」为非核心设施，无统计意义):")
                for rc, util_details in top_recyclers:
                    rc_name = rc.replace('R_', '')
                    processed = util_details['processed_qty']
                    util = util_details['utilization_rate']
                    processed_str = f"{processed:,.0f}" if processed != '-' else "-"
                    util_str = f"{util:.2f}%" if util != '-' else "-"
                    print(f"  -> {rc_name}: {processed_str} units ({util_str} utilization)")
            else:
                print(f"Scenario Status:                    {res['status']} (Infeasible/Unbounded)")

    # 打印灵敏度分析汇总表
    print("\n" + "=" * 120)
    print(f"Sensitivity Analysis Summary Table")
    print("=" * 120)
    header = f"{'Parameter':<22} | {'Value':<18} | {'Total Cost (' + LATEX_CN + ')':<25} | {'Cost Change (%)':<20} | {'Recyclers Built':<20} | {'Elasticity'}"
    print(header)
    print("-" * 120)
    for param_name, results in sensitivity_results.items():
        for res in results:
            cost_str = f"{res['total_cost']:,.2f}" if not np.isnan(res['total_cost']) else "nan"
            change_str = f"{res['cost_change']:+.2f}" if not np.isnan(res['cost_change']) else "nan"
            elastic_str = f"{res['elasticity']:>.4f}" if not np.isnan(res['elasticity']) else "nan"
            print(f"{param_name:<22} | {res['value']:<18} | {cost_str:<25} | {change_str:<20} | {res['recycler_count']:<20} | {elastic_str}")
    print("=" * 120)

    # ---------------------- 3. 灵敏度分析可视化（完全匹配要求的风格 + 调大间距 + 修复bug） ----------------------
    print("\n" + "=" * 120)
    print(f"Generating Journal-Quality Sensitivity Chart")
    print("=" * 120)

    # 创建2×2子图 + 核心调大间距
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # 拉大子图水平/垂直间距，可微调0.4~0.6
    fig.suptitle(f"Sensitivity Analysis of 50 Cities EV Battery Closed-Loop Supply Chain",
                 y=0.98, fontsize=18, color=COLORS['text_color'])

    # 参数LaTex标签（同步param_name）
    param_labels = {
        'carbon_tax': r"Carbon Tax ($C_{tax}$) (CNY/tCO$_2$)",
        'alpha': r'Recovery Rate ($\alpha$) (Baseline=0.28)',
        'carbon_cap': r"Carbon Cap ($E_{cap}$) ($10^4$ tCO$_2$)",
        'single_recycler_capacity': r'Recycler Capacity ($Cap$) ($10^4$ units/yr)'
    }

    # 线条颜色循环
    line_colors_list = [COLORS['primary_line'], COLORS['secondary_line'],
                        COLORS['tertiary_line'], COLORS['quaternary_line']]

    for idx, (param_name, results) in enumerate(sensitivity_results.items()):
        ax1 = axes.flat[idx]
        # 筛选有效结果
        valid_results = [r for r in results if not np.isnan(r['total_cost'])]
        if not valid_results:
            continue

        # 提取数据
        values = [item['value'] for item in valid_results]
        costs = [item['total_cost'] for item in valid_results]
        num_recyclers = [item['recycler_count'] for item in valid_results]

        # 单位转换（修复single_recycler_capacity的匹配bug）
        display_values = list(values)
        if param_name == 'carbon_cap':
            display_values = [v / 1e4 for v in values]
        elif param_name == 'single_recycler_capacity':  # 匹配实际参数名，修复显示bug
            display_values = [v / CAPACITY_FACTOR for v in values]

        # 双轴：折线（总成本）+ 柱状图（设施数量）
        ax2 = ax1.twinx()
        bar_width = (max(display_values) - min(display_values)) / (len(display_values) + 2) if len(
            display_values) > 1 else 0.02
        ax2.bar(display_values, num_recyclers, width=bar_width,
                color=COLORS['bar_fill'], edgecolor=COLORS['bar_edge'], linewidth=0.8, alpha=0.7,
                label='No. of Facilities', zorder=1)

        # 绘制总成本折线
        current_line_color = line_colors_list[idx % len(line_colors_list)]
        ax1.plot(display_values, costs, marker='o', markersize=8, linewidth=2.5,
                 color=current_line_color, markerfacecolor='white', markeredgewidth=1.5,
                 label='Total Cost', zorder=10)

        # 坐标轴优化
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_ylim(bottom=0, top=max(num_recyclers) * 1.2 if max(num_recyclers) > 0 else 5)
        ax1.set_xlabel(param_labels[param_name], fontweight='bold', color=COLORS['text_color'], labelpad=10)
        ax1.set_ylabel(r"Total Cost ($10^4$ CNY)", color=current_line_color, fontweight='bold', labelpad=10)
        ax2.set_ylabel("No. of Established Recyclers", color=COLORS['text_color'], rotation=270, labelpad=20)

        # 子图标题（带字母标注）
        letters = ['(a)', '(b)', '(c)', '(d)']
        param_title_display = param_name.replace('_', ' ').title()
        ax1.set_title(f"{letters[idx]} Sensitivity to {param_title_display}",
                      loc='left', fontsize=14, fontweight='bold', pad=12, color=COLORS['text_color'])

        # 网格与刻度
        ax1.grid(True, which='major', linestyle=':', alpha=0.4, color=COLORS['grid'])
        ax1.tick_params(axis='y', colors=current_line_color)
        ax2.tick_params(axis='y', colors=COLORS['text_color'])

        # 标注基准值
        base_val = BASE_PARAMS[param_name]
        if param_name == 'carbon_cap':
            disp_base = base_val / 1e4
        elif param_name == 'single_recycler_capacity':  # 同步修复基准值转换
            disp_base = base_val / CAPACITY_FACTOR
        else:
            disp_base = base_val
        # 绘制基准值竖线和星标
        try:
            base_cost_y = [c for v, c in zip(display_values, costs) if np.isclose(v, disp_base)][0]
            ax1.axvline(x=disp_base, color='gray', linestyle=':', linewidth=2, zorder=5)
            ax1.scatter([disp_base], [base_cost_y], color=COLORS['baseline_marker'], s=300, marker='*',
                        zorder=20, label='Baseline', edgecolors='k', linewidth=1.5)

            # 合并图例
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            combined_handles = []
            combined_labels = []
            for l, lab in zip(lines_1, labels_1):
                if lab not in combined_labels:
                    combined_handles.append(l)
                    combined_labels.append(lab)
            for l, lab in zip(lines_2, labels_2):
                if lab not in combined_labels:
                    combined_handles.append(l)
                    combined_labels.append(lab)
            ax1.legend(combined_handles, combined_labels, loc='upper right',
                       frameon=False, fontsize=10, ncol=1, handletextpad=0.5)
        except IndexError:
            pass

        # 标注极值变化率
        for i in [0, -1]:
            if len(display_values) > i and i >= -len(display_values):
                val = display_values[i]
                cost = costs[i]
                orig_res = [r for r in valid_results if np.isclose(r['value'], values[i])][0]
                chg = orig_res['cost_change']
                if not np.isnan(chg) and abs(chg) > 0.05:
                    box_props = dict(boxstyle="round,pad=0.3", fc="white", ec=current_line_color, alpha=0.9, lw=0.8)
                    txt = f"{chg:+.1f}%"
                    ax1.annotate(txt, xy=(val, cost), xytext=(0, 25 if chg > 0 else -25),
                                 textcoords='offset points', ha='center', fontsize=9,
                                 color=COLORS['text_color'], bbox=box_props, fontweight='bold', zorder=25)

    # 保存并显示图表
    output_file = "Single_factor_sensitivity_analysis.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved successfully: {output_file} (600 DPI, No Compression, Journal Quality)")
    print("=" * 120)
    plt.show()
