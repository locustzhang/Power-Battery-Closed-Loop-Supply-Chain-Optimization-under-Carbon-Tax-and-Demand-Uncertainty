import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, uniform, gaussian_kde
import matplotlib.ticker as ticker
import warnings

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
# 忽略不必要的警告
warnings.filterwarnings('ignore')
# 设置随机种子以保证结果可复现
np.random.seed(2025)

# 绘图风格设置 (Science/Nature Journal Style)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # 优先使用 Arial
    'mathtext.fontset': 'stix',  # 数学公式使用 Times 风格 (LaTeX look)
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,  # 屏幕显示 DPI
    'savefig.dpi': 600,  # 出版级 DPI
    'axes.unicode_minus': False  # 解决负号显示问题
})

# 学术配色方案 (Colorblind-safe)
COLORS = {
    'bayes': '#00468B',  # Science Dark Blue (Bayesian)
    'robust': '#ED0000',  # Science Red (Robust)
    'base': '#42B540',  # Green (Baseline)
    'gray': '#7F7F7F',  # Gray (Annotations)
    'teal': '#0099B4',  # Teal (Secondary param)
    'purple': '#925E9F'  # Purple (Tertiary param)
}

# ==========================================
# 2. 数据准备 (Data Calibration - 2025 Industry)
# ==========================================
# 基准参数
base_params = {
    'carbon_tax': 65,  # 65 CNY/ton
    'alpha': 0.28,  # Baseline recovery rate
    'carbon_cap': 1500000,  # 1.5M tons
    'capacity': 40000  # 40k tons/year
}

# 需求数据 (12个核心城市)
demand_base = {
    'M_Beijing': 48500, 'M_Shanghai': 49500, 'M_Guangzhou': 46200,
    'M_Chengdu': 51600, 'M_Shenzhen': 50100, 'M_Hangzhou': 51200,
    'M_Zhengzhou': 39600, 'M_XiAn': 37600, 'M_Chongqing': 36700,
    'M_Tianjin': 33300, 'M_Wuhan': 35000, 'M_Changsha': 31700
}
# 需求不确定性 (20%)
demand_uncertainty = {k: v * 0.20 for k, v in demand_base.items()}
# 模拟观测数据 (基于基准 α=0.28, 5% 噪声)
observed_data = {k: v * 0.28 + np.random.normal(0, v * 0.05) for k, v in demand_base.items()}


# ==========================================
# 3. 核心模型逻辑 (Proxy Model Function)
# ==========================================
def solve_model_proxy(params):
    """
    模拟 MILP 求解结果的代理函数 (用于加速贝叶斯采样)
    反映了论文中的阈值效应(Threshold Effect)和规模效应
    """
    alpha = params['alpha']
    capacity = params['capacity']

    # 基准成本 (7.78亿元)
    base_cost = 778742400.0
    total_cost = base_cost

    # 1. 回收率 α 的非线性影响 (阶跃效应)
    # α <= 0.28: 建4个厂; α > 0.28: 建5个厂 (新增南昌)
    if alpha <= 0.28:
        cost_delta = (alpha - 0.28) * 80000000
    else:
        # 超过阈值，成本跳升
        cost_delta = (alpha - 0.28) * 1400000000
    total_cost += cost_delta

    # 2. 规模效应 (U型曲线)
    if capacity < 40000:
        total_cost += (40000 - capacity) * 1200
    else:
        total_cost -= (capacity - 40000) * 50

        # 3. 随机波动 (模拟物流和运营噪声)
    total_cost += np.random.normal(0, total_cost * 0.005)

    return total_cost


# ==========================================
# 4. 贝叶斯推断 (MCMC Implementation)
# ==========================================
# 4.1 先验分布 (Log Prior)
def prior_log_prob(params):
    alpha = params['alpha']
    tax = params['carbon_tax']
    cap = params['carbon_cap']

    # Alpha: Beta(3,7) -> 偏向 0.3
    if 0 < alpha < 1:
        lp_alpha = np.log(beta.pdf(alpha, 3, 7))
    else:
        return -np.inf

    # Tax: Uniform(45, 85)
    if 45 <= tax <= 85:
        lp_tax = np.log(uniform.pdf(tax, 45, 40))
    else:
        return -np.inf

    # Cap: Normal(1.5M, 0.2M)
    lp_cap = norm.logpdf(cap, 1500000, 200000)

    return lp_alpha + lp_tax + lp_cap


# 4.2 似然函数 (Log Likelihood)
def likelihood_log_prob(params, observed, base, uncert):
    alpha = params['alpha']
    ll = 0
    for m in base.keys():
        mu = base[m] * alpha
        sigma = uncert[m] * alpha
        # 避免 sigma 为 0
        sigma = max(sigma, 1e-6)
        ll += norm.logpdf(observed[m], loc=mu, scale=sigma)
    return ll


# 4.3 Metropolis-Hastings 采样器
def run_mcmc(n_samples=10000):
    print("正在运行 MCMC 贝叶斯采样...")
    current = base_params.copy()
    current_lp = prior_log_prob(current) + likelihood_log_prob(current, observed_data, demand_base, demand_uncertainty)

    samples = []
    # 调整步长以获得合适的接受率
    step_sizes = {'alpha': 0.015, 'carbon_tax': 1.5, 'carbon_cap': 10000}

    for _ in range(n_samples):
        proposal = current.copy()
        # 随机游走
        proposal['alpha'] += np.random.normal(0, step_sizes['alpha'])
        proposal['carbon_tax'] += np.random.normal(0, step_sizes['carbon_tax'])
        proposal['carbon_cap'] += np.random.normal(0, step_sizes['carbon_cap'])

        prop_lp = prior_log_prob(proposal) + likelihood_log_prob(proposal, observed_data, demand_base,
                                                                 demand_uncertainty)

        if np.log(np.random.rand()) < (prop_lp - current_lp):
            current = proposal
            current_lp = prop_lp

        samples.append(current.copy())

    return samples[1000:]  # Burn-in 1000


# 运行 MCMC
posterior_samples = run_mcmc()

# 提取各参数序列
alpha_chain = np.array([s['alpha'] for s in posterior_samples])
tax_chain = np.array([s['carbon_tax'] for s in posterior_samples])
cap_chain = np.array([s['carbon_cap'] for s in posterior_samples])

# ==========================================
# 5. 模拟对比: 贝叶斯 vs 鲁棒优化
# ==========================================
# 5.1 贝叶斯模拟 (基于后验采样)
print("正在进行贝叶斯仿真...")
bayes_costs = []
indices = np.random.choice(len(posterior_samples), 2000)
for idx in indices:
    bayes_costs.append(solve_model_proxy(posterior_samples[idx]))
bayes_costs = np.array(bayes_costs) / 1e4  # 转为万元

# 5.2 鲁棒优化 (基于区间不确定性)
print("正在进行鲁棒优化仿真...")
robust_costs = []
gamma = 0.2  # 20% 波动
for _ in range(5000):
    p = base_params.copy()
    # 均匀分布采样 (最保守假设)
    p['alpha'] = np.random.uniform(base_params['alpha'] * (1 - gamma), base_params['alpha'] * (1 + gamma))
    p['capacity'] = np.random.uniform(base_params['capacity'] * (1 - gamma), base_params['capacity'] * (1 + gamma))
    robust_costs.append(solve_model_proxy(p))
robust_costs = np.array(robust_costs) / 1e4  # 转为万元

# ==========================================
# 6. 绘图: 图 1 - 风险分布对比 (Figure 1)
# ==========================================
print("正在绘制 Figure 1 (Risk Profile)...")
fig1, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# KDE 计算
kde_bayes = gaussian_kde(bayes_costs)
x_bayes = np.linspace(min(bayes_costs) * 0.99, max(bayes_costs) * 1.01, 500)

kde_robust = gaussian_kde(robust_costs)
x_robust = np.linspace(min(robust_costs) * 0.98, max(robust_costs) * 1.02, 500)

# 绘制鲁棒分布 (背景)
ax.fill_between(x_robust, 0, kde_robust(x_robust), color=COLORS['robust'], alpha=0.1, label='_nolegend_')
ax.plot(x_robust, kde_robust(x_robust), color=COLORS['robust'], linestyle='--', linewidth=2,
        label='Robust Optimization (Interval-based)')

# 绘制贝叶斯分布 (前景)
ax.fill_between(x_bayes, 0, kde_bayes(x_bayes), color=COLORS['bayes'], alpha=0.3, label='_nolegend_')
ax.plot(x_bayes, kde_bayes(x_bayes), color=COLORS['bayes'], linewidth=2.5,
        label='Bayesian Inference (Data-driven)')

# 标注基准线
base_cost_val = 77874.24
ax.axvline(base_cost_val, color='k', linestyle=':', linewidth=1.5)
ax.text(base_cost_val, ax.get_ylim()[1] * 0.02, ' Baseline', ha='left', fontsize=10)

# 统计信息框 (Uncertainty Reduction)
bayes_ci = np.percentile(bayes_costs, 97.5) - np.percentile(bayes_costs, 2.5)
robust_ci = np.percentile(robust_costs, 97.5) - np.percentile(robust_costs, 2.5)
reduction = (robust_ci - bayes_ci) / robust_ci * 100

stats_text = (r"$\bf{Uncertainty\ Reduction}$" + "\n"
                                                 f"Robust Interval: {robust_ci:.0f}\n"
                                                 f"Bayesian HDI:    {bayes_ci:.0f}\n"
                                                 f"$\Delta$ Precision:   +{reduction:.1f}%")

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# 样式
ax.set_title(r"$\bf{Comparative\ Risk\ Profile:\ Bayesian\ Inference\ vs.\ Robust\ Optimization}$", pad=15)
ax.set_xlabel(r"Total Cost ($10^4$ CNY)")
ax.set_ylabel("Probability Density")
ax.legend(loc='upper right', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 保存图1
plt.savefig('Figure_1_Risk_Profile.png', dpi=600)
print(">>> 图表已保存: Figure_1_Risk_Profile.png")

# ==========================================
# 7. 绘图: 图 2 - 多参数后验分布 (Figure 2)
# ==========================================
print("正在绘制 Figure 2 (Posterior Distributions)...")
fig2, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
fig2.suptitle(r"$\bf{Posterior\ Parameter\ Distributions\ and\ Policy\ Baselines}$", fontsize=16, y=1.05)

# 绘图配置列表
plot_configs = [
    {
        'data': alpha_chain,
        'label': r'Recovery Rate ($\alpha$)',
        'base': 0.28,
        'color': COLORS['bayes'],
        'fmt': '{x:.2f}',
        'title': '(a) Recovery Rate Uncertainty'
    },
    {
        'data': tax_chain,
        'label': r'Carbon Tax ($C_{tax}$)',
        'base': 65,
        'color': COLORS['teal'],
        'fmt': '{x:.0f}',
        'title': '(b) Carbon Tax Fluctuation'
    },
    {
        'data': cap_chain / 10000,  # 换算为万吨
        'label': r'Carbon Cap ($E_{cap}$)',
        'base': 150,
        'color': COLORS['purple'],
        'fmt': '{x:.0f}',
        'title': '(c) Carbon Cap Distribution'
    }
]

for i, (ax, cfg) in enumerate(zip(axes, plot_configs)):
    data = cfg['data']

    # KDE 曲线
    kde = gaussian_kde(data)
    x_grid = np.linspace(min(data), max(data), 500)
    y_grid = kde(x_grid)

    # 绘制
    ax.plot(x_grid, y_grid, color=cfg['color'], lw=2.5)
    ax.fill_between(x_grid, 0, y_grid, color=cfg['color'], alpha=0.15)

    # 95% HDI 填充
    hdi_low = np.percentile(data, 2.5)
    hdi_high = np.percentile(data, 97.5)
    ax.fill_between(x_grid, 0, y_grid, where=(x_grid >= hdi_low) & (x_grid <= hdi_high),
                    color=cfg['color'], alpha=0.3, label='95% HDI')

    # 均值点
    mean_val = np.mean(data)
    ax.scatter([mean_val], [kde(mean_val)], color='white', edgecolor=cfg['color'], s=60, zorder=10)

    # 基准线
    ax.axvline(cfg['base'], color=COLORS['robust'], linestyle='--', linewidth=1.5, alpha=0.8)
    if i == 0:  # 只在第一张图标记文字，避免拥挤
        ax.text(cfg['base'], max(y_grid) * 1.02, ' Baseline', color=COLORS['robust'], fontsize=10, ha='center')

    # 统计信息文本
    stats_txt = f"Mean: {mean_val:.2f}\nHDI: [{hdi_low:.2f}, {hdi_high:.2f}]"
    ax.text(0.95, 0.95, stats_txt, transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    # 样式
    ax.set_xlabel(cfg['label'] + ('\n(10⁴ tons)' if i == 2 else '') + ('\n(CNY/ton)' if i == 1 else ''))
    if i == 0: ax.set_ylabel("Probability Density")
    ax.set_title(cfg['title'], loc='left', fontweight='bold', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(cfg['fmt']))

# 保存图2
plt.savefig('Figure_2_Posterior_Distributions.png', dpi=600)
print(">>> 图表已保存: Figure_2_Posterior_Distributions.png")

plt.show()