import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

N_TRIALS = 10_000

# Attendance
LOWER_BOWL_ATTENDANCE = 30_000  # Fixed

# Upper bowl: Triangular distribution
# Your three data points map cleanly to min / mode / max
UPPER_BOWL_MIN  = 30_000
UPPER_BOWL_MODE = 60_000
UPPER_BOWL_MAX  = 90_000

# Purchase rate: Beta distribution scaled to [5%, 15%]
# Beta(α=2, β=3):
#   mean  ~9%  (reflects the skew toward lower purchase rates in your estimates)
#   mode  ~8.3%
#   hard boundaries at 5% and 15%
PURCHASE_RATE_BETA_ALPHA = 2
PURCHASE_RATE_BETA_BETA  = 3
PURCHASE_RATE_MIN        = 0.05
PURCHASE_RATE_MAX        = 0.15

# Pricing (MXN)
SELL_PRICE_PER_SHIRT    = 1_500 / 12   # $1,500 MXN per dozen → $125 MXN per shirt
SALVAGE_PRICE_PER_SHIRT = 20           # Sell leftovers to discount shop

# Order options
ORDER_OPTIONS = [
    {"size": 10_000, "cost": 578_125},
    {"size":  8_000, "cost": 477_500},
    {"size":  6_000, "cost": 375_000},
]

# ─────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────

def sample_upper_bowl(n_trials):
    """
    Triangular distribution over [30k, 90k] with peak at 60k.
    Reflects your original 30k/60k/90k estimates as min/mode/max.
    """
    return np.random.triangular(
        left=UPPER_BOWL_MIN,
        mode=UPPER_BOWL_MODE,
        right=UPPER_BOWL_MAX,
        size=n_trials
    )

def sample_purchase_rate(n_trials):
    """
    Beta(2, 3) scaled to [5%, 15%].
    Mean ~9%, skewed toward lower purchase rates — consistent
    with your original 60% weight on 10% and 30% weight on 5%.
    """
    raw = np.random.beta(PURCHASE_RATE_BETA_ALPHA, PURCHASE_RATE_BETA_BETA, size=n_trials)
    return PURCHASE_RATE_MIN + raw * (PURCHASE_RATE_MAX - PURCHASE_RATE_MIN)

# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────

def run_simulation(order_size, order_cost, n_trials=N_TRIALS):
    upper_bowl       = sample_upper_bowl(n_trials)
    total_attendance = LOWER_BOWL_ATTENDANCE + upper_bowl
    purchase_rate    = sample_purchase_rate(n_trials)

    shirts_demanded  = (total_attendance * purchase_rate).astype(int)
    shirts_sold      = np.minimum(shirts_demanded, order_size)
    shirts_leftover  = np.maximum(order_size - shirts_demanded, 0)

    revenue_primary  = shirts_sold     * SELL_PRICE_PER_SHIRT
    revenue_salvage  = shirts_leftover * SALVAGE_PRICE_PER_SHIRT
    total_revenue    = revenue_primary + revenue_salvage

    return total_revenue - order_cost

# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def generate_report(results_by_order):
    print("\n─────────────────────────────────────────────────────────────────────────")
    print(f"  AZTECA T-SHIRT MONTE CARLO REPORT  ({N_TRIALS:,} simulations)")
    print("─────────────────────────────────────────────────────────────────────────")
    print(f"\n  Distributions used:")
    print(f"    Upper bowl attendance : Triangular(min=30k, mode=60k, max=90k)")
    print(f"    Purchase rate         : Beta(α={PURCHASE_RATE_BETA_ALPHA}, β={PURCHASE_RATE_BETA_BETA})"
          f" scaled to [{PURCHASE_RATE_MIN*100:.0f}%, {PURCHASE_RATE_MAX*100:.0f}%]  →  mean ≈ 9%")
    print(f"    Lower bowl            : {LOWER_BOWL_ATTENDANCE:,} (fixed)")
    print(f"    Sell price            : ${SELL_PRICE_PER_SHIRT:,.0f} MXN/shirt")
    print(f"    Salvage price         : ${SALVAGE_PRICE_PER_SHIRT} MXN/shirt\n")

    header = f"{'Metric':<28}" + "".join(f"{'Order '+str(o['size'])+' shirts':>22}" for o in ORDER_OPTIONS)
    print(header)
    print("─" * (28 + 22 * len(ORDER_OPTIONS)))

    metrics = [
        ("Avg. Profit (MXN)",      lambda r: f"${r.mean():>18,.0f}"),
        ("Std. Dev / Risk (MXN)",  lambda r: f"${r.std():>18,.0f}"),
        ("5th Pct  (Worst case)",  lambda r: f"${np.percentile(r,  5):>18,.0f}"),
        ("Median",                 lambda r: f"${np.percentile(r, 50):>18,.0f}"),
        ("95th Pct (Best case)",   lambda r: f"${np.percentile(r, 95):>18,.0f}"),
        ("Prob. of Profit",        lambda r: f"{(r > 0).mean()*100:>18.1f}%"),
    ]

    for label, fmt in metrics:
        row = f"{label:<28}" + "".join(fmt(results_by_order[o['size']]) for o in ORDER_OPTIONS)
        print(row)

    print("─" * (28 + 22 * len(ORDER_OPTIONS)))

    best = max(ORDER_OPTIONS, key=lambda o: results_by_order[o['size']].mean())
    print(f"\n  ✔  RECOMMENDATION: Order {best['size']:,} shirts")
    print(f"     Expected profit : ${results_by_order[best['size']].mean():,.0f} MXN")
    print(f"     Order cost      : ${best['cost']:,.0f} MXN")
    print("─────────────────────────────────────────────────────────────────────────\n")

# ─────────────────────────────────────────────
# Visualisation — 3 panels
# ─────────────────────────────────────────────

def generate_visual(results_by_order, output_filename="azteca_tshirt_plot.png"):
    print(f"Saving plot to {output_filename}...")

    palette = sns.color_palette("Set2", len(ORDER_OPTIONS))
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Azteca Gameday T-Shirt Monte Carlo Analysis', fontsize=16, fontweight='bold')

    # ── Panel 1: Input — Attendance distribution ──
    ax = axes[0]
    x_att = np.linspace(UPPER_BOWL_MIN, UPPER_BOWL_MAX, 500)
    triang = stats.triang(
        c=(UPPER_BOWL_MODE - UPPER_BOWL_MIN) / (UPPER_BOWL_MAX - UPPER_BOWL_MIN),
        loc=UPPER_BOWL_MIN,
        scale=UPPER_BOWL_MAX - UPPER_BOWL_MIN
    )
    ax.plot(x_att / 1000, triang.pdf(x_att), color='steelblue', linewidth=2)
    ax.fill_between(x_att / 1000, triang.pdf(x_att), alpha=0.3, color='steelblue')
    # Overlay original discrete estimates as red dots for comparison
    discrete_att  = [30_000, 60_000, 90_000]
    discrete_prob = [0.23, 0.50, 0.27]
    scale_factor  = triang.pdf(60_000) / 0.50  # align peak heights
    ax.scatter([a/1000 for a in discrete_att],
               [p * scale_factor for p in discrete_prob],
               color='red', zorder=5, label='Original estimates', s=60)
    ax.set_title('Upper Bowl Attendance\nTriangular Distribution', fontsize=12)
    ax.set_xlabel('Attendance (thousands)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    # ── Panel 2: Input — Purchase rate distribution ──
    ax = axes[1]
    x_raw    = np.linspace(0, 1, 500)
    x_scaled = PURCHASE_RATE_MIN + x_raw * (PURCHASE_RATE_MAX - PURCHASE_RATE_MIN)
    beta_pdf = stats.beta.pdf(x_raw, PURCHASE_RATE_BETA_ALPHA, PURCHASE_RATE_BETA_BETA)
    ax.plot(x_scaled * 100, beta_pdf, color='darkorange', linewidth=2)
    ax.fill_between(x_scaled * 100, beta_pdf, alpha=0.3, color='darkorange')
    # Overlay original discrete estimates
    discrete_rates = [0.05, 0.10, 0.15]
    discrete_rprob = [0.30, 0.60, 0.10]
    beta_peak      = stats.beta.pdf(1/3, PURCHASE_RATE_BETA_ALPHA, PURCHASE_RATE_BETA_BETA)
    r_scale_factor = beta_peak / 0.60
    ax.scatter([r * 100 for r in discrete_rates],
               [p * r_scale_factor for p in discrete_rprob],
               color='red', zorder=5, label='Original estimates', s=60)
    ax.set_title('Purchase Rate\nBeta(α=2, β=3) scaled to [5%–15%]', fontsize=12)
    ax.set_xlabel('Purchase Rate (%)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    # ── Panel 3: Output — Profit distributions ──
    ax = axes[2]
    for i, order in enumerate(ORDER_OPTIONS):
        results = results_by_order[order['size']]
        label   = f"{order['size']:,} shirts  (mean: ${results.mean():,.0f})"
        sns.kdeplot(results, ax=ax, fill=True, alpha=0.4, color=palette[i], label=label)
        ax.axvline(results.mean(), color=palette[i], linestyle='--', linewidth=1.5)
    ax.axvline(0, color='red', linestyle='-', linewidth=1.2, label='Break-even')
    ax.set_title('Profit Distribution\nby Order Size', fontsize=12)
    ax.set_xlabel('Profit (MXN)')
    ax.set_ylabel('Density')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    print("Plot saved.\n")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_tshirt_simulation():
    print("Starting Azteca T-Shirt Monte Carlo Simulation...")

    results_by_order = {}
    for order in ORDER_OPTIONS:
        results_by_order[order['size']] = run_simulation(order['size'], order['cost'])

    generate_report(results_by_order)
    generate_visual(results_by_order)

if __name__ == "__main__":
    run_tshirt_simulation()