"""
===============================================================================
MONTE CARLO SIMULATION — NYLON PRODUCTION OFFER DECISION
Textile Plant | May Production Run
===============================================================================

DECISION: Accept or reject an offer to sell 5,000 kg of 15/1g Nylon at 182/kg,
          coming entirely out of May's production run.

TWO SCENARIOS COMPARED PER TRIAL:
  SCENARIO A (REJECT): Sell all 15/1g production at the simulated market price.
  SCENARIO B (ACCEPT): Lock in 5,000 kg at 182/kg; remaining 15/1g at market price.

In both scenarios, the production mix (kg of 15/1g vs 84/21fd) is OPTIMISED
each trial to maximise total profit subject to all constraints.

CONSTRAINTS:
  - Extruder machine:   300 hours/month (shared)
  - Drawstring machine: 1,600 hours/month (shared)
  - 15/1g production:   min 3,000 kg  |  max 10,000 kg
  - 84/21fd production: min 6,000 kg  |  max 15,000 kg

THROUGHPUT RATES:
  - Extruder:    15/1g = 30 kg/hr,  84/21fd = 60 kg/hr
  - Drawstring:  15/1g = 15 kg/hr,  84/21fd = 12 kg/hr

PRICES & COSTS:
  - 15/1g:   selling price UNCERTAIN (normal dist), raw material cost = 88/kg
  - 84/21fd: selling price FIXED at 123.94/kg,      raw material cost = 74/kg
  - Offer:   5,000 kg of 15/1g at 182/kg (floor; remainder sold at market)

PRICE MODEL FOR 15/1g (plain normal, no truncation):
  - Forecast target: 178/kg
  - Mean:   97.65% × 178 = 173.817/kg
  - Std Dev: 9.09% × 178 =  16.180/kg
  - Min/Max observed: 71.43% × 178 = 127.145 / 106.45% × 178 = 189.481
    (used for reference only — distribution is unbounded)

SURPLUS RULE: Any 15/1g produced beyond committed sales is sold at the
              simulated market price for that trial.
===============================================================================
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
import subprocess
import os
warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
N_TRIALS = 100_000

# ── Constants ─────────────────────────────────────────────────────────────────
# Machine capacities
EXTRUDER_HRS   = 300    # hours/month
DRAWSTRING_HRS = 1_600  # hours/month

# Production bounds (kg)
MIN_15   =  3_000;  MAX_15   = 10_000
MIN_84   =  6_000;  MAX_84   = 15_000

# Throughput rates (kg/hr)
EXT_RATE_15  = 30;  EXT_RATE_84  = 60   # extruder
DS_RATE_15   = 15;  DS_RATE_84   = 12   # drawstring

# Costs (raw material only, /kg)
COST_15  = 88
COST_84  = 74

# Prices
PRICE_84      = 123.94          # fixed
OFFER_PRICE   = 182.0           # locked-in price for 5,000 kg
OFFER_QTY     = 5_000           # kg committed under offer

# 15/1g market price distribution (plain normal)
PRICE_15_MEAN  = 0.9765 * 178   # = 173.817
PRICE_15_STD   = 0.0909 * 178   # =  16.180

# ── Price simulation ──────────────────────────────────────────────────────────
market_prices_15 = np.random.normal(PRICE_15_MEAN, PRICE_15_STD, N_TRIALS)

# ── LP-based production optimiser ─────────────────────────────────────────────
def optimise_production(price_15, scenario):
    """
    Solve a linear programme to find the profit-maximising production quantities
    of 15/1g (x1) and 84/21fd (x2) subject to all constraints.

    Variables: x = [x1, x2]  (kg produced of each product)

    Objective (maximise profit — linprog minimises, so negate):
      REJECT scenario:  profit = (price_15 - COST_15)*x1 + (PRICE_84 - COST_84)*x2
      ACCEPT scenario:  first OFFER_QTY kg of x1 earn OFFER_PRICE, rest earn price_15
                        profit = OFFER_PRICE*min(x1,OFFER_QTY) + price_15*max(x1-OFFER_QTY,0)
                               - COST_15*x1 + (PRICE_84 - COST_84)*x2
                        Since min(x1,OFFER_QTY) = OFFER_QTY when x1 >= OFFER_QTY (enforced
                        by the floor constraint), the marginal revenue on every kg of x1
                        beyond OFFER_QTY is price_15. The OFFER_QTY kg locked at 182 provide
                        a fixed bonus we add post-optimisation. So the LP margin for x1 is:
                          (price_15 - COST_15) for all x1, plus fixed bonus added later.

    Constraints:
      Extruder:    x1/EXT_RATE_15  + x2/EXT_RATE_84  <= EXTRUDER_HRS
      Drawstring:  x1/DS_RATE_15   + x2/DS_RATE_84   <= DRAWSTRING_HRS
      Min/max bounds on x1 and x2.
      ACCEPT only: x1 >= OFFER_QTY  (must produce at least the committed floor)
    """
    margin_15 = price_15 - COST_15
    margin_84 = PRICE_84 - COST_84

    # linprog minimises, so negate margins
    c = [-margin_15, -margin_84]

    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = [
        [1/EXT_RATE_15,  1/EXT_RATE_84],   # extruder hours
        [1/DS_RATE_15,   1/DS_RATE_84],    # drawstring hours
    ]
    b_ub = [EXTRUDER_HRS, DRAWSTRING_HRS]

    # Bounds
    lb_15 = max(MIN_15, OFFER_QTY) if scenario == "accept" else MIN_15
    bounds = [(lb_15, MAX_15), (MIN_84, MAX_84)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        x1, x2 = result.x
    else:
        # Fallback to minimums if LP fails (shouldn't happen)
        x1 = lb_15
        x2 = MIN_84

    return x1, x2


def compute_profit(x1, x2, price_15, scenario):
    """
    Compute total profit given production quantities and simulated price.
    Revenue: all x1 sold (offer floor at 182, remainder at market price).
    Cost: raw material only.
    """
    cost = COST_15 * x1 + COST_84 * x2

    if scenario == "reject":
        revenue_15 = price_15 * x1
    else:
        # First OFFER_QTY kg at 182, remainder at market price
        offer_kg    = min(x1, OFFER_QTY)
        market_kg   = max(x1 - OFFER_QTY, 0)
        revenue_15  = OFFER_PRICE * offer_kg + price_15 * market_kg

    revenue_84 = PRICE_84 * x2
    return (revenue_15 + revenue_84) - cost


# ── Run simulation ─────────────────────────────────────────────────────────────
print("Running Monte Carlo simulation — 100,000 trials...")
print("This may take ~30–60 seconds due to LP optimisation per trial.\n")

profits_reject = np.zeros(N_TRIALS)
profits_accept = np.zeros(N_TRIALS)
prod_15_reject = np.zeros(N_TRIALS)
prod_15_accept = np.zeros(N_TRIALS)
prod_84_reject = np.zeros(N_TRIALS)
prod_84_accept = np.zeros(N_TRIALS)

for i in range(N_TRIALS):
    p = market_prices_15[i]

    # REJECT scenario
    x1r, x2r = optimise_production(p, "reject")
    profits_reject[i] = compute_profit(x1r, x2r, p, "reject")
    prod_15_reject[i] = x1r
    prod_84_reject[i] = x2r

    # ACCEPT scenario
    x1a, x2a = optimise_production(p, "accept")
    profits_accept[i] = compute_profit(x1a, x2a, p, "accept")
    prod_15_accept[i] = x1a
    prod_84_accept[i] = x2a

    if (i + 1) % 10_000 == 0:
        print(f"  {i+1:,} / {N_TRIALS:,} trials complete...")

print("\nSimulation complete. Generating results...\n")

# ── Derived metrics ────────────────────────────────────────────────────────────
profit_diff = profits_accept - profits_reject   # positive = accept is better

pct_accept_wins  = np.mean(profit_diff > 0) * 100
pct_reject_wins  = np.mean(profit_diff < 0) * 100
pct_tie          = np.mean(profit_diff == 0) * 100

mean_price_15    = np.mean(market_prices_15)
std_price_15     = np.std(market_prices_15)

# ── Print results ──────────────────────────────────────────────────────────────
DIVIDER = "=" * 70

print(DIVIDER)
print("  MONTE CARLO RESULTS — NYLON PRODUCTION OFFER DECISION")
print(DIVIDER)

print("\n📊 SIMULATED 15/1g MARKET PRICE (per kg)")
print(f"   Mean:              {mean_price_15:>10.4f}")
print(f"   Std Dev:           {std_price_15:>10.4f}")
print(f"   5th Percentile:    {np.percentile(market_prices_15, 5):>10.4f}")
print(f"   25th Percentile:   {np.percentile(market_prices_15, 25):>10.4f}")
print(f"   Median:            {np.percentile(market_prices_15, 50):>10.4f}")
print(f"   75th Percentile:   {np.percentile(market_prices_15, 75):>10.4f}")
print(f"   95th Percentile:   {np.percentile(market_prices_15, 95):>10.4f}")

print("\n" + DIVIDER)
print("  SCENARIO A — REJECT OFFER (sell all 15/1g at market price)")
print(DIVIDER)
print(f"   Mean Profit:       {np.mean(profits_reject):>14,.2f}")
print(f"   Std Dev:           {np.std(profits_reject):>14,.2f}")
print(f"   5th Percentile:    {np.percentile(profits_reject, 5):>14,.2f}")
print(f"   25th Percentile:   {np.percentile(profits_reject, 25):>14,.2f}")
print(f"   Median:            {np.percentile(profits_reject, 50):>14,.2f}")
print(f"   75th Percentile:   {np.percentile(profits_reject, 75):>14,.2f}")
print(f"   95th Percentile:   {np.percentile(profits_reject, 95):>14,.2f}")
print(f"   Avg 15/1g Prod:    {np.mean(prod_15_reject):>14,.1f} kg")
print(f"   Avg 84/21fd Prod:  {np.mean(prod_84_reject):>14,.1f} kg")

print("\n" + DIVIDER)
print("  SCENARIO B — ACCEPT OFFER (5,000 kg at 182, remainder at market)")
print(DIVIDER)
print(f"   Mean Profit:       {np.mean(profits_accept):>14,.2f}")
print(f"   Std Dev:           {np.std(profits_accept):>14,.2f}")
print(f"   5th Percentile:    {np.percentile(profits_accept, 5):>14,.2f}")
print(f"   25th Percentile:   {np.percentile(profits_accept, 25):>14,.2f}")
print(f"   Median:            {np.percentile(profits_accept, 50):>14,.2f}")
print(f"   75th Percentile:   {np.percentile(profits_accept, 75):>14,.2f}")
print(f"   95th Percentile:   {np.percentile(profits_accept, 95):>14,.2f}")
print(f"   Avg 15/1g Prod:    {np.mean(prod_15_accept):>14,.1f} kg")
print(f"   Avg 84/21fd Prod:  {np.mean(prod_84_accept):>14,.1f} kg")

print("\n" + DIVIDER)
print("  PROFIT DIFFERENTIAL (Accept minus Reject)")
print(DIVIDER)
print(f"   Mean Difference:   {np.mean(profit_diff):>14,.2f}")
print(f"   Std Dev:           {np.std(profit_diff):>14,.2f}")
print(f"   5th Percentile:    {np.percentile(profit_diff, 5):>14,.2f}")
print(f"   Median:            {np.percentile(profit_diff, 50):>14,.2f}")
print(f"   95th Percentile:   {np.percentile(profit_diff, 95):>14,.2f}")

print("\n" + DIVIDER)
print("  DECISION PROBABILITIES")
print(DIVIDER)
print(f"   Accept beats Reject:   {pct_accept_wins:>7.2f}% of trials")
print(f"   Reject beats Accept:   {pct_reject_wins:>7.2f}% of trials")
print(f"   Tie:                   {pct_tie:>7.2f}% of trials")

mean_diff = np.mean(profit_diff)
if mean_diff > 0:
    rec = "ACCEPT THE OFFER"
    reason = f"Accepting the offer generates on average {mean_diff:,.2f} more profit per month."
else:
    rec = "REJECT THE OFFER"
    reason = f"Rejecting the offer generates on average {abs(mean_diff):,.2f} more profit per month."

print("\n" + DIVIDER)
print(f"  ✅ RECOMMENDATION: {rec}")
print(f"  {reason}")
print(DIVIDER)

# ── Plotting ───────────────────────────────────────────────────────────────────
REJECT_COL = "#2196F3"   # blue
ACCEPT_COL = "#F44336"   # red
DIFF_COL   = "#4CAF50"   # green

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0D1117")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

LABEL_KW  = dict(color="white", fontsize=11)
TITLE_KW  = dict(color="white", fontsize=13, fontweight="bold", pad=10)
TICK_KW   = dict(colors="white")
SPINE_COL = "#444"

def style_ax(ax):
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

# ── Plot 1: Profit distributions (histogram overlay) ──────────────────────────
ax1 = fig.add_subplot(gs[0, :])
bins = np.linspace(
    min(profits_reject.min(), profits_accept.min()),
    max(profits_reject.max(), profits_accept.max()),
    120
)
ax1.hist(profits_reject, bins=bins, alpha=0.6, color=REJECT_COL, label="Reject Offer", density=True)
ax1.hist(profits_accept, bins=bins, alpha=0.6, color=ACCEPT_COL, label="Accept Offer", density=True)
ax1.axvline(np.mean(profits_reject), color=REJECT_COL, lw=2, linestyle="--",
            label=f"Reject Mean: {np.mean(profits_reject):,.0f}")
ax1.axvline(np.mean(profits_accept), color=ACCEPT_COL, lw=2, linestyle="--",
            label=f"Accept Mean: {np.mean(profits_accept):,.0f}")
ax1.set_title("Profit Distribution — Accept vs. Reject (100,000 Trials)", **TITLE_KW)
ax1.set_xlabel("Total Monthly Profit", **LABEL_KW)
ax1.set_ylabel("Density", **LABEL_KW)
ax1.legend(facecolor="#1C2128", labelcolor="white", fontsize=10)
style_ax(ax1)

# ── Plot 2: Profit differential distribution ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(profit_diff, bins=100, color=DIFF_COL, alpha=0.8, density=True)
ax2.axvline(0, color="white", lw=1.5, linestyle=":")
ax2.axvline(np.mean(profit_diff), color="yellow", lw=2, linestyle="--",
            label=f"Mean: {np.mean(profit_diff):,.0f}")
ax2.set_title("Profit Differential\n(Accept − Reject)", **TITLE_KW)
ax2.set_xlabel("Differential", **LABEL_KW)
ax2.set_ylabel("Density", **LABEL_KW)
ax2.legend(facecolor="#1C2128", labelcolor="white", fontsize=9)
# shade accept-wins region
pos_mask = profit_diff > 0
ax2.fill_between(
    np.sort(profit_diff[pos_mask]),
    0,
    np.zeros_like(profit_diff[pos_mask]),
    alpha=0   # just for label; shading done via hist already
)
style_ax(ax2)

# ── Plot 3: Simulated 15/1g price distribution ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(market_prices_15, bins=100, color="#9C27B0", alpha=0.8, density=True)
ax3.axvline(178,              color="white",  lw=1.5, linestyle=":",  label="Target (178)")
ax3.axvline(PRICE_15_MEAN,    color="yellow", lw=2,   linestyle="--", label=f"Mean ({PRICE_15_MEAN:.2f})")
ax3.axvline(OFFER_PRICE,      color="orange", lw=2,   linestyle="-",  label=f"Offer price (182)")
ax3.set_title("Simulated 15/1g Market Price Distribution", **TITLE_KW)
ax3.set_xlabel("Price per kg", **LABEL_KW)
ax3.set_ylabel("Density", **LABEL_KW)
ax3.legend(facecolor="#1C2128", labelcolor="white", fontsize=9)
style_ax(ax3)

# ── Plot 4: CDF comparison ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
sorted_r = np.sort(profits_reject)
sorted_a = np.sort(profits_accept)
cdf      = np.linspace(0, 1, N_TRIALS)
ax4.plot(sorted_r, cdf, color=REJECT_COL, lw=2, label="Reject")
ax4.plot(sorted_a, cdf, color=ACCEPT_COL, lw=2, label="Accept")
ax4.axhline(0.05, color="#888", lw=1, linestyle=":")
ax4.axhline(0.50, color="#888", lw=1, linestyle=":")
ax4.axhline(0.95, color="#888", lw=1, linestyle=":")
ax4.set_title("Cumulative Profit Distribution (CDF)", **TITLE_KW)
ax4.set_xlabel("Total Monthly Profit", **LABEL_KW)
ax4.set_ylabel("Cumulative Probability", **LABEL_KW)
ax4.legend(facecolor="#1C2128", labelcolor="white", fontsize=10)
style_ax(ax4)

# ── Plot 5: Decision summary bar chart ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
categories = ["Accept\nWins", "Reject\nWins", "Tie"]
values     = [pct_accept_wins, pct_reject_wins, pct_tie]
colours    = [ACCEPT_COL, REJECT_COL, "#888888"]
bars = ax5.bar(categories, values, color=colours, alpha=0.85, width=0.5)
for bar, val in zip(bars, values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=12,
             fontweight="bold")
ax5.set_title("Decision Probabilities\n(% of Trials)", **TITLE_KW)
ax5.set_ylabel("% of Trials", **LABEL_KW)
ax5.set_ylim(0, max(values) * 1.2)
style_ax(ax5)

# ── Main title ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "Monte Carlo Simulation — 15/1g Nylon Offer Decision | May Production Run\n"
    f"n = {N_TRIALS:,} trials  |  Recommendation: {rec}",
    color="white", fontsize=15, fontweight="bold", y=0.98
)
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path   = os.path.join(script_dir, "nylon_monte_carlo_v2.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

print("\nChart saved to:", out_path)
plt.close()
print("Done.")

# ── Auto-push outputs to GitHub ───────────────────────────────────────────────
REPO_PATH   = ".santi-heskey/Darden/main/DA1/DA11"   # change this to your local repo path if needed
OUTPUT_FILES = [
    "nylon_monte_carlo_v2.py",
    "nylon_monte_carlo_v2.png"
]

os.chdir(REPO_PATH)

for f in OUTPUT_FILES:
    subprocess.run(["git", "add", f], check=True)

subprocess.run(["git", "commit", "-m", "Auto: update Monte Carlo simulation outputs"], check=True)
subprocess.run(["git", "push", "origin", "main"], check=True)   # change branch if needed

print("Outputs pushed to GitHub.")