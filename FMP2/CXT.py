#!/usr/bin/env python3
"""
Fuel Hedge Cost Model
=====================
Models all-in fuel cost under three hedge strategies, using a market premium
schedule for cap and floor contracts.

CONTRACT MECHANICS
──────────────────
  Swap    : Pay swap_ask per gallon, no market exposure.
            All-in cost = swap_ask

  Cap     : Pay cap_premium upfront. If spot > strike, counterparty pays
            the difference → you are capped at strike price.
            All-in cost/gal = min(spot, cap_strike) + cap_premium

  Collar  : Buy a Cap (pay cap_premium) + Sell a Floor (receive floor_premium).
            Net premium = cap_premium − floor_premium (usually positive/small).
            Spot is bounded between floor_strike and cap_strike.
            All-in cost/gal = clamp(spot, floor_strike, cap_strike) + net_premium

  Unhedged: Pay spot price outright.

INPUTS
──────
  - Swap bid/ask
  - Premium schedule: for each available strike, the cap and floor premium/gal
  - Cap strike selection (from schedule)
  - Floor strike selection (from schedule, for Collar)
  - Forecast volume and hedge ratio
  - Spot price tranches (different price scenarios or delivery periods)

CLI usage:
  python fuel_hedge.py --interactive
  python fuel_hedge.py \\
      --swap-bid 3.742 --swap-ask 3.758 \\
      --cap-strike 3.80 --floor-strike 3.60 \\
      --volume 100000 --hedge-ratio 75 \\
      --tranches 4 \\
      --prices 3.20 3.80 4.20 4.60 \\
      --vol-splits 25 25 25 25
"""

import argparse
import sys
from typing import List, Dict

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN PREMIUM SCHEDULE
# Keys are strike prices; values are (cap_premium, floor_premium) per gallon.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SCHEDULE: Dict[float, tuple] = {
    3.50: (0.346, 0.065),
    3.60: (0.276, 0.097),
    3.70: (0.216, 0.136),
    3.80: (0.164, 0.182),
    3.90: (0.122, 0.241),
    4.00: (0.090, 0.307),
}

SWAP_BID_DEFAULT = 3.742
SWAP_ASK_DEFAULT = 3.758


# ─────────────────────────────────────────────────────────────────────────────
# HEDGE STRATEGY FUNCTIONS  (return all-in effective price per gallon)
# ─────────────────────────────────────────────────────────────────────────────

def aip_unhedged(spot: float) -> float:
    return spot

def aip_swap(swap_ask: float) -> float:
    return swap_ask

def aip_cap(spot: float, cap_strike: float, cap_premium: float) -> float:
    """Cap: pay min(spot, strike) + upfront premium amortised per gallon."""
    return min(spot, cap_strike) + cap_premium

def aip_collar(spot: float, cap_strike: float, cap_premium: float,
               floor_strike: float, floor_premium: float) -> float:
    """Collar: pay clamp(spot, floor, cap) + net premium."""
    net_premium = cap_premium - floor_premium
    return max(floor_strike, min(spot, cap_strike)) + net_premium


def cost_for_tranche(total_vol: float, spot: float, hedge_ratio: float,
                     swap_ask: float,
                     cap_strike: float, cap_premium: float,
                     floor_strike: float, floor_premium: float) -> dict:
    """
    Blended cost for one spot price tranche.
    hedged_vol  → under the chosen strategy
    exposed_vol → always at spot
    """
    hedged_vol  = total_vol * hedge_ratio
    exposed_vol = total_vol * (1.0 - hedge_ratio)

    ep_swap   = aip_swap(swap_ask)
    ep_cap    = aip_cap(spot, cap_strike, cap_premium)
    ep_collar = aip_collar(spot, cap_strike, cap_premium, floor_strike, floor_premium)

    def blended_cost(ep):
        return ep * hedged_vol + spot * exposed_vol

    def blended_ep(ep):
        return blended_cost(ep) / total_vol if total_vol > 0 else ep

    return {
        "spot":         spot,
        "total_vol":    total_vol,
        "hedged_vol":   hedged_vol,
        "exposed_vol":  exposed_vol,
        "hedge_ratio":  hedge_ratio,
        # all-in effective prices (hedged leg)
        "aip_unhedged": spot,
        "aip_swap":     ep_swap,
        "aip_cap":      ep_cap,
        "aip_collar":   ep_collar,
        # blended effective prices (across full tranche volume)
        "bep_unhedged": spot,
        "bep_swap":     blended_ep(ep_swap),
        "bep_cap":      blended_ep(ep_cap),
        "bep_collar":   blended_ep(ep_collar),
        # total costs
        "cost_unhedged": spot     * total_vol,
        "cost_swap":     blended_cost(ep_swap),
        "cost_cap":      blended_cost(ep_cap),
        "cost_collar":   blended_cost(ep_collar),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def banner():
    print(f"\n{BOLD}{CYAN}{'='*78}")
    print("  FUEL HEDGE COST MODEL  |  Transportation Company")
    print(f"{'='*78}{RESET}\n")

def fmt_usd(val: float) -> str:  return f"${val:>14,.2f}"
def fmt_p(val: float)   -> str:  return f"${val:.4f}"
def fmt_pct(val: float) -> str:  return f"{val*100:.1f}%"

def section(title: str):
    print(f"\n{BOLD}{YELLOW}{'-'*78}")
    print(f"  {title}")
    print(f"{'-'*78}{RESET}")


def print_schedule(schedule: Dict[float, tuple],
                   cap_strike: float, floor_strike: float,
                   swap_bid: float, swap_ask: float):
    section("MARKET DATA  —  Swap & Premium Schedule")
    print(f"\n  {'Swap Bid:':<20} {fmt_p(swap_bid)}    {'Swap Ask:':<20} {fmt_p(swap_ask)}")
    print(f"  {DIM}(you transact at the ask: {fmt_p(swap_ask)}){RESET}\n")

    print(f"  {'Strike':<12}{'Cap Premium':>14}{'Floor Premium':>16}{'Net (Cap−Floor)':>18}  {'Selected'}")
    print(f"  {'-'*64}")
    for strike in sorted(schedule):
        cp, fp = schedule[strike]
        net    = cp - fp
        tags   = []
        if strike == cap_strike:   tags.append(f"{BOLD}{CYAN}← CAP STRIKE{RESET}")
        if strike == floor_strike: tags.append(f"{BOLD}{GREEN}← FLOOR STRIKE{RESET}")
        tag_str = "  " + "  ".join(tags) if tags else ""
        print(f"  {fmt_p(strike):<12}{fmt_p(cp):>14}{fmt_p(fp):>16}{fmt_p(net):>18}{tag_str}")


def print_contract_summary(swap_ask: float,
                            cap_strike: float, cap_premium: float,
                            floor_strike: float, floor_premium: float,
                            hedge_ratio: float, total_vol: float):
    section("SELECTED CONTRACTS")
    hedged_vol  = total_vol * hedge_ratio
    net_premium = cap_premium - floor_premium

    print(f"\n  {BOLD}Swap{RESET}")
    print(f"    Execution price :  {fmt_p(swap_ask)}/gal")

    print(f"\n  {BOLD}Cap{RESET}")
    print(f"    Strike          :  {fmt_p(cap_strike)}/gal  (you pay max this for fuel)")
    print(f"    Premium         :  {fmt_p(cap_premium)}/gal  (upfront cost)")
    print(f"    All-in ceiling  :  {fmt_p(cap_strike + cap_premium)}/gal")
    print(f"    Total premium   :  {fmt_usd(cap_premium * hedged_vol)}  ({hedged_vol:,.0f} gal hedged)")

    print(f"\n  {BOLD}Collar{RESET}")
    print(f"    Cap strike      :  {fmt_p(cap_strike)}/gal  (ceiling)")
    print(f"    Floor strike    :  {fmt_p(floor_strike)}/gal  (you sell this floor)")
    print(f"    Cap premium     : +{fmt_p(cap_premium)}/gal  (you pay)")
    print(f"    Floor premium   : -{fmt_p(floor_premium)}/gal  (you receive)")
    print(f"    Net premium     :  {fmt_p(net_premium)}/gal  (net cost to you)")
    print(f"    All-in range    :  {fmt_p(floor_strike + net_premium)}/gal  →  {fmt_p(cap_strike + net_premium)}/gal")
    print(f"    Total net prem  :  {fmt_usd(net_premium * hedged_vol)}  ({hedged_vol:,.0f} gal hedged)")

    print(f"\n  {BOLD}Hedge Ratio{RESET}        :  {fmt_pct(hedge_ratio)}  ({hedged_vol:,.0f} gal of {total_vol:,.0f} gal forecast)")
    print(f"  {BOLD}Exposed Volume{RESET}     :  {fmt_pct(1-hedge_ratio)}  ({total_vol*(1-hedge_ratio):,.0f} gal purchased at spot){RESET}")


def print_tranche_table(tranches: List[dict]):
    section("ALL-IN COST BY TRANCHE  (hedged leg + exposed leg at spot)")
    col = 15
    print(f"  {'T#':<4}{'Spot':>10}{'Vol (gal)':>12}{'Hedge%':>8}  "
          f"{'Unhedged':>{col}}{'Swap':>{col}}{'Cap':>{col}}{'Collar':>{col}}")
    print(f"  {'-'*96}")
    for i, t in enumerate(tranches, 1):
        print(f"  {i:<4}{fmt_p(t['spot']):>10}{t['total_vol']:>12,.0f}{fmt_pct(t['hedge_ratio']):>8}  "
              f"{fmt_usd(t['cost_unhedged']):>{col}}{fmt_usd(t['cost_swap']):>{col}}"
              f"{fmt_usd(t['cost_cap']):>{col}}{fmt_usd(t['cost_collar']):>{col}}")


def _best(totals: dict) -> str:
    return min(("cost_swap","cost_cap","cost_collar"), key=lambda k: totals[k])


def print_summary(tranches: List[dict], swap_ask: float, hedge_ratio: float,
                  cap_strike: float, cap_premium: float,
                  floor_strike: float, floor_premium: float):
    section("PORTFOLIO SUMMARY")
    keys   = ("total_vol","hedged_vol","exposed_vol",
               "cost_unhedged","cost_swap","cost_cap","cost_collar")
    totals = {k: sum(t[k] for t in tranches) for k in keys}
    avg_spot = sum(t["spot"]*t["total_vol"] for t in tranches) / totals["total_vol"]
    best     = _best(totals)
    net_prem = cap_premium - floor_premium

    col = 18
    print(f"\n  {'Total Forecast Volume:':<36} {totals['total_vol']:>12,.0f} gal")
    print(f"  {'Hedged Volume:':<36} {totals['hedged_vol']:>12,.0f} gal  ({fmt_pct(hedge_ratio)})")
    print(f"  {'Exposed Volume (at spot):':<36} {totals['exposed_vol']:>12,.0f} gal  ({fmt_pct(1-hedge_ratio)})")
    print(f"  {'Volume-Weighted Avg Spot:':<36} {fmt_p(avg_spot)}")

    print(f"\n  {'Strategy':<14}{'Total Cost':>{col}}{'Blended $/gal':>{col}}"
          f"{'vs Unhedged':>{col}}  {'Notes'}")
    print(f"  {'-'*82}")

    rows = [
        ("Unhedged", "cost_unhedged", "no contract cost"),
        ("Swap",     "cost_swap",     f"ask {fmt_p(swap_ask)}, no premium"),
        ("Cap",      "cost_cap",      f"strike {fmt_p(cap_strike)}, prem {fmt_p(cap_premium)}/gal"),
        ("Collar",   "cost_collar",   f"cap {fmt_p(cap_strike)}/floor {fmt_p(floor_strike)}, net prem {fmt_p(net_prem)}/gal"),
    ]
    for label, key, note in rows:
        total  = totals[key]
        bep    = total / totals["total_vol"]
        delta  = total - totals["cost_unhedged"]
        if delta < 0:
            d_str = f"{GREEN}  {fmt_usd(delta).strip()}{RESET}"
        elif delta == 0:
            d_str = f"  {fmt_usd(delta).strip()}"
        else:
            d_str = f"{RED} +{fmt_usd(delta).strip()}{RESET}"
        marker = f"  {BOLD}<-- BEST{RESET}" if key == best else ""
        print(f"  {label:<14}{fmt_usd(total):>{col}}{fmt_p(bep):>{col}}   {d_str}  {DIM}{note}{RESET}{marker}")


def print_scenario_table(swap_ask: float, hedge_ratio: float,
                         cap_strike: float, cap_premium: float,
                         floor_strike: float, floor_premium: float,
                         avg_spot: float):
    """Show all-in blended effective price across a spot range."""
    section("SCENARIO TABLE  —  Blended Effective Price at Different Spot Levels")
    h, e   = hedge_ratio, 1.0 - hedge_ratio
    net_p  = cap_premium - floor_premium
    lo     = avg_spot * 0.70
    hi     = avg_spot * 1.40
    spots  = [round(avg_spot * m, 4) for m in
              [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.40]]

    print(f"\n  {DIM}Blended = {fmt_pct(h)} × all_in_strategy_price  +  {fmt_pct(e)} × spot{RESET}")
    print(f"  {DIM}Cap all-in ceiling : {fmt_p(cap_strike + cap_premium)}/gal  |  "
          f"Collar range: {fmt_p(floor_strike + net_p)} – {fmt_p(cap_strike + net_p)}/gal{RESET}\n")

    col = 13
    print(f"  {'Spot':>{col}}{'Unhedged':>{col}}{'Swap':>{col}}{'Cap (AIP)':>{col}}{'Collar (AIP)':>{col}}")
    print(f"  {'-'*65}")
    for s in spots:
        u  = s
        sw = aip_swap(swap_ask)                            * h + s * e
        c  = aip_cap(s, cap_strike, cap_premium)           * h + s * e
        k  = aip_collar(s, cap_strike, cap_premium,
                           floor_strike, floor_premium)    * h + s * e
        tag = f"  {BOLD}<-- avg spot{RESET}" if abs(s - avg_spot) < 0.0001 else ""
        print(f"  {fmt_p(s):>{col}}{fmt_p(u):>{col}}{fmt_p(sw):>{col}}"
              f"{fmt_p(c):>{col}}{fmt_p(k):>{col}}{tag}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

C_UNHEDGED = "#94a3b8"
C_SWAP     = "#60a5fa"
C_CAP      = "#f59e0b"
C_COLLAR   = "#34d399"
C_BG       = "#0f172a"
C_PANEL    = "#1e293b"
C_TXT      = "#f1f5f9"
C_GRID_C   = "#334155"
C_STRIKE   = "#f87171"
C_FLOOR_L  = "#a78bfa"


def _style_ax(ax, title: str):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, color=C_TXT, fontsize=10.5, fontweight="bold", pad=10)
    ax.tick_params(colors=C_TXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID_C)
    ax.xaxis.label.set_color(C_TXT)
    ax.yaxis.label.set_color(C_TXT)
    ax.grid(color=C_GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)


def _plot_payoff(ax, swap_ask: float, hedge_ratio: float,
                 cap_strike: float, cap_premium: float,
                 floor_strike: float, floor_premium: float,
                 avg_spot: float):
    """All-in blended effective price vs. spot for each strategy."""
    h, e   = hedge_ratio, 1.0 - hedge_ratio
    lo     = min(avg_spot, cap_strike, floor_strike) * 0.55
    hi     = max(avg_spot, cap_strike) * 1.50
    spots  = np.linspace(lo, hi, 600)
    net_p  = cap_premium - floor_premium

    unhedged  = spots
    swap_aip  = np.full_like(spots, aip_swap(swap_ask))   * h + spots * e
    cap_aip   = np.minimum(spots, cap_strike) + cap_premium
    cap_blend = cap_aip * h + spots * e
    col_aip   = np.maximum(floor_strike, np.minimum(spots, cap_strike)) + net_p
    col_blend = col_aip * h + spots * e

    ax.plot(spots, unhedged,  color=C_UNHEDGED, lw=1.4, linestyle="--", alpha=0.7,
            label="Unhedged")
    ax.plot(spots, swap_aip,  color=C_SWAP,     lw=2.0,
            label=f"Swap @ {swap_ask:.4f}")
    ax.plot(spots, cap_blend, color=C_CAP,      lw=2.0,
            label=f"Cap  strike={cap_strike:.2f} (+{cap_premium:.3f} prem)")
    ax.plot(spots, col_blend, color=C_COLLAR,   lw=2.0,
            label=f"Collar {floor_strike:.2f}-{cap_strike:.2f} (net {net_p:+.3f})")

    # Strike reference lines
    for price, color, lbl in [
        (cap_strike,   C_STRIKE,  f"cap {cap_strike:.2f}"),
        (floor_strike, C_FLOOR_L, f"floor {floor_strike:.2f}"),
    ]:
        ax.axvline(price, color=color, lw=0.9, linestyle=":", alpha=0.6)
        ax.text(price, lo + (hi-lo)*0.03, f" {lbl}", color=color, fontsize=7, va="bottom")

    # Mark avg spot
    ax.axvline(avg_spot, color=C_TXT, lw=0.7, linestyle=":", alpha=0.35)
    ax.text(avg_spot, lo + (hi-lo)*0.03, f" avg spot", color=C_TXT, fontsize=7, va="bottom")

    _style_ax(ax, f"All-In Blended Payoff  ({fmt_pct(h)} hedged + {fmt_pct(e)} at spot)")
    ax.set_xlabel("Spot Market Price ($/gal)")
    ax.set_ylabel("Blended All-In Effective Price ($/gal)")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:.2f}")
    ax.legend(fontsize=8, framealpha=0.25, facecolor=C_PANEL,
              edgecolor=C_GRID_C, labelcolor=C_TXT, loc="upper left")


def _plot_portfolio_totals(ax, tranches: List[dict], hedge_ratio: float):
    labels = ["Unhedged", "Swap", "Cap", "Collar"]
    keys   = ["cost_unhedged", "cost_swap", "cost_cap", "cost_collar"]
    colors = [C_UNHEDGED, C_SWAP, C_CAP, C_COLLAR]
    totals = [sum(t[k] for t in tranches) for k in keys]
    best_i = 1 + [totals[1],totals[2],totals[3]].index(min(totals[1],totals[2],totals[3]))

    bars = ax.barh(labels, totals, color=colors, height=0.48,
                   edgecolor=C_GRID_C, linewidth=0.6, alpha=0.92)
    for i, (bar, val) in enumerate(zip(bars, totals)):
        ax.text(bar.get_width()*1.015, bar.get_y()+bar.get_height()/2,
                f"${val:,.0f}", va="center", ha="left", color=C_TXT, fontsize=8)
        if i == best_i:
            ax.text(bar.get_width()*0.50, bar.get_y()+bar.get_height()/2,
                    "BEST", va="center", ha="center", color=C_BG, fontsize=7.5, fontweight="bold")

    _style_ax(ax, f"Total Portfolio Cost  ({fmt_pct(hedge_ratio)} hedge ratio)")
    ax.set_xlabel("Total All-In Fuel Cost ($)")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1_000:.0f}K")
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.22)


def _plot_premium_schedule(ax, schedule: Dict[float, tuple],
                            cap_strike: float, floor_strike: float):
    """
    Chart 3: Premium schedule visualised as grouped bars.
    Shows the trade-off between strike level and premium cost.
    """
    strikes     = sorted(schedule.keys())
    cap_prems   = [schedule[s][0] for s in strikes]
    floor_prems = [schedule[s][1] for s in strikes]
    net_prems   = [c - f for c, f in zip(cap_prems, floor_prems)]
    x = np.arange(len(strikes))
    w = 0.25

    ax.bar(x - w,   cap_prems,   w, label="Cap premium (you pay)",    color=C_CAP,    alpha=0.9, edgecolor=C_GRID_C)
    ax.bar(x,       floor_prems, w, label="Floor premium (you recv)",  color=C_FLOOR_L,alpha=0.9, edgecolor=C_GRID_C)
    ax.bar(x + w,   net_prems,   w, label="Net collar premium",        color=C_COLLAR, alpha=0.9, edgecolor=C_GRID_C)

    # Highlight selected strikes
    for i, s in enumerate(strikes):
        if s == cap_strike:
            ax.axvline(i, color=C_STRIKE,  lw=1.2, linestyle=":", alpha=0.7)
            ax.text(i, max(cap_prems)*1.05, "cap\nstrike", ha="center",
                    color=C_STRIKE, fontsize=7)
        if s == floor_strike:
            ax.axvline(i, color=C_FLOOR_L, lw=1.2, linestyle=":", alpha=0.7)
            ax.text(i + 0.12, max(cap_prems)*0.90, "floor\nstrike", ha="center",
                    color=C_FLOOR_L, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"${s:.2f}" for s in strikes], color=C_TXT, fontsize=8)
    _style_ax(ax, "Premium Schedule  (cap vs. floor vs. net collar cost)")
    ax.set_xlabel("Strike Price ($/gal)")
    ax.set_ylabel("Premium ($/gal)")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:.3f}")
    ax.legend(fontsize=8, framealpha=0.25, facecolor=C_PANEL,
              edgecolor=C_GRID_C, labelcolor=C_TXT)


def _plot_hedge_ratio_sensitivity(ax, swap_ask: float,
                                  cap_strike: float, cap_premium: float,
                                  floor_strike: float, floor_premium: float,
                                  tranches: List[dict]):
    ratios = np.linspace(0, 1, 101)
    net_p  = cap_premium - floor_premium

    def portfolio_at_ratio(strategy_aip_fn, ratio):
        h, e = ratio, 1.0 - ratio
        total = 0.0
        for t in tranches:
            aip   = strategy_aip_fn(t["spot"])
            total += (aip * h + t["spot"] * e) * t["total_vol"]
        return total

    unhedged_total = sum(t["cost_unhedged"] for t in tranches)
    swap_c   = [portfolio_at_ratio(lambda s: aip_swap(swap_ask), r) for r in ratios]
    cap_c    = [portfolio_at_ratio(lambda s: aip_cap(s, cap_strike, cap_premium), r) for r in ratios]
    collar_c = [portfolio_at_ratio(lambda s: aip_collar(s, cap_strike, cap_premium,
                                                         floor_strike, floor_premium), r)
                for r in ratios]

    ax.axhline(unhedged_total, color=C_UNHEDGED, lw=1.2, linestyle="--",
               alpha=0.7, label="Unhedged")
    ax.plot(ratios*100, swap_c,   color=C_SWAP,   lw=2.0, label="Swap")
    ax.plot(ratios*100, cap_c,    color=C_CAP,    lw=2.0, label="Cap")
    ax.plot(ratios*100, collar_c, color=C_COLLAR, lw=2.0, label="Collar")

    # Mark current ratio
    current_hr = tranches[0]["hedge_ratio"]
    for fn, color in [
        (lambda s: aip_swap(swap_ask), C_SWAP),
        (lambda s: aip_cap(s, cap_strike, cap_premium), C_CAP),
        (lambda s: aip_collar(s, cap_strike, cap_premium, floor_strike, floor_premium), C_COLLAR),
    ]:
        val = portfolio_at_ratio(fn, current_hr)
        ax.scatter([current_hr*100], [val], color=color, s=45, zorder=5)

    ax.set_xlabel("Hedge Ratio (%)")
    ax.set_ylabel("Total Portfolio Cost ($)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x/1_000:.0f}K")
    _style_ax(ax, "Cost vs. Hedge Ratio  (dots = current ratio)")
    ax.legend(fontsize=8, framealpha=0.25, facecolor=C_PANEL,
              edgecolor=C_GRID_C, labelcolor=C_TXT, loc="upper right")


def build_plot(swap_ask: float, hedge_ratio: float,
               cap_strike: float, cap_premium: float,
               floor_strike: float, floor_premium: float,
               tranches: List[dict],
               schedule: Dict[float, tuple]):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family":      "monospace",
        "figure.facecolor": C_BG,
        "text.color":       C_TXT,
        "axes.labelcolor":  C_TXT,
    })

    total_vol = sum(t["total_vol"] for t in tranches)
    avg_spot  = sum(t["spot"]*t["total_vol"] for t in tranches) / total_vol

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Fuel Hedge Cost Model - Dashboard",
                 color=C_TXT, fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.50, wspace=0.34,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    _plot_payoff(             fig.add_subplot(gs[0, 0]),
                              swap_ask, hedge_ratio,
                              cap_strike, cap_premium,
                              floor_strike, floor_premium, avg_spot)
    _plot_portfolio_totals(   fig.add_subplot(gs[0, 1]), tranches, hedge_ratio)
    _plot_premium_schedule(   fig.add_subplot(gs[1, 0]), schedule, cap_strike, floor_strike)
    _plot_hedge_ratio_sensitivity(fig.add_subplot(gs[1, 1]),
                              swap_ask, cap_strike, cap_premium,
                              floor_strike, floor_premium, tranches)

    net_p = cap_premium - floor_premium
    fig.text(0.5, 0.005,
             f"Swap {swap_ask:.4f}/gal  |  Cap strike {cap_strike:.2f} (+{cap_premium:.3f})  "
             f"|  Collar {floor_strike:.2f}-{cap_strike:.2f} (net {net_p:+.3f})  "
             f"|  Hedge {fmt_pct(hedge_ratio)}  |  {total_vol:,.0f} gal forecast",
             ha="center", va="bottom", color=C_GRID_C, fontsize=7.5)

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_float(label: str, default: float = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError
            return val
        except ValueError:
            print("    Please enter a positive number.")

def prompt_ratio(label: str, default: float = 0.75) -> float:
    while True:
        raw = input(f"  {label} [0-100%]  [{default*100:.0f}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if val > 1.0:
                val /= 100.0
            if not 0.0 < val <= 1.0:
                raise ValueError
            return val
        except ValueError:
            print("    Please enter a value between 0 and 100.")

def prompt_strike(label: str, schedule: Dict[float, tuple]) -> float:
    strikes = sorted(schedule.keys())
    options = "  /  ".join(fmt_p(s) for s in strikes)
    while True:
        raw = input(f"  {label}\n    Options: {options}\n    Enter: ").strip()
        try:
            val = float(raw)
            if val in schedule:
                return val
            # Allow near match
            closest = min(strikes, key=lambda s: abs(s - val))
            if abs(closest - val) < 0.001:
                return closest
            raise ValueError
        except ValueError:
            print(f"    Please enter one of: {', '.join(str(s) for s in strikes)}")

def prompt_int(label: str, default: int = None) -> int:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if val <= 0:
                raise ValueError
            return val
        except ValueError:
            print("    Please enter a positive integer.")


def interactive_mode(schedule: Dict[float, tuple]):
    banner()

    section("MARKET DATA")
    print(f"\n  Available strikes:")
    print(f"  {'Strike':<12}{'Cap Premium':>14}{'Floor Premium':>16}{'Net (C-F)':>12}")
    print(f"  {'-'*56}")
    for s in sorted(schedule):
        cp, fp = schedule[s]
        print(f"  {fmt_p(s):<12}{fmt_p(cp):>14}{fmt_p(fp):>16}{fmt_p(cp-fp):>12}")

    section("SWAP")
    swap_bid = prompt_float("Swap bid ($/gal)", default=SWAP_BID_DEFAULT)
    swap_ask = prompt_float("Swap ask ($/gal)", default=SWAP_ASK_DEFAULT)

    section("CONTRACT SELECTION")
    cap_strike   = prompt_strike("Cap strike  (sets your ceiling):", schedule)
    floor_strike = prompt_strike("Floor strike (sets collar's floor):", schedule)
    cap_premium, _    = schedule[cap_strike]
    _, floor_premium  = schedule[floor_strike]

    section("VOLUME & HEDGE RATIO")
    volume      = prompt_float("Total forecast volume (gal)", default=100_000)
    hedge_ratio = prompt_ratio("Hedge ratio", default=0.75)

    section("SPOT PRICE TRANCHES  (price scenarios / delivery periods)")
    n = prompt_int("Number of tranches", default=3)
    tranche_defs = []
    for i in range(1, n+1):
        print(f"\n  {BOLD}Tranche {i}{RESET}")
        spot = prompt_float("  Spot price scenario ($/gal)")
        vol  = prompt_float("  Volume for this tranche (gal)", default=volume/n)
        tranche_defs.append((spot, vol))

    return swap_ask, hedge_ratio, cap_strike, cap_premium, floor_strike, floor_premium, tranche_defs


def parse_args():
    parser = argparse.ArgumentParser(
        prog="fuel_hedge.py",
        description="Fuel hedge cost model with market premium schedule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cap/Floor premiums come from the built-in schedule (or override with --schedule).
Collar net premium = cap_premium - floor_premium.

Built-in schedule:
  Strike   Cap     Floor
  3.50     0.346   0.065
  3.60     0.276   0.097
  3.70     0.216   0.136
  3.80     0.164   0.182
  3.90     0.122   0.241
  4.00     0.090   0.307

Examples:
  python fuel_hedge.py --interactive
  python fuel_hedge.py \\
      --swap-ask 3.758 --cap-strike 3.80 --floor-strike 3.60 \\
      --volume 100000 --hedge-ratio 75 \\
      --tranches 4 \\
      --prices 3.20 3.80 4.20 4.60 \\
      --vol-splits 25 25 25 25
        """
    )
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--swap-bid",    type=float, default=SWAP_BID_DEFAULT)
    parser.add_argument("--swap-ask",    type=float, default=SWAP_ASK_DEFAULT)
    parser.add_argument("--cap-strike",  type=float, metavar="PRICE",
        help="Strike price for Cap (and Cap leg of Collar)")
    parser.add_argument("--floor-strike",type=float, metavar="PRICE",
        help="Strike price for Floor leg of Collar")
    parser.add_argument("--volume",      type=float, metavar="GAL")
    parser.add_argument("--hedge-ratio", type=float, default=0.75, metavar="RATIO",
        help="0.75 or 75 = 75%% (default: 75%%)")
    parser.add_argument("--tranches",    type=int,   metavar="N")
    parser.add_argument("--prices",      type=float, nargs="+", metavar="PRICE",
        help="Spot price per tranche")
    parser.add_argument("--vol-splits",  type=float, nargs="+", metavar="PCT",
        help="Volume split %% per tranche (must sum to 100; overrides even split)")
    parser.add_argument("--no-scenario", action="store_true")
    parser.add_argument("--no-plot",     action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    schedule = DEFAULT_SCHEDULE
    args     = parse_args()

    if args.interactive or len(sys.argv) == 1:
        (swap_ask, hedge_ratio, cap_strike, cap_premium,
         floor_strike, floor_premium, tranche_defs) = interactive_mode(schedule)
        no_plot     = False
        no_scenario = False
        swap_bid    = SWAP_BID_DEFAULT
    else:
        required = [("--cap-strike", args.cap_strike), ("--floor-strike", args.floor_strike),
                    ("--volume", args.volume), ("--tranches", args.tranches),
                    ("--prices", args.prices)]
        missing = [f for f, v in required if v is None]
        if missing:
            print(f"\n  Missing: {', '.join(missing)}  — run --interactive or see --help\n")
            sys.exit(1)

        for strike_name, strike_val in [("--cap-strike", args.cap_strike),
                                         ("--floor-strike", args.floor_strike)]:
            if strike_val not in schedule:
                avail = ", ".join(str(s) for s in sorted(schedule))
                print(f"\n  {strike_name} {strike_val} not in schedule. Options: {avail}\n")
                sys.exit(1)

        n = args.tranches
        if len(args.prices) != n:
            print(f"\n  --prices: expected {n} values, got {len(args.prices)}\n")
            sys.exit(1)

        # Resolve volumes
        if args.vol_splits:
            if len(args.vol_splits) != n:
                print(f"\n  --vol-splits: expected {n} values\n"); sys.exit(1)
            if abs(sum(args.vol_splits) - 100) > 0.01:
                print(f"\n  --vol-splits must sum to 100 (got {sum(args.vol_splits):.1f})\n"); sys.exit(1)
            vols = [args.volume * pct / 100 for pct in args.vol_splits]
        else:
            vols = [args.volume / n] * n

        hr = args.hedge_ratio
        if hr > 1.0:
            hr /= 100.0
        if not 0.0 < hr <= 1.0:
            print("\n  --hedge-ratio must be between 0 and 100\n"); sys.exit(1)

        swap_bid     = args.swap_bid
        swap_ask     = args.swap_ask
        hedge_ratio  = hr
        cap_strike   = args.cap_strike
        floor_strike = args.floor_strike
        cap_premium, _   = schedule[cap_strike]
        _, floor_premium = schedule[floor_strike]
        tranche_defs = list(zip(args.prices, vols))
        no_plot      = args.no_plot
        no_scenario  = args.no_scenario
        banner()

    # ── Compute ──────────────────────────────────────────────────────────────
    tranches = [
        cost_for_tranche(vol, spot, hedge_ratio, swap_ask,
                         cap_strike, cap_premium, floor_strike, floor_premium)
        for spot, vol in tranche_defs
    ]

    total_vol = sum(t["total_vol"] for t in tranches)
    avg_spot  = sum(t["spot"]*t["total_vol"] for t in tranches) / total_vol

    # ── Terminal output ───────────────────────────────────────────────────────
    print_schedule(schedule, cap_strike, floor_strike, swap_bid, swap_ask)
    print_contract_summary(swap_ask, cap_strike, cap_premium,
                           floor_strike, floor_premium, hedge_ratio, total_vol)
    print_tranche_table(tranches)
    print_summary(tranches, swap_ask, hedge_ratio,
                  cap_strike, cap_premium, floor_strike, floor_premium)
    if not no_scenario:
        print_scenario_table(swap_ask, hedge_ratio, cap_strike, cap_premium,
                             floor_strike, floor_premium, avg_spot)

    # ── Dashboard ────────────────────────────────────────────────────────────
    if not no_plot:
        print(f"  {CYAN}Opening dashboard...{RESET}\n")
        build_plot(swap_ask, hedge_ratio, cap_strike, cap_premium,
                   floor_strike, floor_premium, tranches, schedule)


if __name__ == "__main__":
    main()