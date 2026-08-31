#!/usr/bin/env python3
"""
ship_mtm.py - Mark-to-market valuation for a vessel, from one CSV of comparable sales.

Expects a file along these lines. Header matching is fuzzy, so punctuation,
capitalisation and spacing do not matter:

    Sale Date | Vessel Name | Sale Price ($US millions) | Year Built |
    Age at Sale (Years) | DWT '(000) | Index

"Index" is your 12-month composite of the Baltic Dry and Capesize indices. Any
column name containing index / composite / baltic / bdi / bci will be found.

Fits two models on ln(price):

    MODEL 1  simple     one driver (default: the index)
    MODEL 2  multiple   age + age^2 + ln(DWT) + ln(index) [+ sale_year]

Then prices your vessel off both, with a prediction interval.

Usage
-----
    python ship_mtm.py sales.csv --built 2012 --dwt 180000
    python ship_mtm.py sales.csv --age 14.5 --dwt 180 --index-12m 1450
    python ship_mtm.py sales.csv --built 2004 --dwt 176000 --scrap-usd-m 11.5
    python ship_mtm.py sales.csv --built 2012 --dwt 180000 --drop-sale-year

    python ship_mtm.py --demo          # synthetic data, to see the output shape

Requires: pandas, numpy, statsmodels, scipy
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

RNG = np.random.default_rng(20260831)


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

# Ordered. Each field claims a column before the next field gets a look, so
# "Sale Price" cannot be stolen by the sale_date matcher and vice versa.
ALIASES: list[tuple[str, list[str]]] = [
    ("sale_date",  ["sale date", "date of sale", "sold date", "date sold",
                    "transaction date", "date"]),
    ("price",      ["sale price us millions", "sale price usd millions",
                    "sale price millions", "sale price", "price us millions",
                    "price usd m", "price usd", "price musd", "price m",
                    "price", "value"]),
    ("build_year", ["year built", "build year", "year of build", "built year",
                    "yob", "built"]),
    ("age",        ["age at sale years", "age at sale", "age years", "age yrs",
                    "vessel age", "age"]),
    ("dwt",        ["dwt 000", "dwt000", "dwt k", "deadweight tonnage",
                    "deadweight", "dwt mt", "dwt", "tonnage", "size"]),
    ("index",      ["index 12m", "12m index", "12 month index",
                    "12 month composite", "12m composite", "composite index",
                    "bdi bci composite", "baltic composite", "composite",
                    "baltic index", "baltic", "bdi", "bci", "index"]),
    ("vessel",     ["vessel name", "ship name", "vessel", "ship", "name"]),
]

REQUIRED = ("sale_date", "price", "dwt", "index")


def norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def resolve_columns(df: pd.DataFrame, overrides: dict) -> dict:
    """Map logical field names onto the actual columns in the file."""
    normalized = {c: norm(c) for c in df.columns}
    mapping: dict[str, str] = {}
    claimed: set[str] = set()

    for field, col in overrides.items():
        if col is None:
            continue
        if col not in df.columns:
            raise ValueError(f"--col-{field.replace('_', '-')} '{col}' not in file. "
                             f"Columns are: {list(df.columns)}")
        mapping[field] = col
        claimed.add(col)

    for field, aliases in ALIASES:
        if field in mapping:
            continue
        best, best_score = None, 0
        for col, n in normalized.items():
            if col in claimed:
                continue
            for rank, alias in enumerate(aliases):
                if n == alias:
                    score = 1000 - rank
                elif alias in n or n in alias:
                    score = 500 - rank + len(alias)
                else:
                    continue
                if score > best_score:
                    best, best_score = col, score
                break
        if best is not None:
            mapping[field] = best
            claimed.add(best)

    # Last resort: if the index column was named something unguessable but is the
    # only numeric column left over, take it.
    if "index" not in mapping:
        leftover = [c for c in df.columns
                    if c not in claimed and pd.api.types.is_numeric_dtype(df[c])]
        if len(leftover) == 1:
            mapping["index"] = leftover[0]
            print(f"[note] using leftover numeric column '{leftover[0]}' as the index",
                  file=sys.stderr)

    missing = [f for f in REQUIRED if f not in mapping]
    if missing:
        raise ValueError(
            f"could not find column(s) for: {missing}\n"
            f"  file has: {list(df.columns)}\n"
            f"  pass them explicitly, e.g. --col-price 'Sale Price ($US millions)'")
    if "build_year" not in mapping and "age" not in mapping:
        raise ValueError("need either a year-built column or an age-at-sale column")
    return mapping


def to_number(s: pd.Series) -> pd.Series:
    """Strip $ , % and stray spaces, then coerce to float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[^0-9eE.\-+]", "", regex=True)
    return pd.to_numeric(cleaned.replace("", np.nan), errors="coerce")


# ---------------------------------------------------------------------------
# Load and prepare
# ---------------------------------------------------------------------------

def read_table(path: str, encoding: str | None = None,
               quiet: bool = False) -> pd.DataFrame:
    """Read a CSV that Excel may have saved in a non-UTF-8 encoding.

    Excel on macOS and Windows writes cp1252 or Mac Roman by default, so smart
    quotes and accented characters blow up a plain UTF-8 read. latin-1 is last
    because it never fails, which makes it a guaranteed fallback.
    """
    encs = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp1252",
                                        "mac_roman", "latin-1"]
    last_err = None
    for enc in encs:
        try:
            df = pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.ParserError as e:
            last_err = e
            try:  # ragged rows or a non-comma delimiter
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            except Exception:
                continue

        if df.shape[1] == 1:  # delimiter probably is not a comma
            try:
                alt = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                if alt.shape[1] > 1:
                    df = alt
            except Exception:
                pass

        if enc != "utf-8" and not quiet:
            print(f"[note] file is not UTF-8; read it as {enc}")
        if enc == "latin-1" and not quiet:
            print("[note] latin-1 is a last-resort fallback. Check the column "
                  "mapping below looks right.")

        # Excel exports trail empty columns and rows.
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        junk = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if junk:
            df = df.drop(columns=junk)
        return df.reset_index(drop=True)

    raise ValueError(f"could not read {path}: {last_err}\n"
                     f"  Try re-saving from Excel as 'CSV UTF-8', or pass "
                     f"--encoding cp1252")


def load(path: str, overrides: dict, dwt_units: str, quiet: bool,
         encoding: str | None = None) -> pd.DataFrame:
    raw = read_table(path, encoding, quiet)
    raw.columns = [str(c).strip() for c in raw.columns]
    m = resolve_columns(raw, overrides)

    if not quiet:
        print("Column mapping")
        for field in ("sale_date", "vessel", "price", "build_year", "age", "dwt", "index"):
            if field in m:
                print(f"  {field:<11} <- {m[field]}")
        print()

    d = pd.DataFrame()
    d["sale_date"] = pd.to_datetime(raw[m["sale_date"]], errors="coerce")
    d["price_usd_m"] = to_number(raw[m["price"]])
    d["dwt"] = to_number(raw[m["dwt"]])
    d["index_12m"] = to_number(raw[m["index"]])
    if "vessel" in m:
        d["vessel"] = raw[m["vessel"]].astype(str)
    if "build_year" in m:
        d["build_year"] = to_number(raw[m["build_year"]])
    if "age" in m:
        d["age_given"] = to_number(raw[m["age"]])

    # DWT in thousands vs absolute.
    scale_hint = "000" in norm(m["dwt"]) or norm(m["dwt"]).endswith(" k")
    med = d["dwt"].median()
    if dwt_units == "thousands" or (dwt_units == "auto" and (scale_hint or med < 2000)):
        d["dwt"] = d["dwt"] * 1000.0
        if not quiet:
            print(f"[note] DWT read as thousands (median {med:,.0f} -> "
                  f"{d['dwt'].median():,.0f} MT)\n")

    n0 = len(d)
    d = d.dropna(subset=["sale_date", "price_usd_m", "dwt", "index_12m"])
    if len(d) < n0:
        print(f"[warn] dropped {n0 - len(d)} row(s) with missing core fields",
              file=sys.stderr)
    return d.reset_index(drop=True)


def prepare(d: pd.DataFrame, scrap_usd_m: float) -> pd.DataFrame:
    d = d.copy()
    d["sale_year"] = d["sale_date"].dt.year + (d["sale_date"].dt.month - 1) / 12.0

    if "age_given" in d.columns and d["age_given"].notna().any():
        d["age"] = d["age_given"]
        if "build_year" in d.columns:
            gap = (d["sale_year"] - d["build_year"] - d["age_given"]).abs()
            bad = int((gap > 1.5).sum())
            if bad:
                print(f"[warn] {bad} row(s) where age-at-sale and "
                      f"(sale year - year built) differ by over 18 months. "
                      f"Using the stated age.", file=sys.stderr)
    else:
        d["age"] = d["sale_year"] - d["build_year"]

    d["age"] = d["age"].clip(lower=0.0)
    d["age2"] = d["age"] ** 2
    d["ln_dwt"] = np.log(d["dwt"])
    d["ln_index"] = np.log(d["index_12m"])

    excess = d["price_usd_m"] - scrap_usd_m
    if scrap_usd_m > 0 and (excess <= 0).any():
        n = int((excess <= 0).sum())
        print(f"[warn] {n} comp(s) at or below the ${scrap_usd_m}m scrap floor, dropped",
              file=sys.stderr)
        d = d.loc[excess > 0].copy()
        excess = d["price_usd_m"] - scrap_usd_m
    d["ln_price"] = np.log(excess if scrap_usd_m > 0 else d["price_usd_m"])

    cols = ["ln_price", "age", "ln_dwt", "ln_index", "sale_year"]
    bad = ~np.isfinite(d[cols].to_numpy()).all(axis=1)
    if bad.any():
        print(f"[warn] dropped {int(bad.sum())} row(s) with non-finite features",
              file=sys.stderr)
    return d.loc[~bad].reset_index(drop=True)


# ---------------------------------------------------------------------------
# As-of date and index lookup
# ---------------------------------------------------------------------------

def parse_asof(text: str) -> pd.Timestamp:
    """Accept 2024-03, 03/2024, March 2024, Mar-2024, 2024-03-15, etc."""
    t = str(text).strip()
    try:
        return pd.Timestamp(t)
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m", "%m/%Y", "%m-%Y", "%b %Y", "%B %Y", "%b-%Y", "%Y%m"):
        try:
            return pd.Timestamp(pd.to_datetime(t, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Period(t, freq="M").to_timestamp()
    except Exception:
        raise ValueError(
            f"could not read '{text}' as a date. Try 2024-03, March 2024, or 2024-03-15.")


def index_at(d: pd.DataFrame, vdate: pd.Timestamp) -> tuple[float, str]:
    """Pull the 12m composite that applies to the as-of month, from the comps."""
    month = vdate.to_period("M")
    same = d.loc[d["sale_date"].dt.to_period("M") == month, "index_12m"]
    if len(same):
        n = len(same)
        return float(same.mean()), (f"{'mean of ' + str(n) + ' comps' if n > 1 else '1 comp'} "
                                    f"in {vdate:%b %Y}")
    gap = (d["sale_date"] - vdate).abs()
    i = gap.idxmin()
    off = gap.loc[i].days / 30.44
    return float(d.loc[i, "index_12m"]), (f"nearest comp, {d.loc[i, 'sale_date']:%b %Y}, "
                                          f"{off:.0f} months from the as-of date")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class Fit:
    def __init__(self, name, terms, d):
        self.name, self.terms = name, terms
        X = sm.add_constant(d[terms], has_constant="add")
        self.res = sm.OLS(d["ln_price"], X).fit()
        self.robust = self.res.get_robustcov_results(cov_type="HC3")
        self.smearing = float(np.mean(np.exp(self.res.resid)))  # Duan
        self.n = int(self.res.nobs)


def vif_table(d, terms):
    if len(terms) < 2:
        return pd.DataFrame(columns=["term", "vif"])
    X = sm.add_constant(d[terms], has_constant="add").to_numpy(float)
    return pd.DataFrame([(t, variance_inflation_factor(X, i + 1))
                         for i, t in enumerate(terms)], columns=["term", "vif"])


def kfold(d, terms, k=5, scrap=0.0):
    n = len(d)
    if n < k * 3:
        return float("nan"), float("nan")
    folds = np.array_split(RNG.permutation(n), k)
    errs, pcts = [], []
    for f in folds:
        test, train = d.iloc[f], d.drop(d.index[f])
        Xtr = sm.add_constant(train[terms], has_constant="add")
        m = sm.OLS(train["ln_price"], Xtr).fit()
        smear = float(np.mean(np.exp(m.resid)))
        Xte = sm.add_constant(test[terms], has_constant="add")[Xtr.columns]
        pred = np.exp(m.predict(Xte)).to_numpy() * smear + scrap
        act = test["price_usd_m"].to_numpy()
        errs.append(pred - act)
        pcts.append(np.abs(pred - act) / act)
    e, p = np.concatenate(errs), np.concatenate(pcts)
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(p) * 100)


def value(fit, subject, scrap=0.0, alpha=0.05):
    row = pd.DataFrame([{t: subject[t] for t in fit.terms}])
    X = sm.add_constant(row, has_constant="add")[fit.res.model.exog_names]
    pr = fit.res.get_prediction(X).summary_frame(alpha=alpha)
    ln_hat = float(pr["mean"].iloc[0])
    return {
        "model": fit.name,
        "median": np.exp(ln_hat) + scrap,
        "mean": np.exp(ln_hat) * fit.smearing + scrap,
        "lo": np.exp(float(pr["obs_ci_lower"].iloc[0])) + scrap,
        "hi": np.exp(float(pr["obs_ci_upper"].iloc[0])) + scrap,
    }


# ---------------------------------------------------------------------------
# Nearest comparables
# ---------------------------------------------------------------------------

def nearest_comps(d: pd.DataFrame, subject: dict, n: int = 8) -> pd.DataFrame:
    """The n sales most like the subject, on age, size and market together.

    Standardised distance, so a year of age and a percent of size are weighted
    by how much each actually varies across the comp set.
    """
    def z(col, target):
        sd = d[col].std(ddof=1)
        return (d[col] - target) / (sd if sd > 1e-9 else 1.0)

    dist = np.sqrt(z("age", subject["age"]) ** 2
                   + z("ln_dwt", subject["ln_dwt"]) ** 2
                   + z("ln_index", subject["ln_index"]) ** 2)
    return d.assign(distance=dist).nsmallest(min(n, len(d)), "distance")


def comps_block(near: pd.DataFrame, fit: "Fit", scrap: float) -> str:
    """Broker-style table of the closest sales, with the model's read on each."""
    fitted = np.exp(fit.res.fittedvalues) * fit.smearing + scrap
    has_name = "vessel" in near.columns
    rows = [f"  {'vessel':<20}{'sold':>9}{'age':>6}{'DWT':>9}{'index':>8}"
            f"{'paid':>8}{'model':>8}{'diff':>7}"]
    for i, r in near.iterrows():
        nm = str(r["vessel"])[:19] if has_name else "-"
        f_ = float(fitted.loc[i])
        diff = (r["price_usd_m"] / f_ - 1) * 100
        rows.append(
            f"  {nm:<20}{r['sale_date']:%b %y}".ljust(31)
            + f"{r['age']:>6.1f}{r['dwt'] / 1000:>8,.0f}k{r['index_12m']:>8,.0f}"
              f"{r['price_usd_m']:>8.1f}{f_:>8.1f}{diff:>6.0f}%")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def make_charts(d, fits, subject, basis, scrap, alpha, path, near=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    simple, multi = fits[0], fits[-1]
    v = value(multi, subject, scrap, alpha)
    fitted = np.exp(multi.res.fittedvalues) * multi.smearing + scrap
    resid_pct = (d["price_usd_m"] / fitted - 1) * 100

    GREY, BLUE, RED, GREEN = "#9aa5b1", "#2a6fb0", "#c1392b", "#2e8b57"

    def curve(varying, lo, hi, npts=120):
        """Model 2 price across one variable, everything else held at subject."""
        xs = np.linspace(lo, hi, npts)
        rows = []
        for x in xs:
            r = dict(subject)
            r[varying] = x
            if varying == "age":
                r["age2"] = x ** 2
            rows.append(r)
        X = sm.add_constant(pd.DataFrame(rows)[multi.terms],
                            has_constant="add")[multi.res.model.exog_names]
        pr = multi.res.get_prediction(X).summary_frame(alpha=alpha)
        return (xs,
                np.exp(pr["mean"].to_numpy()) * multi.smearing + scrap,
                np.exp(pr["obs_ci_lower"].to_numpy()) + scrap,
                np.exp(pr["obs_ci_upper"].to_numpy()) + scrap)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(f"Mark to market: {np.exp(subject['ln_dwt']):,.0f} DWT, "
                 f"age {subject['age']:.1f} yrs, as of {basis['date']:%b %Y}, "
                 f"index {basis['index']:,.0f}   |   "
                 f"mark ${v['median']:.1f}m  (${v['lo']:.1f}m - ${v['hi']:.1f}m)",
                 fontsize=13, fontweight="bold")

    hi_mask = np.zeros(len(d), bool)
    if near is not None:
        hi_mask = d.index.isin(near.index)

    def scatter(ax, x, xlabel, subj_x, varname, lo, hi, xscale=1.0):
        ax.scatter(x[~hi_mask], d["price_usd_m"][~hi_mask], s=26, c=GREY,
                   alpha=.55, edgecolors="none", label="comparable sales", zorder=2)
        if hi_mask.any():
            ax.scatter(x[hi_mask], d["price_usd_m"][hi_mask], s=52, c=GREEN,
                       alpha=.9, edgecolors="white", linewidths=.6,
                       label="closest comps", zorder=3)
        gx, gy, glo, ghi = curve(varname, lo, hi)
        px = (np.exp(gx) if varname.startswith("ln_") else gx) / xscale
        ax.fill_between(px, glo, ghi, color=BLUE, alpha=.10, zorder=1,
                        label=f"{int((1 - alpha) * 100)}% range")
        ax.plot(px, gy, color=BLUE, lw=2, zorder=4, label="model, your ship's specs")
        ax.errorbar(subj_x, v["median"], yerr=[[v["median"] - v["lo"]],
                                               [v["hi"] - v["median"]]],
                    fmt="*", ms=20, color=RED, ecolor=RED, elinewidth=1.6,
                    capsize=5, zorder=5, label="your ship")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sale price ($m)")
        ax.grid(alpha=.25, lw=.6)

    ax = axes[0, 0]
    scatter(ax, d["age"], "age at sale (years)", subject["age"], "age",
            float(d["age"].min()), float(d["age"].max()))
    ax.set_title("Price vs age\n(curve holds size and market at your ship's)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=.9)

    ax = axes[0, 1]
    scatter(ax, d["dwt"] / 1000, "DWT (000)", np.exp(subject["ln_dwt"]) / 1000,
            "ln_dwt", float(d["ln_dwt"].min()), float(d["ln_dwt"].max()),
            xscale=1000.0)
    ax.set_xlabel("DWT (000)")
    ax.set_title("Price vs size\n(curve holds age and market at your ship's)", fontsize=10)

    ax = axes[0, 2]
    scatter(ax, d["index_12m"], "12-month composite index", basis["index"],
            "ln_index", float(d["ln_index"].min()), float(d["ln_index"].max()))
    ax.set_title("Price vs freight market\n(curve holds age and size at your ship's)",
                 fontsize=10)

    # Model quality
    ax = axes[1, 0]
    ax.scatter(fitted, d["price_usd_m"], s=26, c=GREY, alpha=.6, edgecolors="none")
    if hi_mask.any():
        ax.scatter(fitted[hi_mask], d["price_usd_m"][hi_mask], s=52, c=GREEN,
                   alpha=.9, edgecolors="white", linewidths=.6)
    lim = [0, max(fitted.max(), d["price_usd_m"].max()) * 1.05]
    ax.plot(lim, lim, "--", color=RED, lw=1.3, label="perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    _, mape = kfold(d, multi.terms, scrap=scrap)
    ax.set_title(f"Model vs reality\nR2 {multi.res.rsquared:.2f}, "
                 f"typical miss {mape:.0f}%", fontsize=10)
    ax.set_xlabel("model says ($m)"); ax.set_ylabel("actually paid ($m)")
    ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.6)

    # Where the model misses, over time
    ax = axes[1, 1]
    sc = ax.scatter(d["sale_date"], resid_pct, s=28, c=d["age"], cmap="viridis",
                    alpha=.85, edgecolors="none")
    ax.axhline(0, color=RED, lw=1.2)
    ax.axhline(30, color=GREY, lw=.8, ls=":")
    ax.axhline(-30, color=GREY, lw=.8, ls=":")
    ax.axvline(basis["date"], color=BLUE, lw=1.4, ls="--", label="your as-of date")
    ax.set_title("Where the model misses, over time\n"
                 "(above 0 = sold for more than the model says)", fontsize=10)
    ax.set_ylabel("actual vs model (%)")
    plt.colorbar(sc, ax=ax, label="age at sale")
    ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.6)
    ax.tick_params(axis="x", rotation=30)

    # Sensitivity of the mark to the market
    ax = axes[1, 2]
    lo_i, hi_i = float(d["index_12m"].min()), float(d["index_12m"].max())
    gx, gy, glo, ghi = curve("ln_index", np.log(lo_i), np.log(hi_i))
    px = np.exp(gx)
    ax.fill_between(px, glo, ghi, color=BLUE, alpha=.12)
    ax.plot(px, gy, color=BLUE, lw=2)
    ax.axvline(basis["index"], color=RED, ls="--", lw=1.3)
    ax.plot([basis["index"]], [v["median"]], "*", ms=20, color=RED)
    ax.annotate(f"${v['median']:.1f}m\nat {basis['index']:,.0f}",
                xy=(basis["index"], v["median"]),
                xytext=(12, -34), textcoords="offset points", fontsize=9,
                color=RED, fontweight="bold")
    ax.set_title("What your ship is worth at other market levels\n"
                 "(age and size fixed)", fontsize=10)
    ax.set_xlabel("12-month composite index"); ax.set_ylabel("value ($m)")
    ax.grid(alpha=.25, lw=.6)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def coef_block(fit):
    names = fit.res.model.exog_names
    b = fit.res.params.to_numpy()
    se = np.sqrt(np.diag(fit.robust.cov_params()))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=int(fit.res.df_resid)))
    out = [f"  {'term':<12}{'coef':>10}{'HC3 SE':>10}{'t':>8}{'p':>9}"]
    for i, nm in enumerate(names):
        star = " *" if p[i] < 0.05 else ""
        out.append(f"  {nm:<12}{b[i]:>10.4f}{se[i]:>10.4f}{t[i]:>8.2f}{p[i]:>9.3f}{star}")
    return "\n".join(out)


def interpret(fit):
    p, out = fit.res.params, []
    if "age" in p:
        tail = " at age 0" if "age2" in p else ""
        out.append(f"each year of age: {(np.exp(p['age']) - 1) * 100:+.1f}%{tail}")
    if "age2" in p and p["age2"] != 0:
        turn = -p["age"] / (2 * p["age2"])
        if 0 < turn < 45:
            out.append(f"depreciation flattens around age {turn:.0f}")
    if "ln_dwt" in p:
        out.append(f"+10% DWT: {p['ln_dwt'] * 10:+.1f}% value "
                   f"(elasticity {p['ln_dwt']:.2f})")
    if "ln_index" in p:
        out.append(f"+10% on the index: {p['ln_index'] * 10:+.1f}% value")
    if "sale_year" in p:
        out.append(f"residual time trend {(np.exp(p['sale_year']) - 1) * 100:+.1f}%/yr "
                   f"beyond the index")
    return out


def extrapolation_warnings(d, basis, subject) -> list[str]:
    """Flag when the mark is being asked for outside what the comps can support."""
    out = []
    if basis:
        lo, hi = d["index_12m"].min(), d["index_12m"].max()
        idx = basis["index"]
        if idx > hi or idx < lo:
            over = idx > hi
            ref = hi if over else lo
            pct = abs(idx / ref - 1) * 100
            out.append(
                f"Index {idx:,.0f} is outside your comps ({lo:,.0f} to {hi:,.0f}), "
                f"{pct:.0f}% {'above the highest' if over else 'below the lowest'} "
                f"market any of these ships actually sold in. The model is guessing "
                f"beyond its evidence, and the interval below is too narrow as a "
                f"result.")

        newest, oldest = d["sale_date"].max(), d["sale_date"].min()
        vdate = basis["date"]
        ahead = (vdate - newest).days / 30.44
        if ahead > 6:
            out.append(
                f"As-of date is {ahead:.0f} months past your newest comp "
                f"({newest:%b %Y}). Nothing in the file covers that period. Add "
                f"--drop-sale-year so the model stops projecting a time trend it "
                f"cannot see.")
        elif vdate < oldest:
            out.append(f"As-of date is before your oldest comp ({oldest:%b %Y}).")

    age = subject.get("age")
    if age is not None and (age > d["age"].max() or age < d["age"].min()):
        out.append(
            f"Subject age {age:.1f} yrs is outside your comps "
            f"({d['age'].min():.1f} to {d['age'].max():.1f}). Depreciation curves "
            f"bend at the ends, so the age term is unreliable here.")

    dwt = np.exp(subject["ln_dwt"])
    if dwt > d["dwt"].max() or dwt < d["dwt"].min():
        out.append(
            f"Subject size {dwt:,.0f} DWT is outside your comps "
            f"({d['dwt'].min():,.0f} to {d['dwt'].max():,.0f}).")
    return out


def report(d, fits, subject, desc, scrap, alpha, basis=None, near=None):
    W = 78
    bar = "=" * W
    print(bar)
    print("MARK-TO-MARKET VALUATION".center(W))
    print(bar)
    print(f"Subject      : {desc}")
    if basis:
        print(f"As of        : {basis['date']:%B %Y}  ({basis['date_src']})")
        print(f"Index used   : {basis['index']:,.0f}  ({basis['index_src']})")
    print(f"Comparables  : {len(d)} sales, {d['sale_date'].min():%b %Y} to "
          f"{d['sale_date'].max():%b %Y}")
    print(f"Price        : ${d['price_usd_m'].min():.1f}m to "
          f"${d['price_usd_m'].max():.1f}m (median ${d['price_usd_m'].median():.1f}m)")
    print(f"Age          : {d['age'].min():.1f} to {d['age'].max():.1f} yrs")
    print(f"DWT          : {d['dwt'].min():,.0f} to {d['dwt'].max():,.0f}")
    print(f"Index        : {d['index_12m'].min():,.0f} to {d['index_12m'].max():,.0f}")
    if scrap > 0:
        print(f"Scrap floor  : ${scrap:.2f}m (fitted on price less scrap)")

    for n in extrapolation_warnings(d, basis, subject):
        print()
        print(textwrap.fill(f"[!] {n}", width=W, subsequent_indent="    "))

    for fit in fits:
        print("\n" + "-" * W)
        print(f"{fit.name}   ln(price{' - scrap' if scrap else ''}) ~ "
              f"{' + '.join(fit.terms)}")
        print("-" * W)
        print(coef_block(fit))
        rmse, mape = kfold(d, fit.terms, scrap=scrap)
        sigma = float(np.std(fit.res.resid, ddof=1))
        print(f"\n  R2 {fit.res.rsquared:.3f}   adj R2 {fit.res.rsquared_adj:.3f}   "
              f"AIC {fit.res.aic:.1f}   n {fit.n}   df {int(fit.res.df_resid)}")
        print(f"  typical error {(np.exp(sigma) - 1) * 100:.1f}%   "
              f"5-fold CV RMSE ${rmse:.2f}m   MAPE {mape:.1f}%")
        bp = het_breuschpagan(fit.res.resid, fit.res.model.exog)
        print(f"  Breusch-Pagan p {bp[3]:.3f}   "
              f"Jarque-Bera p {jarque_bera(fit.res.resid)[1]:.3f}   "
              f"Durbin-Watson {durbin_watson(fit.res.resid):.2f}")
        v = vif_table(d, fit.terms)
        if len(v):
            print("  VIF: " + ", ".join(f"{r.term} {r.vif:.1f}" for r in v.itertuples()))
            real = v[~v["term"].isin(["age", "age2"])]  # polynomial VIF is structural
            if len(real) and real["vif"].max() > 10:
                w = real.loc[real["vif"].idxmax()]
                print(f"  [!] {w['term']} VIF {w['vif']:.1f}: not separately identified "
                      f"from the terms it moves with.")
                print("      If sale_year and ln_index are fighting, use --drop-sale-year.")
        for s in interpret(fit):
            print(f"  - {s}")

    if near is not None and len(near):
        print("\n" + "-" * W)
        print(f"CLOSEST {len(near)} COMPARABLES   (nearest on age, size and market together)")
        print("-" * W)
        print(comps_block(near, fits[-1], scrap))
        print(f"\n  'diff' is what the buyer paid versus what the model says that "
              f"ship was worth.")

    print("\n" + bar)
    print("VALUATION".center(W))
    print(bar)
    hdr = f"{int((1 - alpha) * 100)}% prediction interval"
    print(f"  {'model':<22}{'point ($m)':>12}{'mean ($m)':>12}{hdr:>30}")
    vals = [value(f, subject, scrap, alpha) for f in fits]
    for v in vals:
        span = f"${v['lo']:.1f}m - ${v['hi']:.1f}m"
        print(f"  {v['model']:<22}{v['median']:>12.2f}{v['mean']:>12.2f}{span:>30}")

    vs, vm = vals[0], vals[-1]
    gap = abs(vm["median"] - vs["median"]) / vm["median"] * 100
    half = (vm["hi"] - vm["lo"]) / 2 / vm["median"] * 100
    verdict = ("normal for S&P hedonics, and is why brokers quote a range"
               if half <= 40 else
               "too wide to defend as a mark: add comps or tighten the set to your "
               "vessel's size class")
    print()
    print(textwrap.fill(
        f"Carry ${vm['median']:.2f}m from the multiple regression as the mark. The "
        f"simple model disagrees by {gap:.0f}%, which is the value that age and size "
        f"explain and the freight market alone does not. Half the interval width is "
        f"{half:.0f}% of value, which is {verdict}.",
        width=W, initial_indent="  ", subsequent_indent="  "))
    print(bar)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def write_demo(path="demo_sales.csv"):
    months = pd.date_range("2013-01-01", "2026-08-01", freq="MS")
    t = np.arange(len(months))
    bdi = np.clip(700 + 520 * np.sin(2 * np.pi * t / 46)
                  + 260 * np.sin(2 * np.pi * t / 17)
                  + RNG.normal(0, 90, len(months)), 300, None)
    comp = pd.Series(0.5 * bdi + 0.5 * bdi * 1.28, index=months).rolling(12).mean()

    n = 160
    pick = RNG.choice(np.arange(12, len(months)), n)
    sd = months[pick]
    idx = comp.iloc[pick].to_numpy()
    sy = sd.year + (sd.month - 1) / 12
    by = np.floor(sy - RNG.uniform(0.5, 24.0, n)).astype(int)  # never sold pre-build
    age = sy - by
    dwt_k = RNG.choice([82, 95, 122, 176, 180, 205, 210], n)
    ln_p = (-10.28 - 0.055 * age + 0.0009 * age ** 2
            + 0.95 * np.log(dwt_k * 1000) + 0.42 * np.log(idx)
            + RNG.normal(0, 0.17, n))
    pd.DataFrame({
        "Sale Date": sd,
        "Vessel Name": [f"MV Demo {i:03d}" for i in range(n)],
        "Sale Price ($US millions)": np.exp(ln_p).round(2),
        "Year Built": by,
        "Age at Sale (Years)": age.round(1),
        "DWT '(000)": dwt_k,
        "Index": idx.round(0),
    }).sort_values("Sale Date").to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Mark a vessel to market off comparable sales.")
    ap.add_argument("csv", nargs="?", help="CSV of comparable sales")
    ap.add_argument("--demo", action="store_true", help="generate and run on demo data")
    ap.add_argument("--built", type=float, help="subject year built")
    ap.add_argument("--age", type=float, help="subject age in years (overrides --built)")
    ap.add_argument("--dwt", type=float, help="subject DWT (thousands or absolute)")
    ap.add_argument("--index-12m", "--index", type=float, dest="index_12m",
                    metavar="VALUE",
                    help="12-month composite to value against. Default: the index "
                         "your comps show for the as-of month.")
    ap.add_argument("--as-of", "--valuation-date", dest="valuation_date",
                    metavar="DATE",
                    help="year and month to value as of, e.g. 2024-03 or 'March 2024'. "
                         "Default: the latest sale in your file.")
    ap.add_argument("--simple-driver", default="ln_index",
                    choices=["ln_index", "age", "ln_dwt", "sale_year"])
    ap.add_argument("--drop-sale-year", action="store_true",
                    help="omit sale_year from the multiple model")
    ap.add_argument("--no-age2", action="store_true", help="drop the quadratic age term")
    ap.add_argument("--scrap-usd-m", type=float, default=0.0, help="scrap floor in $m")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dwt-units", default="auto",
                    choices=["auto", "thousands", "absolute"])
    ap.add_argument("--save-fitted", help="write comps + fitted values to this CSV")
    ap.add_argument("--quiet", action="store_true", help="suppress the column mapping")
    ap.add_argument("--encoding", help="force a file encoding, e.g. cp1252 or latin-1")
    ap.add_argument("--chart", nargs="?", const="ship_mtm_charts.png", metavar="PATH",
                    help="write a 6-panel chart showing your ship among the comps")
    ap.add_argument("--neighbors", type=int, default=8, metavar="N",
                    help="how many closest comparables to list (0 to skip)")
    for f in ("sale-date", "price", "build-year", "age", "dwt", "index", "vessel"):
        ap.add_argument(f"--col-{f}", help=f"force the column used for {f}")
    a = ap.parse_args(argv)

    path = a.csv
    if a.demo:
        path = write_demo()
        print(f"[demo] wrote {path}\n")
        a.built = a.built if a.built is not None else 2012
        a.dwt = a.dwt if a.dwt is not None else 180000
    if not path:
        ap.error("give a CSV path, or use --demo")
    if a.dwt is None or (a.built is None and a.age is None):
        ap.error("need --dwt and one of --built / --age")

    overrides = {"sale_date": a.col_sale_date, "price": a.col_price,
                 "build_year": a.col_build_year, "age": a.col_age, "dwt": a.col_dwt,
                 "index": a.col_index, "vessel": a.col_vessel}
    d = prepare(load(path, overrides, a.dwt_units, a.quiet, a.encoding), a.scrap_usd_m)

    terms = ["age", "ln_dwt", "ln_index"]
    if not a.no_age2:
        terms.insert(1, "age2")
    if not a.drop_sale_year:
        terms.append("sale_year")
    k = len(terms) + 1
    if len(d) < 10 * k:
        print(f"[warn] {len(d)} comps for {k} parameters. Under ~{10 * k} the multiple "
              f"model overfits and the interval reads optimistic.\n", file=sys.stderr)

    fits = [Fit("MODEL 1  simple", [a.simple_driver], d),
            Fit("MODEL 2  multiple", terms, d)]

    if a.valuation_date:
        vdate = parse_asof(a.valuation_date)
        date_src = "you set it"
    else:
        vdate = d["sale_date"].max()
        date_src = "latest sale in the file, no --as-of given"

    lo, hi = d["sale_date"].min(), d["sale_date"].max()
    if vdate > hi + pd.Timedelta(days=120) or vdate < lo - pd.Timedelta(days=120):
        print(f"[warn] as-of {vdate:%b %Y} sits outside your comps "
              f"({lo:%b %Y} to {hi:%b %Y}). The model is extrapolating in time.\n",
              file=sys.stderr)

    sale_year = vdate.year + (vdate.month - 1) / 12.0
    age = a.age if a.age is not None else max(0.0, sale_year - a.built)
    dwt = a.dwt * 1000.0 if a.dwt < 2000 else a.dwt

    if a.index_12m is not None:
        idx, idx_src = float(a.index_12m), "you set it"
    else:
        idx, idx_src = index_at(d, vdate)

    subject = {"age": age, "age2": age ** 2, "ln_dwt": np.log(dwt),
               "ln_index": np.log(idx), "sale_year": sale_year}
    desc = f"{dwt:,.0f} DWT, age {age:.1f} yrs at the as-of date"
    basis = {"date": vdate, "date_src": date_src, "index": idx, "index_src": idx_src}
    near = nearest_comps(d, subject, a.neighbors) if a.neighbors > 0 else None
    report(d, fits, subject, desc, a.scrap_usd_m, a.alpha, basis, near)

    if a.chart:
        try:
            out = make_charts(d, fits, subject, basis, a.scrap_usd_m, a.alpha,
                              a.chart, near)
            print(f"\n[ok] chart written to {out}")
        except ImportError:
            print("\n[warn] charts need matplotlib: pip install matplotlib",
                  file=sys.stderr)

    if a.save_fitted:
        out = d.copy()
        for f in fits:
            tag = "simple" if "simple" in f.name else "multiple"
            out[f"fitted_{tag}"] = np.exp(f.res.fittedvalues) * f.smearing + a.scrap_usd_m
        out["resid_pct"] = (out["fitted_multiple"] / out["price_usd_m"] - 1) * 100
        out.to_csv(a.save_fitted, index=False)
        print(f"\n[ok] wrote {a.save_fitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())