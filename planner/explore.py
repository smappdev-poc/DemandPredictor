"""Exploratory analysis of the loaded data.

Profiles every shipment series (trend, seasonality, outliers, intermittency,
variability, ABC class), checks data quality, and turns the findings into
suggested changes to the engine settings and master data.

Works on the inputs directly, so it is available before a forecast is run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import customer_groups, idx_to_month, month_to_idx

LEVELS = {
    "base": ("Base SKU", ["base_sku"]),
    "base_group": ("Base SKU x customer group", ["base_sku", "customer_group"]),
    "base_group_loc": ("Base SKU x customer group x location (forecast level)", ["base_sku", "customer_group", "location"]),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Syntetos-Boylan cut-offs for demand patterns
ADI_CUT, CV2_CUT = 1.32, 0.49
OUTLIER_Z = 3.5          # robust z-score beyond which a month is an outlier
OUTLIER_MIN_DEV = 0.25   # ...and it must also be at least 25% away from its local median
TREND_CUT = 10.0  # % per year


# ------------------------------------------------------------------ series
def _roots(items: pd.DataFrame, hard: pd.DataFrame) -> dict[str, str]:
    succ = {str(r.from_base): str(r.to_base) for r in hard.itertuples(index=False)
            if pd.notna(r.from_base) and pd.notna(r.to_base)}
    out = {}
    for b in items["base_sku"].dropna().astype(str).unique():
        cur, seen = b, {b}
        while cur in succ and succ[cur] not in seen:
            cur = succ[cur]
            seen.add(cur)
        out[b] = cur
    return out


def build_series(items, customers, shipments, hard, level="base_group_loc", *, nonkey_by_channel=False,
                 drop_last_month=False, combine_chains=True):
    """Monthly shipment series at the chosen level.

    Returns (wide frame: one row per series, one column per month index; list of 'YYYY-MM' months).
    Shipments that do not match master data are left out (see quality checks).
    """
    keys = LEVELS[level][1]
    if shipments.empty or items.empty:
        return pd.DataFrame(), []
    v2b = dict(zip(items["version_sku"].astype(str), items["base_sku"].fillna("").astype(str)))
    grp = customer_groups(customers, nonkey_by_channel).to_dict() if len(customers) else {}
    sh = shipments.copy()
    sh["base_sku"] = sh["version_sku"].astype(str).map(v2b)
    sh["customer_group"] = sh["account"].astype(str).map(grp)
    sh = sh[sh["base_sku"].fillna("").ne("") & sh["customer_group"].notna()]
    if sh.empty:
        return pd.DataFrame(), []
    if combine_chains:
        sh["base_sku"] = sh["base_sku"].map(_roots(items, hard)).fillna(sh["base_sku"])
    sh["t"] = sh["month"].map(month_to_idx)
    t0, t1 = int(sh["t"].min()), int(sh["t"].max())
    if drop_last_month and t1 > t0:
        sh = sh[sh["t"] < t1]
        t1 -= 1
    wide = (sh.groupby(keys + ["t"])["qty"].sum().unstack("t")
              .reindex(columns=range(t0, t1 + 1)).fillna(0.0))
    return wide, [idx_to_month(t) for t in range(t0, t1 + 1)]


# ------------------------------------------------------------------ one series
def _seasonality(y: np.ndarray, cal0: int) -> tuple[np.ndarray, float]:
    """Seasonal indices and strength (0-1) from ratios to the centred 2x12 moving average.

    Ratios are clipped to a robust band first, so one spike can neither bend an index nor
    pass for seasonality. Strength = share of the ratios' variation the indices explain.
    """
    n = len(y)
    if n < 24:
        return np.ones(12), np.nan
    ma = np.convolve(y, np.r_[0.5, np.ones(11), 0.5] / 12.0, mode="valid")
    t = np.arange(6, n - 6)
    ok = ma > 0
    r, cal = y[t][ok] / ma[ok], (cal0 + t[ok]) % 12
    if len(r) < 12:
        return np.ones(12), np.nan
    med = np.median(r)
    sd = 1.4826 * np.median(np.abs(r - med)) or r.std() or 1.0
    r = np.clip(r, med - 3 * sd, med + 3 * sd)
    idx = np.array([r[cal == m].mean() if (cal == m).any() else np.nan for m in range(12)])
    idx = np.where(np.isnan(idx), np.nanmean(idx), idx)
    idx = np.maximum(idx / idx.mean(), 0.05)
    v = r.var()
    strength = float(max(0.0, 1 - (r / idx[cal]).var() / v)) if v > 0 else 0.0
    # With two or three years, noise alone explains some variation by chance: discount for it.
    years = len(r) / 12
    chance = 11 / max(len(r) - 1, 1)
    if years < 6:
        # too few ratios for any of the variation to be told apart from chance
        strength = float(max(0.0, (strength - chance) / (1 - chance))) if chance < 1 else 0.0
    return idx, strength


def analyse(y_full: np.ndarray, t0: int) -> dict:
    """Statistics for one series. y_full covers every month in the data; t0 = month index of y_full[0]."""
    y_full = np.asarray(y_full, dtype=float)
    neg = int((y_full < 0).sum())
    nz = np.flatnonzero(y_full > 0)
    if len(nz) == 0:
        return dict(months_active=0, volume=0.0, pattern="No shipments", trend="No shipments", negatives=neg)
    start = nz[0]
    y = np.maximum(y_full[start:], 0.0)
    n, cal0 = len(y), (t0 + start) % 12
    cal = (cal0 + np.arange(n)) % 12
    mean = y.mean()
    res = dict(first_month=idx_to_month(t0 + start), last_month=idx_to_month(t0 + nz[-1]),
               months_since_last=len(y_full) - 1 - nz[-1], months_active=n, volume=float(y.sum()),
               last12=float(y[-12:].sum()), prior12=float(y[-24:-12].sum()) if n >= 24 else np.nan,
               zero_pct=float((y == 0).mean() * 100), negatives=neg)
    res["yoy_pct"] = (res["last12"] / res["prior12"] - 1) * 100 if n >= 24 and res["prior12"] > 0 else np.nan

    # Intermittency: average demand interval and squared CV of non-zero demand sizes
    nonzero = y[y > 0]
    res["adi"] = n / len(nonzero)
    res["cv2"] = float((nonzero.std() / nonzero.mean()) ** 2) if len(nonzero) > 1 else 0.0
    if n < 6:
        res["pattern"] = "Too short"
    else:
        smooth_t, smooth_s = res["adi"] < ADI_CUT, res["cv2"] < CV2_CUT
        res["pattern"] = ("Smooth" if smooth_t and smooth_s else "Erratic" if smooth_t
                          else "Intermittent" if smooth_s else "Lumpy")

    # Seasonality: indices by calendar month and the share of variation they explain
    idx, res["seasonal_strength"] = _seasonality(y, cal0)
    res["peak_month"] = MONTHS[int(np.argmax(idx))] if n >= 24 else ""
    res["peak_ratio"] = float(idx.max()) if n >= 24 else np.nan
    res["indices"] = idx
    if not res["seasonal_strength"] >= 0.3:
        idx = np.ones(12)  # weak or no seasonality: do not deseasonalise noise

    # Trend: slope of a straight line through the deseasonalised last 36 months, % of the mean per year
    d = y / idx[cal]
    w = d[-36:]
    if n >= 12 and w.mean() > 0:
        slope = np.polyfit(np.arange(len(w)), w, 1)[0]
        res["trend_pct"] = float(slope * 12 / w.mean() * 100)
        res["trend"] = "Up" if res["trend_pct"] > TREND_CUT else "Down" if res["trend_pct"] < -TREND_CUT else "Flat"
    else:
        res["trend_pct"], res["trend"] = np.nan, "Too short"

    # Variability after seasonality (coefficient of variation of the deseasonalised series)
    dv = d[-24:]
    res["cov"] = float(dv.std() / dv.mean()) if dv.mean() > 0 else np.nan

    # Level shift: last 3 months against the 12 before them, both deseasonalised
    res["recent_shift_pct"] = (float(d[-3:].mean() / d[-15:-3].mean() - 1) * 100
                               if n >= 15 and d[-15:-3].mean() > 0 else np.nan)

    # Outliers: robust z-score of the deseasonalised series against a centred rolling median
    res["outliers"], res["outlier_months"], res["outlier_flags"] = 0, "", np.zeros(n, dtype=bool)
    if n >= 12 and res["pattern"] in ("Smooth", "Erratic"):
        # On the log scale shipment noise is close to symmetric, so a ratio spike and a dip score alike.
        ld = np.log1p(d)
        # Median of the 3 months either side, leaving the month itself out so it cannot hide its own spike.
        med = np.array([np.median(np.r_[ld[max(0, i - 3):i], ld[i + 1:i + 4]]) for i in range(n)])
        r = ld - med
        mad = np.median(np.abs(r - np.median(r)))
        scale = mad / 0.6745 if mad > 0 else np.mean(np.abs(r)) * 1.2533
        if scale > 0:
            z = r / scale
            flags = (np.abs(z) > OUTLIER_Z) & (np.abs(np.expm1(r)) > OUTLIER_MIN_DEV)
            res["outlier_flags"] = flags
            res["outliers"] = int(flags.sum())
            res["outlier_months"] = ", ".join(idx_to_month(t0 + start + i) for i in np.flatnonzero(flags))
            res["recent_outliers"] = int(flags[-12:].sum())
    res.setdefault("recent_outliers", 0)
    res["_start"] = start
    return res


# ------------------------------------------------------------------ all series
def profile(wide: pd.DataFrame, months: list[str], items: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row of statistics per series, plus ABC (volume, last 12 months) and XYZ (variability) classes."""
    if wide.empty:
        return pd.DataFrame()
    t0 = month_to_idx(months[0])
    rows = []
    for key, y in zip(wide.index, wide.values):
        key = key if isinstance(key, tuple) else (key,)
        a = analyse(y, t0)
        a = {k: v for k, v in a.items() if k not in ("indices", "outlier_flags", "_start")}
        rows.append((*key, a))
    names = list(wide.index.names)
    df = pd.DataFrame([dict(zip(names, r[:-1]), **r[-1]) for r in rows])
    if items is not None and len(items):
        fam = items.groupby("base_sku")["family"].first()
        df.insert(len(names), "family", df["base_sku"].map(fam).fillna(""))
    # ABC by last-12-month volume
    order = df["last12"].fillna(0).sort_values(ascending=False)
    tot = order.sum()
    cum = order.cumsum() / tot if tot > 0 else order * 0
    df["abc"] = pd.Series(np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C")), index=order.index)
    first_b = cum[cum > 0.8].index[:1]  # the series that crosses 80% is still an A
    df.loc[first_b, "abc"] = "A" if len(cum[cum <= 0.8]) == 0 else df.loc[first_b, "abc"]
    df["xyz"] = np.select([df["cov"] < 0.5, df["cov"] < 1.0, df["cov"] >= 1.0], ["X", "Y", "Z"], "")
    df["label"] = df[names].astype(str).agg(" · ".join, axis=1)
    return df


# ------------------------------------------------------------------ data quality
def quality_checks(D: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Plain-language checks on the loaded inputs. severity: High, Medium or Info."""
    items, cust, locs, sh = D["items"], D["customers"], D["locations"], D["shipments"]
    out = []

    def add(sev, check, count, detail, examples=()):
        if count:
            ex = list(examples)[:5]
            out.append(dict(severity=sev, check=check, count=int(count), detail=detail,
                            examples=", ".join(map(str, ex)) + ("..." if count > 5 else "")))

    if sh.empty:
        return pd.DataFrame(columns=["severity", "check", "count", "detail", "examples"])
    vset, aset, lset = set(items["version_sku"].astype(str)), set(cust["account"].astype(str)), set(locs["location"].astype(str))
    for col, known, name in (("version_sku", vset, "item"), ("account", aset, "customer"), ("location", lset, "location")):
        bad = sh[~sh[col].astype(str).isin(known)]
        add("High", f"Shipments for a {name} not in the {name} master", bad[col].nunique(),
            f"{len(bad):,} monthly rows, {bad['qty'].sum():,.0f} units left out of history. Add them to the master and reload.",
            bad.groupby(col)["qty"].sum().sort_values(ascending=False).index)
    nob = items[items["base_sku"].fillna("").astype(str).str.strip() == ""]
    add("High", "Items with no base SKU", len(nob), "Their shipments are not forecast.", nob["version_sku"])

    neg = sh[sh["qty"] < 0]
    add("Medium", "Negative monthly quantities", len(neg),
        "Returns larger than shipments in a month. They are counted as zero by the forecast.",
        (neg["version_sku"] + " " + neg["month"]).tolist())

    tot = sh.groupby("month")["qty"].sum().sort_index()
    allm = [idx_to_month(t) for t in range(month_to_idx(tot.index[0]), month_to_idx(tot.index[-1]) + 1)]
    gaps = [m for m in allm if m not in tot.index]
    add("High", "Months with no shipments at all", len(gaps), "Usually a missing extract. Reload those months.", gaps)
    if len(tot) >= 4 and not cfg.get("drop_last_month"):
        last, prev3 = tot.iloc[-1], tot.iloc[-4:-1].mean()
        ly = tot.get(idx_to_month(month_to_idx(tot.index[-1]) - 12))
        low_vs_prev = prev3 > 0 and last < 0.6 * prev3
        low_vs_ly = ly is None or (ly > 0 and last < 0.6 * ly)
        if low_vs_prev and low_vs_ly:
            add("High", "Latest month looks incomplete", 1,
                f"{tot.index[-1]} has {last:,.0f} units against {prev3:,.0f} a month before it"
                + (f" and {ly:,.0f} a year earlier" if ly else "") + ". Turn on Ignore the latest month, or reload it when complete.",
                [tot.index[-1]])
    n_hist = len(allm)
    if n_hist < 24:
        add("Medium", "Less than 24 months of history", 1,
            f"{n_hist} months loaded. Series cannot use their own seasonality; they borrow the family profile at best.", [])

    last_t = month_to_idx(tot.index[-1])
    recent = sh[sh["month"].map(month_to_idx) > last_t - 3]
    nf = items[~items["forecastable"].astype(bool)]
    shipped_nf = sorted(set(nf["version_sku"]) & set(recent["version_sku"]))
    add("Medium", "Not-forecastable versions still shipping (last 3 months)", len(shipped_nf),
        "Check the item master: they get no forecast even though they still ship.", shipped_nf)
    yr = set(sh.loc[sh["month"].map(month_to_idx) > last_t - 12, "version_sku"])
    future_launch = set(items.loc[items["launch_month"].map(lambda m: (month_to_idx(m) or -1) > last_t), "version_sku"])
    ruled = set(D["version_transitions"]["to_version"]) if len(D["version_transitions"]) else set()
    succ_bases = set(D["hard_transitions"]["to_base"]) if len(D["hard_transitions"]) else set()
    npi_bases = set(D["new_products"]["base_sku"]) if len(D["new_products"]) else set()
    f = items[items["forecastable"].astype(bool)]
    idle = sorted(v for v, b in zip(f["version_sku"], f["base_sku"])
                  if v not in yr and v not in future_launch and v not in ruled and b not in succ_bases | npi_bases)
    add("Medium", "Forecastable versions with no shipments in 12 months", len(idle),
        "They can take a share of the forecast through equal or manual splits. Untick Forecastable or add a launch rule.", idle)
    fc_bases = set(f["base_sku"])
    pred = set(D["hard_transitions"]["from_base"]) if len(D["hard_transitions"]) else set()
    shipping_bases = set(items.loc[items["version_sku"].isin(yr), "base_sku"])
    orphan = sorted((shipping_bases - fc_bases) - pred)
    add("High", "Shipping base SKUs with no forecastable version", len(orphan),
        "Their forecast will sit in Unallocated. Mark a version forecastable or add a hard transition.", orphan)
    idle_c = sorted(set(cust["account"]) - set(sh.loc[sh["month"].map(month_to_idx) > last_t - 12, "account"]))
    add("Info", "Customers with no shipments in 12 months", len(idle_c), "No effect on the forecast; consider tidying the master.", idle_c)

    for t, cols, known in (("version_transitions", ["from_version", "to_version"], vset),
                           ("hard_transitions", ["from_base", "to_base"], set(items["base_sku"])),
                           ("new_products", ["base_sku", "like_base"], set(items["base_sku"]))):
        r = D[t]
        if len(r):
            miss = sorted({str(x) for c in cols for x in r[c]} - known)
            add("High", f"{t.replace('_', ' ').capitalize()} naming unknown codes", len(miss),
                "These rules are ignored until the codes exist in the item master.", miss)
    sev = {"High": 0, "Medium": 1, "Info": 2}
    df = pd.DataFrame(out, columns=["severity", "check", "count", "detail", "examples"])
    return df.sort_values("severity", key=lambda s: s.map(sev), kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ recommendations
def recommendations(prof: pd.DataFrame, checks: pd.DataFrame, cfg: dict, key_customers: list[str]) -> pd.DataFrame:
    """Suggested setting and data changes, from the forecast-level profile."""
    rec = []

    def add(area, current, suggested, why):
        rec.append(dict(area=area, current=current, suggested=suggested, why=why))

    on_off = lambda b: "On" if b else "Off"  # noqa: E731
    if len(checks) and (checks["check"] == "Latest month looks incomplete").any():
        add("Setting: Ignore the latest month", "Off", "On",
            checks.loc[checks["check"] == "Latest month looks incomplete", "detail"].iloc[0])
    p = prof[prof["months_active"] > 0] if len(prof) else prof
    if len(p) == 0:
        return pd.DataFrame(rec, columns=["area", "current", "suggested", "why"])
    vol = p["last12"].fillna(0)
    tv = vol.sum() or 1.0

    seasonal_share = vol[p["seasonal_strength"].fillna(0) >= 0.4].sum() / tv
    if cfg.get("seasonal") and seasonal_share < 0.10 and p["months_active"].median() >= 24:
        add("Setting: Use seasonal models", "On", "Try Off",
            f"Only {seasonal_share:.0%} of volume shows clear seasonality. Compare backtest error with seasonal models off.")
    elif not cfg.get("seasonal") and seasonal_share >= 0.30:
        add("Setting: Use seasonal models", "Off", "On",
            f"{seasonal_share:.0%} of volume is in clearly seasonal series.")

    med_n = p["months_active"].median()
    if med_n < 30 and int(cfg.get("holdout", 6)) > 3:
        add("Setting: Backtest holdout", f"{cfg.get('holdout')} months", "3 months",
            f"Half the series have under {med_n:.0f} months of history; a 6-month holdout leaves little to fit on.")
    short_share = vol[p["months_active"] < int(cfg.get("min_seasonal_months", 24))].sum() / tv
    if short_share >= 0.25:
        add("Data: product families", "", "Check family codes",
            f"{short_share:.0%} of volume is in series with under 24 months, which borrow their family's seasonality. "
            "Families that mix seasonal and non-seasonal products will mislead them.")

    rec_out = p[p["recent_outliers"] > 0]
    rec_share = vol[p["recent_outliers"] > 0].sum() / tv
    if rec_share >= 0.20 and int(cfg.get("recent_window", 6)) < 12:
        add("Setting: Recent-mix window", f"{cfg.get('recent_window')} months", "12 months",
            f"{rec_share:.0%} of volume has outliers in the last 12 months, which can swing the version split.")
    if len(rec_out):
        top = rec_out.sort_values("last12", ascending=False).head(5)["label"].tolist()
        add("Data: outliers", f"{len(rec_out)} series", "Review and correct at source",
            "Recent outliers feed straight into the forecast; there is no in-app cleansing yet. Largest: " + "; ".join(top))

    sparse = p["pattern"].isin(["Intermittent", "Lumpy"])
    sparse_share = vol[sparse].sum() / tv
    if sparse_share >= 0.15:
        add("Forecast level", f"{sparse.mean():.0%} of series sparse", "Consider fewer splits",
            f"{sparse_share:.0%} of volume sits in intermittent or lumpy series at the forecast level. "
            "Fewer key customers or no per-channel pooling gives fuller series.")
    if cfg.get("nonkey_by_channel") and "customer_group" in p.columns:
        pooled = p[p["customer_group"].astype(str).str.startswith("All other - ")]
        if len(pooled) and pooled["pattern"].isin(["Intermittent", "Lumpy"]).mean() > 0.5:
            add("Setting: Pool non-key customers by channel", "On", "Off",
                "Most per-channel pools are intermittent or lumpy; one pool gives a steadier series.")
    if "customer_group" in p.columns and key_customers:
        kg = p[p["customer_group"].isin(key_customers)]
        by = kg.groupby("customer_group").apply(
            lambda g: g.loc[g["pattern"].isin(["Intermittent", "Lumpy"]), "last12"].sum() / max(g["last12"].sum(), 1),
            include_groups=False)
        weak = by[by > 0.5].index.tolist()
        if weak:
            add("Data: key customer flags", ", ".join(weak[:5]), "Consider non-key",
                "Most of their volume is intermittent or lumpy at the forecast level, so they forecast better pooled.")

    dead = p[(p["months_since_last"] >= 6) & (p["months_active"] >= 6)]
    if len(dead):
        add("Data: inactive series", f"{len(dead)} series", "Check lifecycle",
            "No shipments for 6+ months but history remains, so the model may still forecast a level. "
            "Mark versions not forecastable or add a hard transition. Examples: " + "; ".join(dead["label"].head(5)))
    falling = p[(p["trend"] == "Down") & (p["trend_pct"] < -40)]
    if len(falling):
        add("Data: steep decline", f"{len(falling)} series", "Check for a transition",
            "Falling over 40% a year; often a product being replaced. Examples: " + "; ".join(falling["label"].head(5)))
    return pd.DataFrame(rec, columns=["area", "current", "suggested", "why"])


def pattern_summary(prof: pd.DataFrame) -> pd.DataFrame:
    """Series count and last-12-month volume share by demand pattern."""
    p = prof[prof["months_active"] > 0]
    g = p.groupby("pattern").agg(series=("label", "size"), volume=("last12", "sum"))
    g["volume_share"] = g["volume"] / max(g["volume"].sum(), 1)
    return g.reindex([x for x in ["Smooth", "Erratic", "Intermittent", "Lumpy", "Too short"] if x in g.index]).reset_index()


def abc_xyz(prof: pd.DataFrame) -> pd.DataFrame:
    p = prof[prof["xyz"] != ""]
    m = p.pivot_table(index="abc", columns="xyz", values="last12", aggfunc="size", fill_value=0)
    return m.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
