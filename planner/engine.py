"""Forecast engine.

Flow:
  1. Shipments are mapped to base SKU, customer group and ship-from location.
  2. Hard transitions chain a predecessor base SKU to its successor, so the
     successor is forecast on the combined history.
  3. Each (base SKU chain, customer group, location) series gets the model that
     wins a holdout backtest.
  4. The forecast is handed to the base SKU that is live in each month, then
     split to the versions eligible in that month for that customer group.

Everything works on month indexes (year * 12 + month - 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALL = "ALL"
DEFAULTS = {
    "horizon": 18,            # months to forecast
    "seasonal": True,         # allow seasonal models
    "holdout": 6,             # months held out for the backtest
    "recent_window": 6,       # months of shipments used for the recent-mix split
    "min_seasonal_months": 24,  # history needed before a series may use its own seasonality
    "npi_own_history_months": 6,  # own history after which a new item stops borrowing
    "nonkey_by_channel": False,   # one "All other" bucket per channel instead of one overall
    "drop_last_month": False,     # ignore the latest month if it is incomplete
}


# ---------------------------------------------------------------- months
def month_to_idx(m) -> int | None:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return None
    s = str(m).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def idx_to_month(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


# ---------------------------------------------------------------- models
_ALPHAS = np.round(np.arange(0.1, 0.95, 0.1), 2)
_PHI = 0.9


def seasonal_indices(y: np.ndarray, cal0: int) -> np.ndarray:
    """Ratio-to-centred-moving-average indices by calendar month (mean 1)."""
    n = len(y)
    if n < 13:
        return np.ones(12)
    kernel = np.r_[0.5, np.ones(11), 0.5] / 12.0
    ma = np.convolve(y, kernel, mode="valid")  # centred on t = 6 .. n-7
    buckets: list[list[float]] = [[] for _ in range(12)]
    for k, s in enumerate(ma):
        if s > 0:
            t = k + 6
            buckets[(cal0 + t) % 12].append(y[t] / s)
    idx = np.array([np.mean(b) if b else np.nan for b in buckets])
    if np.isnan(idx).all():
        return np.ones(12)
    idx = np.where(np.isnan(idx), np.nanmean(idx), idx)
    mean = idx.mean()
    if mean <= 0:
        return np.ones(12)
    return np.maximum(idx / mean, 0.05)


def _ses(y, cal0, h, **_):
    best_sse, best_l = None, float(y[0])
    for a in _ALPHAS:
        level, sse = float(y[0]), 0.0
        for v in y[1:]:
            sse += (v - level) ** 2
            level = a * v + (1 - a) * level
        if best_sse is None or sse < best_sse:
            best_sse, best_l = sse, level
    return np.full(h, max(0.0, best_l))


def _snaive(y, cal0, h, **_):
    n = len(y)
    return np.array([y[n - 12 + (k % 12)] for k in range(h)], dtype=float)


def _damped(y, cal0, h, idx):
    n = len(y)
    d = y / idx[(cal0 + np.arange(n)) % 12]
    best = None
    for a in (0.2, 0.4, 0.6):
        for b in (0.05, 0.2):
            level, trend, sse = float(d[0]), 0.0, 0.0
            for v in d[1:]:
                p = level + _PHI * trend
                sse += (v - p) ** 2
                new = a * v + (1 - a) * p
                trend = b * (new - level) + (1 - b) * _PHI * trend
                level = new
            if best is None or sse < best[0]:
                best = (sse, level, trend)
    _, level, trend = best
    damp = np.cumsum(_PHI ** np.arange(1, h + 1))
    return np.maximum(0.0, (level + damp * trend) * idx[(cal0 + n + np.arange(h)) % 12])


def _seasonal_damped(y, cal0, h, **_):
    return _damped(y, cal0, h, seasonal_indices(y, cal0))


def _family_profile(y, cal0, h, fam_idx=None, **_):
    return _damped(y, cal0, h, fam_idx)


def _croston(y, cal0, h, **_):
    """Croston with the Syntetos-Boylan correction, for sparse series."""
    a = 0.1
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(h)
    z, p, last = float(y[nz[0]]), float(nz[0] + 1), nz[0]
    for t in nz[1:]:
        z = a * y[t] + (1 - a) * z
        p = a * (t - last) + (1 - a) * p
        last = t
    return np.full(h, (1 - a / 2) * z / max(p, 1.0))


M_SES = "Exponential smoothing"
M_SNAIVE = "Seasonal naive"
M_DAMPED = "Seasonal with damped trend"
M_FAMILY = "Family seasonal profile"
M_CROSTON = "Intermittent (Croston)"
M_NONE = "No history"


def select_model(y: np.ndarray, cal0: int, h: int, *, seasonal=True, fam_idx=None,
                 holdout=6, min_seasonal=24) -> dict:
    """Fit candidate models, pick the best on a holdout, refit on all history.

    y starts at the first month with shipments; cal0 is that month's calendar
    month (0 = January).
    """
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    n = len(y)
    if n == 0 or y.sum() == 0:
        return dict(f=np.zeros(h), model=M_NONE, abs_err=0.0, err=0.0, act=0.0, n=n)

    cands = {M_SES: _ses}
    if n >= 6 and (y == 0).mean() >= 0.4:
        cands[M_CROSTON] = _croston
    if seasonal:
        if n >= min_seasonal:
            cands[M_SNAIVE] = _snaive
            cands[M_DAMPED] = _seasonal_damped
        elif fam_idx is not None:
            cands[M_FAMILY] = _family_profile

    ho = holdout if n - holdout >= 12 else (3 if n >= 8 else 0)
    if ho == 0:
        # Too short to backtest: a family profile is the better prior when there is one.
        name = M_FAMILY if M_FAMILY in cands else M_SES
        return dict(f=cands[name](y, cal0, h, fam_idx=fam_idx), model=name,
                    abs_err=0.0, err=0.0, act=0.0, n=n)

    train, test = y[:-ho], y[-ho:]
    best = None
    for name, fn in cands.items():
        p = fn(train, cal0, ho, fam_idx=fam_idx)
        ae = float(np.abs(p - test).sum())
        if best is None or ae < best[1]:
            best = (name, ae, float((p - test).sum()))
    name, ae, er = best
    return dict(f=cands[name](y, cal0, h, fam_idx=fam_idx), model=name,
                abs_err=ae, err=er, act=float(test.sum()), n=n)


# ---------------------------------------------------------------- result
@dataclass
class Result:
    hist_months: list[str] = field(default_factory=list)
    fc_months: list[str] = field(default_factory=list)
    hist: pd.DataFrame = None        # version_sku, base_sku, customer_group, location, month, qty
    base_fc: pd.DataFrame = None     # base_sku, customer_group, location, month, qty
    version_fc: pd.DataFrame = None  # version_sku, base_sku, customer_group, location, month, qty
    unallocated: pd.DataFrame = None  # base_sku, customer_group, location, month, qty
    series: pd.DataFrame = None      # root_base, customer_group, location, model, abs_err, err, act, n_months
    chains: pd.DataFrame = None      # base_sku, root_base, source ("own" | "like item <x>")
    exceptions: pd.DataFrame = None  # type, key, detail, rows, qty


def customer_groups(customers: pd.DataFrame, by_channel: bool = False) -> pd.Series:
    """Planning group per account: key customers stand alone, the rest are pooled."""
    c = customers
    name = c["name"].fillna("").astype(str).str.strip()
    name = name.where(name != "", c["account"].astype(str))
    grp = c.get("customer_group", pd.Series("", index=c.index)).fillna("").astype(str).str.strip()
    key_group = grp.where(grp != "", name)
    if by_channel:
        ch = c["channel"].fillna("").astype(str).str.strip().replace("", "No channel")
        other = "All other - " + ch
    else:
        other = pd.Series("All other", index=c.index)
    out = key_group.where(c["key_customer"].astype(bool), other)
    return pd.Series(out.values, index=c["account"].astype(str).values)


def _long(rows: list[tuple], key_cols: list[str], months: list[str]) -> pd.DataFrame:
    """rows: (*keys, array over months) -> long frame without zero rows."""
    if not rows:
        return pd.DataFrame(columns=key_cols + ["month", "qty"])
    nm = len(months)
    keys = pd.DataFrame([r[:-1] for r in rows], columns=key_cols)
    out = keys.loc[keys.index.repeat(nm)].reset_index(drop=True)
    out["month"] = np.tile(np.array(months), len(rows))
    out["qty"] = np.concatenate([np.asarray(r[-1], dtype=float) for r in rows])
    return out[out["qty"] != 0].reset_index(drop=True)


def run_forecast(items, customers, locations, shipments, version_transitions,
                 hard_transitions, new_products, split_settings, manual_split,
                 settings: dict | None = None, progress=None) -> Result:
    cfg = {**DEFAULTS, **(settings or {})}
    F = int(cfg["horizon"])
    exc: list[dict] = []

    def flag(kind, key, detail, rows=0, qty=0.0):
        exc.append(dict(type=kind, key=str(key), detail=detail, rows=int(rows), qty=float(qty)))

    items = items.copy()
    items["version_sku"] = items["version_sku"].astype(str)
    items["base_sku"] = items["base_sku"].fillna("").astype(str).str.strip()
    for v in items.loc[items["base_sku"] == "", "version_sku"]:
        flag("Item without base SKU", v, "Held out of the forecast until a base SKU is assigned")
    items = items[items["base_sku"] != ""]
    v2b = dict(zip(items["version_sku"], items["base_sku"]))
    base_family = items.groupby("base_sku")["family"].first().fillna("").to_dict()
    acct2grp = customer_groups(customers, bool(cfg["nonkey_by_channel"])).to_dict()
    loc_set = set(locations["location"].astype(str))

    # ---- 1. map shipments, quarantine what does not match master data
    sh = shipments.copy()
    if len(sh) == 0:
        raise ValueError("There are no shipments loaded.")
    for c in ("version_sku", "account", "location"):
        sh[c] = sh[c].astype(str)
    sh["t"] = sh["month"].map(month_to_idx)
    checks = [("version_sku", sh["version_sku"].isin(v2b.keys()), "Shipments for an item not in the item master"),
              ("account", sh["account"].isin(acct2grp.keys()), "Shipments for a customer not in the customer master"),
              ("location", sh["location"].isin(loc_set), "Shipments from a location not in the location master")]
    ok = pd.Series(True, index=sh.index)
    for col, mask, label in checks:
        bad = sh[~mask].groupby(col)["qty"].agg(["size", "sum"])
        for key, r in bad.iterrows():
            flag(label, key, "Excluded from history", r["size"], r["sum"])
        ok &= mask
    sh = sh[ok]
    if len(sh) == 0:
        raise ValueError("No shipments match the item, customer and location masters.")
    sh["base_sku"] = sh["version_sku"].map(v2b)
    sh["customer_group"] = sh["account"].map(acct2grp)

    t0, t1 = int(sh["t"].min()), int(sh["t"].max())
    if cfg["drop_last_month"] and t1 > t0:
        sh = sh[sh["t"] < t1]
        t1 -= 1
    H = t1 - t0 + 1
    hist_months = [idx_to_month(t) for t in range(t0, t1 + 1)]
    fc_idx = np.arange(t1 + 1, t1 + 1 + F)
    fc_months = [idx_to_month(t) for t in fc_idx]

    hv = (sh.groupby(["version_sku", "base_sku", "customer_group", "location", "t"])["qty"].sum()
            .unstack("t").reindex(columns=range(t0, t1 + 1)).fillna(0.0))
    hb = hv.groupby(level=["base_sku", "customer_group", "location"]).sum()

    # ---- 2. hard-transition chains: successor, cumulative scale, root
    succ: dict[str, tuple[str, int, float]] = {}
    all_bases = set(items["base_sku"])
    for r in hard_transitions.itertuples(index=False):
        eff = month_to_idx(r.effective_month)
        fb, tb = str(r.from_base).strip(), str(r.to_base).strip()
        if eff is None or fb not in all_bases or tb not in all_bases or fb == tb:
            flag("Hard transition ignored", f"{fb} -> {tb}", "Unknown base SKU or missing effective month")
            continue
        if fb in succ:
            flag("Hard transition ignored", f"{fb} -> {tb}", "A base SKU can have one successor; one-to-many is not supported")
            continue
        scale = float(r.scale) if pd.notna(r.scale) and float(r.scale) > 0 else 1.0
        succ[fb] = (tb, eff, scale)

    def root_of(b):
        c, seen = 1.0, {b}
        while b in succ:
            nb, _, s = succ[b]
            if nb in seen:  # cycle
                return None, 1.0
            c *= s
            b = nb
            seen.add(b)
        return b, c

    root, cum = {}, {}
    for b in all_bases:
        rb, c = root_of(b)
        if rb is None:
            flag("Hard transition ignored", b, "Transition rules form a loop")
            succ.pop(b, None)
            rb, c = b, 1.0
        root[b], cum[b] = rb, c

    # ---- new products that still borrow from a like item
    base_tot = hb.groupby(level="base_sku").sum()
    own_months = {}
    for b, row in zip(base_tot.index, base_tot.values):
        ix = np.flatnonzero(row > 0)
        own_months[b] = H - int(ix[0]) if len(ix) else 0
    npi: dict[str, dict] = {}
    for r in new_products.itertuples(index=False):
        b, like = str(r.base_sku).strip(), str(r.like_base).strip()
        launch = month_to_idx(r.launch_month)
        if b not in all_bases or like not in all_bases or launch is None or b == like:
            flag("New product rule ignored", b, "Unknown base SKU, like item or launch month")
            continue
        if own_months.get(b, 0) >= int(cfg["npi_own_history_months"]):
            continue  # enough own history: forecast it like any other item
        ramp = []
        for x in str(r.ramp if pd.notna(r.ramp) else "").replace(";", ",").split(","):
            try:
                ramp.append(float(x))
            except ValueError:
                pass
        scale = float(r.scale) if pd.notna(r.scale) else 1.0
        npi[b] = dict(like=like, launch=launch, scale=scale, ramp=ramp)

    # ---- 3. planning series = history of every base in the chain, in root units
    hb_plan = hb[~hb.index.get_level_values("base_sku").isin(npi.keys())]
    bases_lvl = hb_plan.index.get_level_values("base_sku")
    scaled = hb_plan.mul(bases_lvl.map(cum).values, axis=0)
    scaled.index = pd.MultiIndex.from_arrays(
        [bases_lvl.map(root), hb_plan.index.get_level_values(1), hb_plan.index.get_level_values(2)],
        names=["root_base", "customer_group", "location"])
    plan = scaled.groupby(level=[0, 1, 2]).sum()

    fam_idx: dict[str, np.ndarray] = {}
    if cfg["seasonal"]:
        fam_of_series = plan.index.get_level_values(0).map(lambda b: base_family.get(b, ""))
        for fam, tot in plan.groupby(fam_of_series.values).sum().iterrows():
            y = tot.values
            nz = np.flatnonzero(y > 0)
            if fam and len(nz) and H - nz[0] >= int(cfg["min_seasonal_months"]):
                fam_idx[fam] = seasonal_indices(y[nz[0]:], (t0 + nz[0]) % 12)

    fc_root: dict[tuple, np.ndarray] = {}
    meta = []
    n_series = len(plan)
    for i, (key, row) in enumerate(zip(plan.index, plan.values)):
        if progress and i % 200 == 0:
            progress(i / max(n_series, 1))
        nz = np.flatnonzero(row > 0)
        if len(nz) == 0:
            continue
        y = row[nz[0]:]
        res = select_model(y, (t0 + nz[0]) % 12, F, seasonal=bool(cfg["seasonal"]),
                           fam_idx=fam_idx.get(base_family.get(key[0], "")),
                           holdout=int(cfg["holdout"]), min_seasonal=int(cfg["min_seasonal_months"]))
        fc_root[key] = res["f"]
        meta.append((*key, res["model"], res["abs_err"], res["err"], res["act"], res["n"]))
    series = pd.DataFrame(meta, columns=["root_base", "customer_group", "location", "model",
                                         "abs_err", "err", "act", "n_months"])

    # ---- 4. hand the chain forecast to the base that is live in each month
    members: dict[str, list[str]] = {}
    for b in all_bases:
        members.setdefault(root[b], []).append(b)
    win = min(12, H)
    recent_b = hb_plan.iloc[:, -win:].sum(axis=1).mul(bases_lvl.map(cum).values)  # root units
    recent_b_tot = recent_b.groupby(level="base_sku").sum().to_dict()
    recent_b = recent_b.to_dict()
    has_pred = {s[0] for s in succ.values()}

    def holder(b, t):
        while b in succ and succ[b][1] <= t:
            b = succ[b][0]
        return b

    BF: dict[tuple, np.ndarray] = {}
    for (rb, g, l), f in fc_root.items():
        mem = members[rb]
        if len(mem) == 1:
            BF[(rb, g, l)] = f.copy()
            continue
        w = np.array([recent_b.get((b, g, l), 0.0) for b in mem])
        if w.sum() <= 0:
            w = np.array([recent_b_tot.get(b, 0.0) for b in mem])
        if w.sum() <= 0:
            w = np.array([0.0 if b in has_pred else 1.0 for b in mem])
        w = w / w.sum()
        for k, t in enumerate(fc_idx):
            for b, wb in zip(mem, w):
                if wb <= 0:
                    continue
                hb_ = holder(b, t)
                arr = BF.setdefault((hb_, g, l), np.zeros(F))
                arr[k] += f[k] * wb / cum[hb_]

    for b, r in npi.items():
        like_keys = [k for k in BF if k[0] == r["like"]]
        if not like_keys:
            flag("New product rule ignored", b, f"Like item {r['like']} has no forecast")
            continue
        mult = np.zeros(F)
        for k, t in enumerate(fc_idx):
            if t >= r["launch"]:
                step = t - r["launch"]
                mult[k] = r["scale"] * (r["ramp"][step] if step < len(r["ramp"]) else 1.0)
        for (_, g, l) in like_keys:
            BF[(b, g, l)] = BF[(r["like"], g, l)] * mult

    for b in sorted(all_bases):
        if not any(k[0] == b for k in BF) and root[b] == b and b not in has_pred:
            flag("No forecast", b, "No shipment history and no new product rule")

    # ---- 5. split each base SKU forecast to the versions eligible in each month
    recent_v = hv.iloc[:, -min(int(cfg["recent_window"]), H):].sum(axis=1)
    recent_v = recent_v.groupby(level=["version_sku", "customer_group"]).sum().to_dict()
    method = dict(zip(split_settings["base_sku"].astype(str), split_settings["method"]))
    manual = {(str(r.base_sku), str(r.version_sku)): float(r.pct or 0)
              for r in manual_split.itertuples(index=False)}
    vt: dict[str, list[tuple]] = {}
    for r in version_transitions.itertuples(index=False):
        eff = month_to_idx(r.effective_month)
        fv, tv = str(r.from_version).strip(), str(r.to_version).strip()
        if eff is None or fv not in v2b or tv not in v2b or v2b[fv] != v2b[tv]:
            flag("Version transition ignored", f"{fv} -> {tv}",
                 "Both versions must exist under the same base SKU and the rule needs an effective month")
            continue
        scope = str(r.customer_group).strip() or ALL
        vt.setdefault(fv, []).append((scope, eff, "from"))
        vt.setdefault(tv, []).append((scope, eff, "to"))

    it = items.assign(fc=items["forecastable"].astype(bool),
                      li=items["launch_month"].map(month_to_idx),
                      di=items["discontinue_month"].map(month_to_idx))
    vers_of = {b: g.sort_values("version_seq") for b, g in it.groupby("base_sku")}

    def eligibility(v, g):
        e = np.full(F, bool(v.fc))
        if v.li is not None and not pd.isna(v.li):
            e &= fc_idx >= v.li
        if v.di is not None and not pd.isna(v.di):
            e &= fc_idx < v.di
        rules = vt.get(v.version_sku)
        if rules:
            # A rule for this customer group wins over the ALL rule.
            by_role: dict[str, int] = {}
            for scope, eff, role in rules:
                if scope == ALL:
                    by_role.setdefault(role, eff)
            for scope, eff, role in rules:
                if scope == g:
                    by_role[role] = eff
            for role, eff in by_role.items():
                e &= (fc_idx >= eff) if role == "to" else (fc_idx < eff)
        return e

    base_rows, ver_rows, un_rows = [], [], []
    cache: dict[tuple, tuple] = {}
    for (b, g, l), f in BF.items():
        Q = np.rint(np.maximum(f, 0.0)).astype(np.int64)
        if not Q.any():
            continue
        base_rows.append((b, g, l, Q))
        if (b, g) not in cache:
            vs = vers_of[b]
            m = method.get(b, "recent")
            vlist = list(vs.itertuples(index=False))
            if m == "equal":
                w = np.ones(len(vlist))
            elif m == "manual":
                w = np.array([max(0.0, manual.get((b, v.version_sku), 0.0)) for v in vlist])
            else:
                w = np.array([recent_v.get((v.version_sku, g), 0.0) for v in vlist])
            E = np.array([eligibility(v, g) for v in vlist])
            W = w[:, None] * E
            dead = (W.sum(axis=0) <= 0) & E.any(axis=0)
            W[:, dead] = E[:, dead]  # eligible but no weight: share equally
            cache[(b, g)] = ([v.version_sku for v in vlist], W)
        vids, W = cache[(b, g)]
        tot = W.sum(axis=0)
        live = tot > 0
        alloc = np.zeros(W.shape, dtype=np.int64)
        alloc[:, live] = np.floor(Q[live] * W[:, live] / tot[live] + 1e-9).astype(np.int64)
        rem = Q - alloc.sum(axis=0)
        big = W.argmax(axis=0)
        cols = np.flatnonzero(live)
        alloc[big[cols], cols] += rem[cols]  # rounding difference goes to the largest version
        for vid, a in zip(vids, alloc):
            if a.any():
                ver_rows.append((vid, b, g, l, a))
        if (~live & (Q > 0)).any():
            un_rows.append((b, g, l, np.where(live, 0, Q)))

    hist_rows = [(*k, r) for k, r in zip(hv.index, hv.values)]
    res = Result(hist_months=hist_months, fc_months=fc_months)
    res.hist = _long(hist_rows, ["version_sku", "base_sku", "customer_group", "location"], hist_months)
    res.base_fc = _long(base_rows, ["base_sku", "customer_group", "location"], fc_months)
    res.version_fc = _long(ver_rows, ["version_sku", "base_sku", "customer_group", "location"], fc_months)
    res.unallocated = _long(un_rows, ["base_sku", "customer_group", "location"], fc_months)
    res.series = series
    res.chains = pd.DataFrame(
        [(b, root[npi[b]["like"]] if b in npi else root[b],
          f"like item {npi[b]['like']}" if b in npi else "own") for b in sorted(all_bases)],
        columns=["base_sku", "root_base", "source"])
    for (b, g, l), grp in res.unallocated.groupby(["base_sku", "customer_group", "location"]):
        flag("Forecast with no eligible version", f"{b} / {g} / {l}",
             "Held at base SKU level. Check forecastable flags and dates.", len(grp), grp["qty"].sum())
    res.exceptions = pd.DataFrame(exc, columns=["type", "key", "detail", "rows", "qty"])
    if progress:
        progress(1.0)
    return res
