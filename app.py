"""Base SKU Demand Planner - local app.  Start with:  streamlit run app.py"""
from __future__ import annotations

import datetime as dt
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import importlib

from planner import accuracy, breakdown, dataio, db, demo, engine, explore, sample

# Pick up edits to the planner package without restarting the app.
for _m in (engine, db, dataio, explore, breakdown, sample, demo, accuracy):
    importlib.reload(_m)
from planner.engine import ALL, DEFAULTS, run_forecast  # noqa: E402

st.set_page_config(page_title="Base SKU Demand Planner", layout="wide", initial_sidebar_state="expanded")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS_COLORS = {"Discontinued": "#9aa3a0", "Active": "#2a78d6", "Phasing out": "#eda100", "New": "#1baf7a"}
SPLIT_LABELS = {"recent": "Recent shipment mix", "equal": "Equal split", "manual": "Manual percentages"}
MONTH_RE = r"^\d{4}-(0[1-9]|1[0-2])$"


def label(m: str) -> str:
    return dt.date(int(m[:4]), int(m[5:7]), 1).strftime("%b %y")


def fmt(n) -> str:
    return f"{n:,.0f}"


# ------------------------------------------------------------------ data access
@st.cache_data(show_spinner=False)
def load_inputs(version: int) -> dict[str, pd.DataFrame]:
    return {t: db.read(t) for t in db.SCHEMA}


@st.cache_data(show_spinner=False)
def load_result(when: str | None):
    return db.load_result()


@st.cache_data(show_spinner=False)
def excel_export(when: str) -> bytes:
    res, _ = db.load_result()
    return dataio.export_excel(res, db.read("items"))


def inputs() -> dict[str, pd.DataFrame]:
    return load_inputs(db.get_setting("inputs_version", 0) or 0)


def result():
    info = db.get_setting("run_info")
    return load_result(info["when"] if info else None)


def do_run() -> bool:
    d = inputs()
    if d["shipments"].empty or d["items"].empty:
        st.error("Load an item master and shipments first (Data page).")
        return False
    cfg = db.get_config()
    t_start = time.time()
    bar = st.progress(0.0, text="Forecasting...")
    try:
        res = run_forecast(d["items"], d["customers"], d["locations"], d["shipments"], d["version_transitions"],
                           d["hard_transitions"], d["new_products"], d["split_settings"], d["manual_split"], cfg,
                           progress=lambda p: bar.progress(min(1.0, p), text="Forecasting..."))
    except ValueError as e:
        bar.empty()
        st.error(str(e))
        return False
    db.save_result(res, cfg, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    db.set_setting("last_run_seconds", round(time.time() - t_start, 1))
    db.set_setting("last_run_workers", engine.LAST_RUN["workers"])
    bar.empty()
    return True


def load_demo():
    with st.spinner("Loading the demo dataset..."):
        demo.load()
    st.cache_data.clear()
    do_run()


def load_sample():
    db.reset()
    for table, df in sample.build().items():
        db.write(table, df)
    st.cache_data.clear()
    do_run()


def save(table: str, df: pd.DataFrame, msg: str = "Saved. Run the forecast to apply the change."):
    db.write(table, df)
    st.session_state["flash"] = msg
    st.rerun()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Base SKU Demand Planner")
    page = st.radio("Go to", ["Forecast", "Explore", "Lifecycle", "Master data", "Accuracy", "Data"], label_visibility="collapsed")
    st.divider()
    info = db.get_setting("run_info")
    if st.button("Run forecast", type="primary", width="stretch"):
        if do_run():
            st.rerun()
    if info:
        secs = db.get_setting("last_run_seconds")
        st.caption(f"Last run {info['when']}" + (f" · took {secs:g} s" if secs is not None else "")
                   + (f" on {db.get_setting('last_run_workers')} processes" if (db.get_setting("last_run_workers") or 1) > 1 else ""))
        if db.is_stale():
            st.warning("Inputs changed since the last run.")
    else:
        st.caption("No forecast has been run yet.")
    with st.expander("Settings"):
        cfg = db.get_config()
        with st.form("settings"):
            horizon = st.number_input("Forecast horizon, months", 3, 36, int(cfg["horizon"]))
            seasonal = st.checkbox("Use seasonal models", bool(cfg["seasonal"]))
            arima_on = st.checkbox("Use ARIMA and SARIMA models", bool(cfg["arima"]),
                                   help="Adds ARIMA and SARIMA to the models each series tries. About 1 second "
                                        "more per series; turn off for very large runs.",
                                   disabled=not engine.HAS_STATSMODELS)
            workers = st.number_input("Worker processes for ARIMA (0 = automatic)", 0, 64, int(cfg.get("workers", 0)),
                                      help=f"Automatic uses {engine.worker_count({'workers': 0})} on this computer. "
                                           f"Used when there are {engine.PARALLEL_MIN_SERIES}+ series.")
            holdout = st.number_input("Backtest holdout, months", 3, 12, int(cfg["holdout"]))
            recent = st.number_input("Recent-mix window, months", 1, 24, int(cfg["recent_window"]))
            npi_m = st.number_input("New item borrows until it has this many months of its own history", 1, 24,
                                    int(cfg["npi_own_history_months"]))
            by_ch = st.checkbox("Pool non-key customers by channel", bool(cfg["nonkey_by_channel"]))
            drop = st.checkbox("Ignore the latest month (incomplete)", bool(cfg["drop_last_month"]))
            if st.form_submit_button("Save settings"):
                db.set_config({**cfg, "horizon": int(horizon), "seasonal": seasonal, "holdout": int(holdout),
                               "recent_window": int(recent), "arima": bool(arima_on), "workers": int(workers), "npi_own_history_months": int(npi_m),
                               "nonkey_by_channel": by_ch, "drop_last_month": drop})
                st.rerun()

# ------------------------------------------------------------------ banner (every page)
st.markdown(
    """
    <div style="background:#0b6e5f;color:#ffffff;border-radius:8px;padding:14px 20px;margin-bottom:12px;">
      <div style="font-size:1.35rem;font-weight:600;line-height:1.3;">Welcome to Demand Predictor</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if demo.is_demo_mode() or db.get_setting("dataset") == "demo-m5":
    st.caption(demo.NOTE + " See DEMO_DATA.md in the app folder.")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

D = inputs()
res, info = result()

if D["items"].empty and page != "Data":
    st.header("Start by loading data")
    if demo.is_demo_mode() and demo.available():
        st.write("This is the demo copy of the app. Load the demo dataset: 1,000 Walmart food and household "
                 "products across 10 stores and 3 DCs, with versions, transitions and launches added.")
        if st.button("Load demo dataset", type="primary"):
            load_demo()
            st.rerun()
        st.stop()
    st.write("The database is empty. Load the sample data to see how the planner works, "
             "or go to the Data page to load your own item, customer, location and shipment files.")
    c1, c2, _ = st.columns([2, 2, 4])
    if c1.button("Load sample data", type="primary"):
        load_sample()
        st.rerun()
    if demo.available() and c2.button("Load demo dataset (M5)"):
        load_demo()
        st.rerun()
    st.stop()

if res is None and page in ("Forecast", "Lifecycle", "Accuracy"):
    st.header(page)
    st.info("No forecast yet. Press Run forecast in the sidebar.")
    st.stop()


DIMS = {"Location": "location", "Customer group": "customer_group", "Account": "account"}


@st.cache_data(show_spinner="Splitting the forecast by account...")
def account_frames(when: str, version: int):
    """Account-level history, version forecast and unallocated forecast for the last run."""
    r, inf = db.load_result()
    d = load_inputs(version)
    nk = bool(inf["config"].get("nonkey_by_channel", False))
    ha = breakdown.account_history(d["shipments"], d["items"], d["customers"], r.hist_months, nk)
    va = breakdown.split_by_account(r.version_fc, ha, d["customers"], r.hist_months, nk)
    ua = breakdown.split_by_account(r.unallocated, ha, d["customers"], r.hist_months, nk)
    return ha, va, ua


def level_wape_text(bt) -> str:
    """'Base SKU 21% · customer group 26%' for the backtest rows in view."""
    if bt is None:
        return "Run the forecast to see base SKU and customer group levels"
    if bt.empty:
        return ""
    b = accuracy.measure(bt, ["root_base"])["wape"]
    g = accuracy.measure(bt, ["root_base", "customer_group"])["wape"]
    return f"Base SKU level {b:.1f}% · customer group level {g:.1f}%"


def account_names() -> dict[str, str]:
    c = D["customers"]
    return {a: (f"{a} · {n}" if n else a) for a, n in zip(c["account"].astype(str), c["name"].fillna(""))}


def breakdown_rows(df: pd.DataFrame, dimcols: list[str], months: list[str]) -> pd.DataFrame:
    """Months across, one row per combination of dimcols."""
    w = df.pivot_table(index=dimcols, columns="month", values="qty", aggfunc="sum", fill_value=0)
    return w.reindex(columns=months, fill_value=0).reset_index()


def base_names() -> dict[str, str]:
    it = D["items"]
    name = it["base_description"].fillna("").where(it["base_description"].fillna("") != "", it["description"].fillna(""))
    return dict(zip(it["base_sku"], name))


# ------------------------------------------------------------------ Forecast
if page == "Forecast":
    names = base_names()
    bases = sorted(b for b in names if b)
    groups = sorted(set(res.base_fc["customer_group"]) | set(res.hist["customer_group"]),
                    key=lambda g: (g.startswith("All other"), g))
    locs = sorted(D["locations"]["location"])

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    base = c1.selectbox("Base SKU", [ALL] + bases, format_func=lambda b: "All base SKUs" if b == ALL else f"{b} · {names[b]}")
    group = c2.selectbox("Customer group", [ALL] + groups, format_func=lambda g: "All customers" if g == ALL else g)
    loc = c3.selectbox("Ship-from", [ALL] + locs, format_func=lambda l: "All locations" if l == ALL else l)

    # ---------------------------------------------------------- all base SKUs at once
    if base == ALL:
        it = D["items"]
        fam_of = it.groupby("base_sku")["family"].first().fillna("").to_dict()
        fams = sorted({f for f in fam_of.values() if f})
        fam = c4.selectbox("Family", ["All"] + fams, format_func=lambda f: "All families" if f == "All" else f)
        in_bases = [b for b in bases if fam == "All" or fam_of.get(b) == fam]

        def pick_all(df, col="base_sku"):
            df = df[df[col].isin(in_bases)] if col == "base_sku" else df
            if group != ALL:
                df = df[df["customer_group"] == group]
            if loc != ALL:
                df = df[df["location"] == loc]
            return df

        ch = res.chains.set_index("base_sku")
        hist, bfc, vfc, un = pick_all(res.hist), pick_all(res.base_fc), pick_all(res.version_fc), pick_all(res.unallocated)
        roots = {ch.loc[b, "root_base"] for b in in_bases if b in ch.index and ch.loc[b, "source"] == "own"}
        ser = pick_all(res.series[res.series["root_base"].isin(roots)], col="root_base")
        bt_view = (pick_all(res.backtest[res.backtest["root_base"].isin(roots)], col="root_base")
                   if res.backtest is not None else None)

        n12 = bfc[bfc["month"].isin(res.fc_months[:12])]["qty"].sum()
        l12 = hist[hist["month"].isin(res.hist_months[-12:])]["qty"].sum()
        act = ser["act"].sum()
        k1, k2, k3, k4 = st.columns(4)
        for col, title, value, sub in [
                (k1, "Forecast, next 12 months", fmt(n12) + " units",
                 f"{(n12 / l12 - 1) * 100:+.1f}% vs last 12 months shipped ({fmt(l12)})" if l12 else ""),
                (k2, "Backtest error (WAPE)", f"{ser['abs_err'].sum() / act * 100:.1f}% WAPE" if act else "n/a",
                 (f"{level_wape_text(bt_view)} · bias {ser['err'].sum() / act * 100:+.1f}% across {len(ser)} series")
                 if act else ""),
                (k3, "Base SKUs", fmt(len(in_bases)), f"{bfc['base_sku'].nunique()} with a forecast"),
                (k4, "Unallocated", fmt(un["qty"].sum()) + " units",
                 "forecast with no eligible version" if un["qty"].sum() else "every unit reaches a version")]:
            col.caption(title)
            col.markdown(f"#### {value}")
            if sub:
                col.caption(sub)

        # chart: history and forecast stacked by family (or by base SKU inside one family)
        by = "family" if fam == "All" else "base_sku"
        plot = pd.concat([hist.groupby(["base_sku", "month"], as_index=False)["qty"].sum().assign(kind="Shipped"),
                          bfc.groupby(["base_sku", "month"], as_index=False)["qty"].sum().assign(kind="Forecast")])
        plot["family"] = plot["base_sku"].map(fam_of).replace("", "No family")
        top = plot.groupby(by)["qty"].sum().sort_values(ascending=False).index.tolist()
        keep = top[:7]
        plot["stack"] = plot[by].where(plot[by].isin(keep), "Other")
        plot = plot.groupby(["stack", "month", "kind"], as_index=False)["qty"].sum()
        domain = keep + (["Other"] if len(top) > 7 else [])
        months = res.hist_months + res.fc_months
        lab = "timeFormat(datetime(parseInt(slice(datum.value,0,4)), parseInt(slice(datum.value,5,7))-1, 1), '%b %y')"
        x = alt.X("month:N", sort=months, title=None, scale=alt.Scale(domain=months),
                  axis=alt.Axis(values=[m for m in months if m[5:7] in ("01", "04", "07", "10")], labelAngle=0, labelExpr=lab))
        band = alt.Chart(pd.DataFrame({"month": [res.fc_months[0]], "month2": [res.fc_months[-1]]})).mark_rect(
            opacity=0.10, color="#2a78d6").encode(x=x, x2="month2:N")
        bars = alt.Chart(plot).mark_bar().encode(
            x=x, y=alt.Y("qty:Q", title="Units per month", stack="zero"),
            color=alt.Color("stack:N", title="Family" if by == "family" else "Base SKU", legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=domain, range=SERIES[:len(keep)] + (["#9aa3a0"] if len(top) > 7 else []))),
            opacity=alt.Opacity("kind:N", title=None, scale=alt.Scale(domain=["Shipped", "Forecast"], range=[1.0, 0.6]),
                                legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("kind:N", title="Type"),
                     alt.Tooltip("stack:N", title="Family" if by == "family" else "Base SKU"),
                     alt.Tooltip("qty:Q", title="Units", format=",.0f")])
        st.subheader("All base SKUs" + ("" if fam == "All" else f" · {fam}"))
        st.caption("Monthly units, stacked by " + ("family" if by == "family" else "base SKU")
                   + ". The shaded months are forecast. Hard transitions show under the base SKU live in each month.")
        st.altair_chart((band + bars).properties(height=320), width="stretch")

        # grid: one row per base SKU, optional version rows
        st.subheader("Forecast by base SKU")
        g0, g1, g2 = st.columns([3, 2, 2])
        dims = g0.multiselect("Break down by", list(DIMS), default=[], placeholder="No breakdown: one row per base SKU",
                              key="all_dims", help="Rows for each location, customer group and/or account. Accounts in a "
                              "group share its forecast by their shipments over the last 12 months.")
        show_versions = g1.checkbox("Show versions under each base SKU")
        only_unalloc = g2.checkbox("Only base SKUs with unallocated forecast")
        mlabels = [label(m) for m in res.fc_months]
        bwide = bfc.pivot_table(index="base_sku", columns="month", values="qty", aggfunc="sum", fill_value=0)
        bwide = bwide.reindex(columns=res.fc_months, fill_value=0)
        vwide = vfc.pivot_table(index=["base_sku", "version_sku"], columns="month", values="qty", aggfunc="sum",
                                fill_value=0).reindex(columns=res.fc_months, fill_value=0)
        hist12 = hist[hist["month"].isin(res.hist_months[-12:])].groupby("base_sku")["qty"].sum()
        un_b = un.groupby("base_sku")["qty"].sum()
        sm = ser.groupby("root_base").agg(abs_err=("abs_err", "sum"), act=("act", "sum"))
        top_model = ser.groupby("root_base")["model"].agg(lambda m: m.value_counts().index[0]) if len(ser) else pd.Series(dtype=str)
        rows = []
        for b in in_bases:
            if only_unalloc and not un_b.get(b, 0):
                continue
            f = bwide.loc[b] if b in bwide.index else pd.Series(0, index=res.fc_months)
            src = ch.loc[b, "source"] if b in ch.index else "own"
            root = ch.loc[b, "root_base"] if b in ch.index else b
            wape = (sm.loc[root, "abs_err"] / sm.loc[root, "act"] * 100) if root in sm.index and sm.loc[root, "act"] else None
            n12b, l12b = float(f.iloc[:12].sum()), float(hist12.get(b, 0))
            rows.append({"Item": b, "Description": names.get(b, ""), "Family": fam_of.get(b, ""),
                         "Model": (top_model.get(root, "") if src == "own" else "L" + src[1:]),
                         "WAPE %": wape if src == "own" else None,
                         "Last 12 shipped": l12b, "Next 12 forecast": n12b,
                         "Change %": (n12b / l12b - 1) * 100 if l12b else None,
                         "Unallocated": float(un_b.get(b, 0)),
                         **{ml: float(v) for ml, v in zip(mlabels, f.values)}, "Total": float(f.sum())})
            if show_versions and b in vwide.index.get_level_values(0):
                for v, vr in vwide.loc[b].iterrows():
                    rows.append({"Item": f"    {v}", "Description": "", "Family": "", "Model": "", "WAPE %": None,
                                 "Last 12 shipped": float(hist[(hist["base_sku"] == b) & (hist["version_sku"] == v) &
                                                               hist["month"].isin(res.hist_months[-12:])]["qty"].sum()),
                                 "Next 12 forecast": float(vr.iloc[:12].sum()), "Change %": None, "Unallocated": None,
                                 **{ml: float(x) for ml, x in zip(mlabels, vr.values)}, "Total": float(vr.sum())})
        if dims:
            # One row per base SKU (and version) x the chosen dimensions
            dimcols = [DIMS[d] for d in dims]
            if "account" in dimcols:
                ha_all, va_all, ua_all = account_frames(info["when"], db.get_setting("inputs_version", 0) or 0)
                hsrc, vsrc, usrc, anames = pick_all(ha_all), pick_all(va_all), pick_all(ua_all), account_names()
                bsrc = pd.concat([vsrc, usrc])
            else:
                hsrc, vsrc, usrc, bsrc, anames = hist, vfc, un, bfc, {}
            keep_b = [b for b in in_bases if not only_unalloc or un_b.get(b, 0)]
            h12 = hsrc[hsrc["month"].isin(res.hist_months[-12:])]
            levels = [("base_sku", bsrc, h12)]
            if show_versions:
                levels.append(("version_sku", vsrc, h12))
            rows = []
            for lvl, src, hh in levels:
                keys = ["base_sku"] + (["version_sku"] if lvl == "version_sku" else []) + dimcols
                f = breakdown_rows(src[src["base_sku"].isin(keep_b)], keys, res.fc_months) if len(src) else pd.DataFrame(columns=keys)
                l12_ = hh.groupby(keys)["qty"].sum().rename("l12").reset_index() if len(hh) else pd.DataFrame(columns=keys + ["l12"])
                f = f.merge(l12_, on=keys, how="left") if len(f) else f
                for _, r_ in f.iterrows():
                    vals = [float(r_[m]) for m in res.fc_months]
                    l12v, n12v = float(r_.get("l12", 0) or 0), sum(vals[:12])
                    row = {"Item": r_["base_sku"] if lvl == "base_sku" else f"    {r_['version_sku']}",
                           "Description": names.get(r_["base_sku"], "") if lvl == "base_sku" else "",
                           "Family": fam_of.get(r_["base_sku"], "") if lvl == "base_sku" else ""}
                    for d, c in zip(dims, dimcols):
                        row[d] = anames.get(r_[c], r_[c]) if c == "account" else r_[c]
                    row.update({"Last 12 shipped": l12v, "Next 12 forecast": n12v,
                                "Change %": (n12v / l12v - 1) * 100 if l12v else None,
                                **dict(zip(mlabels, vals)), "Total": sum(vals),
                                "_b": r_["base_sku"], "_v": "" if lvl == "base_sku" else r_["version_sku"]})
                    rows.append(row)
            grid = pd.DataFrame(rows)
            if len(grid):
                grid = grid.sort_values(["_b", "_v"] + dims, kind="stable").drop(columns=["_b", "_v"])
        grid = pd.DataFrame(rows) if not dims else grid
        if grid.empty:
            st.info("No base SKUs match this selection.")
            st.stop()
        num = st.column_config.NumberColumn(format="localized")
        st.dataframe(grid, hide_index=True, width="stretch", height=min(38 * (len(grid) + 1) + 3, 640),
                     column_config={"Item": st.column_config.TextColumn(pinned=True),
                                    **{d: st.column_config.TextColumn(pinned=True) for d in dims},
                                    "WAPE %": st.column_config.NumberColumn(format="%.1f"),
                                    "Change %": st.column_config.NumberColumn(format="%+.1f"),
                                    "Last 12 shipped": num, "Next 12 forecast": num, "Unallocated": num, "Total": num,
                                    **{ml: num for ml in mlabels}})
        st.caption(("Model and WAPE show without a breakdown. " if dims else
                    "WAPE is the backtest error of the series behind each base SKU; a new product shows the like item it "
                    "borrows from. ") + "Change % compares the next 12 months with the last 12 shipped."
                   + (" Accounts in a customer group share its forecast by their shipments over the last 12 months."
                      if "Account" in dims else ""))
        e1, e2, e3 = st.columns([3, 3, 3])
        e1.download_button("Download this table (CSV)", grid.to_csv(index=False).encode(),
                           file_name=f"forecast_all_base_skus_{info['when'][:10]}.csv", mime="text/csv")
        if e3.button("Prepare forecast by version and account (CSV)"):
            _, va_all, _ = account_frames(info["when"], db.get_setting("inputs_version", 0) or 0)
            e3.download_button("Download forecast by version and account", va_all.to_csv(index=False).encode(),
                               file_name=f"forecast_by_account_{info['when'][:10]}.csv", mime="text/csv")
        e2.download_button("Download all forecasts (Excel)", excel_export(info["when"]),
                           file_name=f"forecast_{info['when'][:10]}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.stop()
    ss = D["split_settings"]
    cur = dict(zip(ss["base_sku"], ss["method"])).get(base, "recent")
    method = c4.selectbox(f"Version split for {base}", list(SPLIT_LABELS), index=list(SPLIT_LABELS).index(cur),
                          format_func=SPLIT_LABELS.get)
    if method != cur:
        save("split_settings", pd.concat([ss[ss["base_sku"] != base],
                                          pd.DataFrame([[base, method]], columns=["base_sku", "method"])]))

    base_versions = D["items"][D["items"]["base_sku"] == base].sort_values("version_seq")
    if method == "manual":
        ms = D["manual_split"]
        mine = dict(zip(ms.loc[ms["base_sku"] == base, "version_sku"], ms.loc[ms["base_sku"] == base, "pct"]))
        fv = base_versions[base_versions["forecastable"]]
        ed = st.data_editor(
            pd.DataFrame({"version_sku": fv["version_sku"].values,
                          "pct": [float(mine.get(v, round(100 / max(len(fv), 1)))) for v in fv["version_sku"]]}),
            hide_index=True, disabled=["version_sku"], key=f"manual_{base}",
            column_config={"version_sku": "Version", "pct": st.column_config.NumberColumn(
                "Share of base SKU forecast, %", min_value=0, max_value=100, step=5)})
        st.caption("Shares are rescaled across the versions eligible in each month.")
        if st.button("Save percentages"):
            save("manual_split", pd.concat([ms[ms["base_sku"] != base], ed.assign(base_sku=base)]))

    ch = res.chains.set_index("base_sku")
    if base not in ch.index:
        st.info("This base SKU was added after the last run. Run the forecast to include it.")
        st.stop()
    root, source = ch.loc[base, "root_base"], ch.loc[base, "source"]
    chain = [base] if source != "own" else [b for b in ch.index if ch.loc[b, "root_base"] == root and ch.loc[b, "source"] == "own"]
    order = D["items"][D["items"]["base_sku"].isin(chain)].copy()
    order["rank"] = order["base_sku"].map({b: (0 if b != root else 1) for b in chain})
    versions = list(order.sort_values(["rank", "base_sku", "version_seq"])["version_sku"])

    def pick(df, base_col="base_sku", in_bases=chain):
        df = df[df[base_col].isin(in_bases)]
        if group != ALL:
            df = df[df["customer_group"] == group]
        if loc != ALL:
            df = df[df["location"] == loc]
        return df

    hist, vfc, bfc, un = pick(res.hist), pick(res.version_fc), pick(res.base_fc), pick(res.unallocated)
    ser = pick(res.series, "root_base", [root])
    bt_view = pick(res.backtest, "root_base", [root]) if res.backtest is not None else None
    months = res.hist_months + res.fc_months

    # ---- headline figures
    n12 = bfc[bfc["month"].isin(res.fc_months[:12])]["qty"].sum()
    l12 = hist[hist["month"].isin(res.hist_months[-12:])]["qty"].sum()
    act = ser["act"].sum()
    borrowed = f" (from {root})" if root != base else ""
    def stat(col, title, value, sub=""):
        col.caption(title)
        col.markdown(f"#### {value}")
        if sub:
            col.caption(sub)

    k1, k2, k3, k4 = st.columns(4)
    stat(k1, "Forecast, next 12 months", fmt(n12) + " units",
         f"{(n12 / l12 - 1) * 100:+.0f}% vs last 12 months shipped ({fmt(l12)})" if l12 else "New item, no shipments yet")
    stat(k2, "Backtest error (WAPE)" + borrowed, f"{ser['abs_err'].sum() / act * 100:.1f}% WAPE" if act else "n/a",
         (f"{level_wape_text(bt_view)} · bias {ser['err'].sum() / act * 100:+.0f}% (positive = over-forecast)")
         if act else "Not enough history to backtest")
    if len(ser):
        top = ser["model"].value_counts()
        stat(k3, "Model" + borrowed, top.index[0], f"chosen for {top.iloc[0]} of {len(ser)} series")
    src_hist = pick(res.hist, in_bases=[b for b in ch.index if ch.loc[b, "root_base"] == root and ch.loc[b, "source"] == "own"])
    if len(src_hist):
        mm = src_hist["month"].str[5:7]
        cal = src_hist.groupby(mm)["qty"].sum() / src_hist.groupby(mm)["month"].nunique()
        stat(k4, "Seasonal peak" + borrowed, dt.date(2000, int(cal.idxmax()), 1).strftime("%B"),
             f"{cal.max() / cal.mean():.2f}x an average month")

    # ---- chart: history and forecast stacked by version
    plot = pd.concat([hist.groupby(["version_sku", "month"], as_index=False)["qty"].sum().assign(kind="Shipped"),
                      vfc.groupby(["version_sku", "month"], as_index=False)["qty"].sum().assign(kind="Forecast"),
                      un.groupby("month", as_index=False)["qty"].sum().assign(kind="Forecast", version_sku="Unallocated")])
    plot["label"] = plot["month"].map(label)
    plot["order"] = plot["version_sku"].map({v: i for i, v in enumerate(versions + ["Unallocated"])})
    shown = [v for v in versions if v in set(plot["version_sku"])] + (["Unallocated"] if len(un) else [])
    colors = [SERIES[versions.index(v) % len(SERIES)] if v in versions else "#9aa3a0" for v in shown]
    lab = "timeFormat(datetime(parseInt(slice(datum.value,0,4)), parseInt(slice(datum.value,5,7))-1, 1), '%b %y')"
    x = alt.X("month:N", sort=months, title=None, scale=alt.Scale(domain=months),
              axis=alt.Axis(values=[m for m in months if m[5:7] in ("01", "04", "07", "10")], labelAngle=0, labelExpr=lab))
    band = alt.Chart(pd.DataFrame({"month": [res.fc_months[0]], "month2": [res.fc_months[-1]]})).mark_rect(
        opacity=0.10, color="#2a78d6").encode(x=x, x2="month2:N")
    bars = alt.Chart(plot).mark_bar(cornerRadiusTopLeft=1, cornerRadiusTopRight=1).encode(
        x=x, y=alt.Y("qty:Q", title="Units per month", stack="zero"),
        color=alt.Color("version_sku:N", title="Version", scale=alt.Scale(domain=shown, range=colors),
                        legend=alt.Legend(orient="top")),
        opacity=alt.Opacity("kind:N", title=None, scale=alt.Scale(domain=["Shipped", "Forecast"], range=[1.0, 0.6]),
                            legend=alt.Legend(orient="top")),
        order="order:Q",
        tooltip=[alt.Tooltip("label:N", title="Month"), alt.Tooltip("kind:N", title="Type"),
                 alt.Tooltip("version_sku:N", title="Version"), alt.Tooltip("qty:Q", title="Units", format=",.0f")])
    st.subheader("  →  ".join(f"{b} {names[b]}" for b in sorted(chain, key=lambda b: b == root)))
    st.caption("Monthly units by version. The shaded months are forecast.")
    st.altair_chart((band + bars).properties(height=340), width="stretch")

    # ---- lifecycle rules that apply to this chain
    notes = []
    it = D["items"]
    for b in chain:
        dead = it[(it["base_sku"] == b) & ~it["forecastable"] & it["version_sku"].isin(hist["version_sku"])]
        if len(dead):
            notes.append(f"{', '.join(dead['version_sku'])}: not forecastable. History counts toward {b}; no forecast is allocated.")
    for r in D["version_transitions"].itertuples():
        if r.from_version in versions:
            who = "all other customer groups move" if r.customer_group == ALL else f"{r.customer_group} moves"
            notes.append(f"Staggered transition: {who} from {r.from_version} to {r.to_version} in {label(r.effective_month)}.")
    for r in D["hard_transitions"].itertuples():
        if r.from_base in chain:
            notes.append(f"Hard transition: every version of {r.from_base} stops in {label(r.effective_month)}. "
                         f"{r.to_base} takes over using {r.from_base} history x {r.scale:g}.")
    for r in D["new_products"].itertuples():
        if r.base_sku == base:
            notes.append(f"New product: launches {label(r.launch_month)}. Borrows the {r.like_base} forecast x {r.scale:g}"
                         + (f", ramping {r.ramp} then 1." if r.ramp else ".") if source != "own"
                         else f"New product rule no longer applies: {base} has enough history of its own.")
    if len(un):
        notes.append(":red[Some forecast has no eligible version and is held at base SKU level as Unallocated.]")
    if notes:
        st.markdown("**Lifecycle rules in effect**\n" + "\n".join(f"- {n}" for n in notes))

    # ---- grid
    st.subheader("Forecast grid")
    dims = st.multiselect("Break down by", list(DIMS), default=[], placeholder="No breakdown: totals only",
                          key="grid_dims", help="Adds rows for each location, customer group and/or account. "
                          "Accounts in a group share its forecast by their shipments over the last 12 months.")
    dimcols = [DIMS[d] for d in dims]
    if "account" in dimcols:
        _, va_all, ua_all = account_frames(info["when"], db.get_setting("inputs_version", 0) or 0)
        vsrc, usrc = pick(va_all), pick(ua_all)
        anames = account_names()
    else:
        vsrc, usrc, anames = vfc, un, {}
    mcols = [label(m) for m in res.fc_months]
    rows = []

    def line(name, df, dv=None):
        s_ = df.groupby("month")["qty"].sum().reindex(res.fc_months, fill_value=0)
        row = {"Item": name}
        for d, c in zip(dims, dimcols):
            row[d] = "All" if dv is None else (anames.get(dv[c], dv[c]) if c == "account" else dv[c])
        return {**row, **{label(m): int(v) for m, v in s_.items()}, "Total": int(s_.sum())}

    def split_lines(name, df):
        if not dimcols or df.empty:
            return []
        out = []
        for _, r_ in breakdown_rows(df, dimcols, res.fc_months).iterrows():
            row = {"Item": name}
            for d, c in zip(dims, dimcols):
                row[d] = anames.get(r_[c], r_[c]) if c == "account" else r_[c]
            vals = [int(r_[m]) for m in res.fc_months]
            out.append({**row, **dict(zip(mcols, vals)), "Total": sum(vals)})
        return out

    for b in sorted(chain, key=lambda b: b == root):
        bb, vv, uu = bfc[bfc["base_sku"] == b], vfc[vfc["base_sku"] == b], un[un["base_sku"] == b]
        rows.append(line(f"{b} base SKU forecast", bb))
        rows += split_lines(f"{b} base SKU forecast", pd.concat([vsrc[vsrc["base_sku"] == b], usrc[usrc["base_sku"] == b]]))
        for v in [v for v in versions if v in set(it.loc[it["base_sku"] == b, "version_sku"])]:
            rows.append(line(f"    {v}", vv[vv["version_sku"] == v]))
            rows += split_lines(f"    {v}", vsrc[(vsrc["base_sku"] == b) & (vsrc["version_sku"] == v)])
        if len(uu):
            rows.append(line("    Unallocated", uu))
            rows += split_lines("    Unallocated", usrc[usrc["base_sku"] == b])
        diff = line("    Check: base minus versions", bb)
        alloc = line("", pd.concat([vv, uu]))
        rows.append({k: (v if k in ["Item"] + dims else v - alloc[k]) for k, v in diff.items()})
    grid = pd.DataFrame(rows)
    st.dataframe(grid, hide_index=True, width="stretch", height=min(38 * (len(grid) + 1) + 3, 640),
                 column_config={"Item": st.column_config.TextColumn(pinned=True),
                                **{d: st.column_config.TextColumn(pinned=True) for d in dims},
                                **{label(m): st.column_config.NumberColumn(format="localized") for m in res.fc_months},
                                "Total": st.column_config.NumberColumn(format="localized")})
    if dimcols:
        st.caption("\"All\" rows are totals; the rows after them break the same figures down. "
                   + ("Accounts in a customer group share its forecast by their shipments over the last 12 months."
                      if "account" in dimcols else ""))

    e0, e1, e2 = st.columns([3, 3, 3])
    e0.download_button("Download this table (CSV)", grid.to_csv(index=False).encode(),
                       file_name=f"forecast_{base}_{info['when'][:10]}.csv", mime="text/csv")
    e1.download_button("Download all forecasts (Excel)", excel_export(info["when"]),
                       file_name=f"forecast_{info['when'][:10]}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    e2.download_button("Download version forecast (CSV)", res.version_fc.to_csv(index=False).encode(),
                       file_name=f"version_forecast_{info['when'][:10]}.csv", mime="text/csv")

# ------------------------------------------------------------------ Explore
elif page == "Explore":
    PATTERN_COLORS = {"Smooth": "#2a78d6", "Erratic": "#eb6834", "Intermittent": "#1baf7a", "Lumpy": "#eda100",
                      "Too short": "#9aa3a0"}
    st.header("Explore data")
    st.caption("A profile of the shipments loaded now, before any forecast: data problems, demand patterns, trend, "
               "seasonality and outliers, with suggested changes to settings and master data.")
    cfg = db.get_config()
    ver = db.get_setting("inputs_version", 0) or 0

    @st.cache_data(show_spinner="Profiling the data...")
    def explore_data(version, level, by_channel, drop_last, chains):
        d = load_inputs(version)
        wide, months = explore.build_series(d["items"], d["customers"], d["shipments"], d["hard_transitions"], level,
                                            nonkey_by_channel=by_channel, drop_last_month=drop_last, combine_chains=chains)
        return wide, months, explore.profile(wide, months, d["items"])

    @st.cache_data(show_spinner=False)
    def explore_checks(version, cfg_items):
        d, c = load_inputs(version), dict(cfg_items)
        checks = explore.quality_checks(d, c)
        _, _, prof_f = explore_data(version, "base_group_loc", c["nonkey_by_channel"], c["drop_last_month"], True)
        keys = sorted(set(engine.customer_groups(d["customers"], c["nonkey_by_channel"]))
                      - {g for g in engine.customer_groups(d["customers"], c["nonkey_by_channel"]) if g.startswith("All other")})
        return checks, explore.recommendations(prof_f, checks, c, keys)

    if D["shipments"].empty:
        st.info("Load shipments on the Data page to explore them.")
        st.stop()

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    level = c1.selectbox("Level", list(explore.LEVELS), index=2, format_func=lambda k: explore.LEVELS[k][0])
    chains = c4.checkbox("Combine hard-transition chains", True,
                         help="Show a successor base SKU with its predecessor's history, as the forecast does.")
    wide, months, prof = explore_data(ver, level, bool(cfg["nonkey_by_channel"]), bool(cfg["drop_last_month"]), chains)
    if prof.empty:
        st.info("No shipments match the item and customer masters yet.")
        st.stop()
    fams = sorted(f for f in prof["family"].unique() if f) if "family" in prof else []
    fam = c2.selectbox("Family", ["All"] + fams)
    grp = "All"
    if "customer_group" in prof:
        grp = c3.selectbox("Customer group", ["All"] + sorted(prof["customer_group"].unique(),
                                                             key=lambda g: (g.startswith("All other"), g)))
    view = prof
    if fam != "All":
        view = view[view["family"] == fam]
    if grp != "All":
        view = view[view["customer_group"] == grp]
    act = view[view["months_active"] > 0]

    # ---- 1. health
    st.subheader("Data health")
    l12, p12 = act["last12"].sum(), act["prior12"].fillna(0).sum()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("History", f"{len(months)} months")
    h1.caption(f"{label(months[0])} to {label(months[-1])}")
    h2.metric("Series at this level", fmt(len(act)))
    h3.metric("Shipped, last 12 months", fmt(l12) + " units")
    h4.metric("Change vs prior 12 months", f"{(l12 / p12 - 1) * 100:+.1f}%" if p12 > 0 else "n/a")

    sel_keys = view.set_index(list(wide.index.names)).index
    tot = wide.loc[wide.index.isin(sel_keys)].sum(axis=0).clip(lower=0)
    tdf = pd.DataFrame({"month": months, "qty": tot.values})
    tdf["avg12"] = tdf["qty"].rolling(12).mean()
    tx = alt.X("month:N", sort=months, title=None, axis=alt.Axis(
        values=[m for m in months if m[5:7] in ("01", "04", "07", "10")], labelAngle=0,
        labelExpr="timeFormat(datetime(parseInt(slice(datum.value,0,4)), parseInt(slice(datum.value,5,7))-1, 1), '%b %y')"))
    tbars = alt.Chart(tdf).mark_bar(color="#9ec5f4").encode(
        x=tx, y=alt.Y("qty:Q", title="Units per month"),
        tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("qty:Q", title="Units", format=",.0f"),
                 alt.Tooltip("avg12:Q", title="12-month average", format=",.0f")])
    tline = alt.Chart(tdf.dropna()).mark_line(color="#1c5cab", strokeWidth=2).encode(x=tx, y="avg12:Q")
    st.caption("Total monthly shipments for the selection. The line is the 12-month moving average: its slope is the underlying trend.")
    st.altair_chart((tbars + tline).properties(height=220), width="stretch")

    checks, recs = explore_checks(ver, tuple(sorted(cfg.items())))
    st.markdown("**Data checks**")
    if checks.empty:
        st.success("No data problems found.")
    else:
        st.dataframe(checks.rename(columns=str.capitalize), hide_index=True, width="stretch",
                     column_config={"Detail": st.column_config.TextColumn(width="large"),
                                    "Examples": st.column_config.TextColumn(width="medium")})

    # ---- 2. recommendations
    st.subheader("Suggested changes")
    st.caption("Worked out at the forecast level over all data. Settings are in the sidebar; flags and dates on Master data.")
    if recs.empty:
        st.success("Nothing to change: the current settings suit this data.")
    else:
        st.dataframe(recs.rename(columns={"area": "What", "current": "Now", "suggested": "Suggested", "why": "Why"}),
                     hide_index=True, width="stretch", column_config={"Why": st.column_config.TextColumn(width="large")})

    # ---- 3. demand patterns
    st.subheader("Demand patterns")
    st.caption(f"Each dot is a series. Average demand interval (ADI) = months per month with shipments; CV² = how much "
               f"the non-zero quantities vary. Cut-offs at ADI {explore.ADI_CUT} and CV² {explore.CV2_CUT} give four patterns. "
               "Smooth series forecast well; intermittent and lumpy ones use the Croston model and carry more error.")
    pc1, pc2 = st.columns([3, 2])
    sc = act[act["pattern"] != "Too short"].copy()
    sc["adi_plot"] = sc["adi"].clip(upper=6)
    sc["cv2_plot"] = sc["cv2"].clip(upper=3)
    pts = alt.Chart(sc).mark_circle(opacity=0.75, stroke="white", strokeWidth=1).encode(
        x=alt.X("adi_plot:Q", title="Average demand interval, months (capped at 6)", scale=alt.Scale(domain=[1, 6], zero=False, clamp=True)),
        y=alt.Y("cv2_plot:Q", title="CV² of non-zero quantities (capped at 3)", scale=alt.Scale(domain=[0, 3])),
        size=alt.Size("last12:Q", title="Last 12 months", legend=None, scale=alt.Scale(range=[30, 600])),
        color=alt.Color("pattern:N", title="Pattern", legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=list(PATTERN_COLORS)[:4], range=list(PATTERN_COLORS.values())[:4])),
        tooltip=[alt.Tooltip("label:N", title="Series"), alt.Tooltip("pattern:N", title="Pattern"),
                 alt.Tooltip("adi:Q", title="ADI", format=".2f"), alt.Tooltip("cv2:Q", title="CV²", format=".2f"),
                 alt.Tooltip("last12:Q", title="Last 12 months", format=",.0f")])
    cuts = alt.Chart(pd.DataFrame({"v": [explore.ADI_CUT]})).mark_rule(strokeDash=[4, 3], color="#888").encode(x="v:Q") + \
        alt.Chart(pd.DataFrame({"h": [explore.CV2_CUT]})).mark_rule(strokeDash=[4, 3], color="#888").encode(y="h:Q")
    pc1.altair_chart((cuts + pts).properties(height=320), width="stretch")
    ps = explore.pattern_summary(view)
    pc2.markdown("**Series and volume by pattern**")
    pc2.dataframe(ps.rename(columns={"pattern": "Pattern", "series": "Series", "volume": "Last 12 months",
                                     "volume_share": "Share of volume"}), hide_index=True, width="stretch",
                  column_config={"Last 12 months": st.column_config.NumberColumn(format="localized"),
                                 "Share of volume": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    pc2.markdown("**ABC x XYZ (number of series)**")
    pc2.dataframe(explore.abc_xyz(view), width="stretch")
    pc2.caption("ABC: A = series making the first 80% of last-12-month volume, B the next 15%, C the rest. "
                "XYZ: variability after removing seasonality; X under 0.5, Y under 1, Z above. AX series are the "
                "ones to get right; CZ series are best planned at a higher level.")

    # ---- 4. trend and seasonality
    st.subheader("Trend and seasonality")
    tc1, tc2 = st.columns(2)
    tr = act.groupby("trend")["last12"].sum().reindex(["Down", "Flat", "Up", "Too short"]).dropna().reset_index()
    tr["share"] = tr["last12"] / max(tr["last12"].sum(), 1)
    tc1.caption(f"Volume by trend: the straight-line slope of the last 36 months after removing seasonality. "
                f"Up or Down means more than {explore.TREND_CUT:.0f}% a year.")
    tc1.altair_chart(alt.Chart(tr).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("trend:N", sort=["Down", "Flat", "Up", "Too short"], title=None),
        x=alt.X("share:Q", title="Share of last-12-month volume", axis=alt.Axis(format="%")),
        color=alt.Color("trend:N", legend=None, scale=alt.Scale(domain=["Down", "Flat", "Up", "Too short"],
                                                               range=["#e34948", "#9aa3a0", "#1baf7a", "#cfd4d2"])),
        tooltip=[alt.Tooltip("trend:N", title="Trend"), alt.Tooltip("share:Q", title="Share", format=".0%")]
    ).properties(height=170), width="stretch")
    pk = act[act["peak_month"] != ""].copy()
    pk["seasonal"] = np.where(pk["seasonal_strength"] >= 0.4, "Clearly seasonal", "Weak or none")
    tc2.caption("Volume by peak month. Clearly seasonal = the monthly pattern explains at least 40% of the variation "
                "(strength 0.4). Series under 24 months are left out.")
    tc2.altair_chart(alt.Chart(pk).mark_bar().encode(
        x=alt.X("peak_month:N", sort=explore.MONTHS, title=None, scale=alt.Scale(domain=explore.MONTHS)),
        y=alt.Y("sum(last12):Q", title="Last 12 months, units"),
        color=alt.Color("seasonal:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Clearly seasonal", "Weak or none"], range=["#2a78d6", "#cfd4d2"])),
        tooltip=[alt.Tooltip("peak_month:N", title="Peak"), alt.Tooltip("seasonal:N", title="Seasonality"),
                 alt.Tooltip("sum(last12):Q", title="Units", format=",.0f")]).properties(height=170), width="stretch")

    # ---- 5. series profiles
    st.subheader("Series profiles")
    f1, f2, f3 = st.columns([3, 2, 2])
    pats = f1.multiselect("Pattern", list(PATTERN_COLORS), default=[], placeholder="All patterns")
    only_out = f2.checkbox("Only series with outliers")
    only_inactive = f3.checkbox("Only inactive (6+ months without shipments)")
    tbl = act
    if pats:
        tbl = tbl[tbl["pattern"].isin(pats)]
    if only_out:
        tbl = tbl[tbl["outliers"] > 0]
    if only_inactive:
        tbl = tbl[tbl["months_since_last"] >= 6]
    cols = ["label", "family", "abc", "xyz", "pattern", "trend", "trend_pct", "yoy_pct", "seasonal_strength", "peak_month",
            "cov", "adi", "cv2", "zero_pct", "outliers", "outlier_months", "recent_shift_pct", "months_active",
            "last_month", "last12"]
    cols = [c for c in cols if c in tbl.columns]
    st.dataframe(tbl.sort_values("last12", ascending=False)[cols], hide_index=True, width="stretch", height=360,
                 column_config={
                     "label": st.column_config.TextColumn("Series", pinned=True), "family": "Family", "abc": "ABC", "xyz": "XYZ",
                     "pattern": "Pattern", "trend": "Trend",
                     "trend_pct": st.column_config.NumberColumn("Trend %/yr", format="%+.0f"),
                     "yoy_pct": st.column_config.NumberColumn("Last 12 vs prior 12 %", format="%+.0f"),
                     "seasonal_strength": st.column_config.NumberColumn("Seasonal strength", format="%.2f"),
                     "peak_month": "Peak", "cov": st.column_config.NumberColumn("Variability (CoV)", format="%.2f"),
                     "adi": st.column_config.NumberColumn("ADI", format="%.2f"),
                     "cv2": st.column_config.NumberColumn("CV²", format="%.2f"),
                     "zero_pct": st.column_config.NumberColumn("Zero months %", format="%.0f"),
                     "outliers": st.column_config.NumberColumn("Outliers", format="%d"), "outlier_months": "Outlier months",
                     "recent_shift_pct": st.column_config.NumberColumn("Last 3 vs prior 12 %", format="%+.0f"),
                     "months_active": st.column_config.NumberColumn("Months of history", format="%d"),
                     "last_month": "Last shipment", "last12": st.column_config.NumberColumn("Last 12 months", format="localized")})
    st.download_button("Download profiles (CSV)", tbl[cols].to_csv(index=False).encode(), file_name="series_profiles.csv",
                       mime="text/csv")

    # ---- 6. drill-down
    st.subheader("Look at one series")
    opts = act.sort_values("last12", ascending=False)["label"].tolist()
    pick_s = st.selectbox("Series", opts, help="Sorted by last-12-month volume")
    row = act[act["label"] == pick_s].iloc[0]
    key = tuple(row[k] for k in wide.index.names)
    y = wide.loc[key if len(key) > 1 else key[0]].values
    a = explore.analyse(y, engine.month_to_idx(months[0]))
    st0 = a["_start"]
    sdf = pd.DataFrame({"month": months[st0:], "qty": np.maximum(y[st0:], 0), "outlier": a["outlier_flags"]})
    nn = len(sdf)
    cal = (engine.month_to_idx(months[st0]) + np.arange(nn)) % 12
    idx = a["indices"] if a.get("seasonal_strength", 0) and a["seasonal_strength"] >= 0.3 else np.ones(12)
    dd = sdf["qty"].values / idx[cal]
    if nn >= 12 and dd.mean() > 0:
        w = min(36, nn)
        coef = np.polyfit(np.arange(w), dd[-w:], 1)
        tline_v = np.full(nn, np.nan)
        tline_v[-w:] = np.polyval(coef, np.arange(w))
        sdf["trend"] = tline_v
    sx = alt.X("month:N", sort=months[st0:], title=None, axis=alt.Axis(
        values=[m for m in months[st0:] if m[5:7] in ("01", "04", "07", "10")], labelAngle=0,
        labelExpr="timeFormat(datetime(parseInt(slice(datum.value,0,4)), parseInt(slice(datum.value,5,7))-1, 1), '%b %y')"))
    base_line = alt.Chart(sdf).mark_line(color="#2a78d6", strokeWidth=2, point=alt.OverlayMarkDef(size=25, color="#2a78d6")).encode(
        x=sx, y=alt.Y("qty:Q", title="Units per month"),
        tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("qty:Q", title="Units", format=",.0f")])
    layers = base_line
    if "trend" in sdf:
        layers += alt.Chart(sdf.dropna(subset=["trend"])).mark_line(strokeDash=[5, 4], color="#6b7280").encode(
            x=sx, y="trend:Q", tooltip=[alt.Tooltip("trend:Q", title="Trend level (deseasonalised)", format=",.0f")])
    if sdf["outlier"].any():
        layers += alt.Chart(sdf[sdf["outlier"]]).mark_point(size=160, shape="circle", filled=False, strokeWidth=2.5,
                                                            color="#e34948").encode(
            x=sx, y="qty:Q", tooltip=[alt.Tooltip("month:N", title="Outlier"), alt.Tooltip("qty:Q", title="Units", format=",.0f")])
    d1, d2 = st.columns([3, 1])
    d1.altair_chart(layers.properties(height=280), width="stretch")
    d1.caption("Blue: monthly shipments. Dashed: straight-line trend of the deseasonalised level over the last 36 months. "
               "Red rings: outliers (more than 3.5 robust standard deviations and 25% away from the months around them).")
    if a["months_active"] >= 24:
        sidx = pd.DataFrame({"month": explore.MONTHS, "index": a["indices"]})
        d2.altair_chart(alt.Chart(sidx).mark_bar(cornerRadiusEnd=2, color="#2a78d6").encode(
            x=alt.X("index:Q", title="Seasonal index (1 = average)"), y=alt.Y("month:N", sort=explore.MONTHS, title=None),
            tooltip=[alt.Tooltip("month:N"), alt.Tooltip("index:Q", format=".2f")]).properties(height=280), width="stretch")
    else:
        d2.info("Under 24 months: no seasonal profile of its own.")
    facts = [f"**{a['pattern']}** demand, **{a['trend'].lower()}** trend"
             + (f" ({a['trend_pct']:+.1f}% a year)" if not np.isnan(a.get("trend_pct", np.nan)) else ""),
             f"seasonal strength {a['seasonal_strength']:.2f}, peak in {a['peak_month']}" if a["months_active"] >= 24
             else "too short for its own seasonality",
             f"{a['zero_pct']:.0f}% of months with no shipments",
             f"{a['outliers']} outlier{'s' if a['outliers'] != 1 else ''}" + (f": {a['outlier_months']}" if a["outliers"] else "")]
    st.markdown("; ".join(facts) + ".")

# ------------------------------------------------------------------ Lifecycle
elif page == "Lifecycle":
    st.header("Lifecycle")
    names = base_names()
    it = D["items"]
    ruled = set(D["hard_transitions"]["from_base"]) | set(D["hard_transitions"]["to_base"]) | set(D["new_products"]["base_sku"])
    ruled |= set(it.loc[it["version_sku"].isin(D["version_transitions"]["from_version"]), "base_sku"])
    all_bases = sorted(b for b in names if b)
    default = sorted(ruled & set(all_bases)) or all_bases[:5]
    pick_b = st.multiselect("Base SKUs on the calendar", all_bases, default, format_func=lambda b: f"{b} · {names[b]}")

    months = res.hist_months + res.fc_months
    act = pd.concat([res.hist[["version_sku", "month"]].assign(kind="h"), res.version_fc[["version_sku", "month"]].assign(kind="f")])
    act = act[act["version_sku"].isin(it.loc[it["base_sku"].isin(pick_b), "version_sku"])]
    if act.empty:
        st.info("Nothing to show for this selection.")
    else:
        g = act.groupby("version_sku")
        span = pd.DataFrame({"start": g["month"].min(), "end": g["month"].max(),
                             "has_h": g["kind"].apply(lambda s: (s == "h").any()),
                             "has_f": g["kind"].apply(lambda s: (s == "f").any())}).reset_index()
        last = res.fc_months[-1]
        span["status"] = np.select([~span["has_f"], ~span["has_h"], span["end"] < last],
                                   ["Discontinued", "New", "Phasing out"], "Active")
        nxt = lambda m: (pd.Period(m, "M") + 1).to_timestamp()  # noqa: E731
        span["from"] = pd.to_datetime(span["start"])
        span["to"] = span["end"].map(nxt)
        span = span.merge(it[["version_sku", "base_sku", "version_seq"]], on="version_sku").sort_values(["base_sku", "version_seq"])
        bars = alt.Chart(span).mark_bar(height=12, cornerRadius=2).encode(
            x=alt.X("from:T", title=None, axis=alt.Axis(format="%b %y"), scale=alt.Scale(domain=[pd.to_datetime(months[0]), nxt(months[-1])])), x2="to:T",
            y=alt.Y("version_sku:N", sort=list(span["version_sku"]), title=None),
            color=alt.Color("status:N", title="Status", legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values()))),
            tooltip=[alt.Tooltip("version_sku:N", title="Version"), alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("start:N", title="First month"), alt.Tooltip("end:N", title="Last month")])
        today = alt.Chart(pd.DataFrame({"t": [pd.to_datetime(res.fc_months[0])]})).mark_rule(strokeDash=[4, 3]).encode(x="t:T")
        st.caption("Each bar runs from a version's first shipment to its last forecast month. The dashed line is the first forecast month.")
        st.altair_chart((bars + today).properties(height=alt.Step(26)), width="stretch")

    month_col = lambda t: st.column_config.TextColumn(t, validate=MONTH_RE, help="YYYY-MM", required=True)  # noqa: E731
    st.subheader("Version transitions")
    st.caption("One version replaces another inside a base SKU. Use a customer group name to move that customer first, "
               "and ALL for everyone else. This is how a staggered transition is set up.")
    vt = st.data_editor(D["version_transitions"], num_rows="dynamic", hide_index=True, key="vt", width="stretch",
                        column_config={"from_version": st.column_config.TextColumn("From version", required=True),
                                       "to_version": st.column_config.TextColumn("To version", required=True),
                                       "customer_group": st.column_config.TextColumn("Customer group or ALL", default=ALL, required=True),
                                       "effective_month": month_col("Effective month")})
    st.subheader("Hard transitions")
    st.caption("A whole base SKU is discontinued and replaced. The successor is forecast on the combined history.")
    ht = st.data_editor(D["hard_transitions"], num_rows="dynamic", hide_index=True, key="ht", width="stretch",
                        column_config={"from_base": st.column_config.TextColumn("From base SKU", required=True),
                                       "to_base": st.column_config.TextColumn("To base SKU", required=True),
                                       "effective_month": month_col("Effective month"),
                                       "scale": st.column_config.NumberColumn("Volume factor", default=1.0, min_value=0.0, step=0.05)})
    st.subheader("New products")
    st.caption("A new base SKU borrows the forecast of a like item, times a factor, with an optional ramp for the first months "
               "(for example 0.3,0.6,0.85).")
    npd = st.data_editor(D["new_products"], num_rows="dynamic", hide_index=True, key="npd", width="stretch",
                         column_config={"base_sku": st.column_config.TextColumn("New base SKU", required=True),
                                        "like_base": st.column_config.TextColumn("Like item (base SKU)", required=True),
                                        "launch_month": month_col("Launch month"),
                                        "scale": st.column_config.NumberColumn("Volume factor", default=1.0, min_value=0.0, step=0.05),
                                        "ramp": st.column_config.TextColumn("Ramp")})
    if st.button("Save rules", type="primary"):
        db.write("version_transitions", vt.dropna(subset=["from_version", "to_version", "effective_month"]))
        db.write("hard_transitions", ht.dropna(subset=["from_base", "to_base", "effective_month"]))
        save("new_products", npd.dropna(subset=["base_sku", "like_base", "launch_month"]))

# ------------------------------------------------------------------ Master data
elif page == "Master data":
    st.header("Master data")
    t1, t2, t3 = st.tabs([f"Items ({len(D['items']):,})", f"Customers ({len(D['customers']):,})",
                          f"Locations ({len(D['locations']):,})"])
    opt_month = lambda t: st.column_config.TextColumn(t, validate=MONTH_RE, help="YYYY-MM, or leave blank")  # noqa: E731
    with t1:
        st.caption("Every version rolls up to its base SKU. Untick Forecastable to stop a version receiving forecast; "
                   "its history still counts toward the base SKU.")
        ed = st.data_editor(D["items"], num_rows="dynamic", hide_index=True, key="items", width="stretch",
                            column_config={"version_sku": st.column_config.TextColumn("Version SKU", required=True),
                                           "description": "Description", "base_sku": "Base SKU",
                                           "base_description": "Base description",
                                           "version_seq": st.column_config.NumberColumn("Seq", step=1),
                                           "family": "Family",
                                           "forecastable": st.column_config.CheckboxColumn("Forecastable", default=True),
                                           "launch_month": opt_month("Launch month"),
                                           "discontinue_month": opt_month("Discontinue month")})
        if st.button("Save items", type="primary"):
            if ed["version_sku"].duplicated().any():
                st.error("Version SKU must be unique.")
            else:
                save("items", ed)
    with t2:
        st.caption("Key customers are forecast on their own. Everyone else is forecast together as All other. "
                   "Give two key accounts the same customer group to plan them as one.")
        ed = st.data_editor(D["customers"], num_rows="dynamic", hide_index=True, key="customers", width="stretch",
                            column_config={"account": st.column_config.TextColumn("Account", required=True),
                                           "name": "Customer", "channel": "Channel",
                                           "key_customer": st.column_config.CheckboxColumn("Key customer", default=False),
                                           "customer_group": "Customer group (optional)"})
        if st.button("Save customers", type="primary"):
            if ed["account"].duplicated().any():
                st.error("Account must be unique.")
            else:
                save("customers", ed)
    with t3:
        ed = st.data_editor(D["locations"], num_rows="dynamic", hide_index=True, key="locations", width="stretch",
                            column_config={"location": st.column_config.TextColumn("Location", required=True),
                                           "name": "Name", "type": "Type", "region": "Region"})
        if st.button("Save locations", type="primary"):
            save("locations", ed)

# ------------------------------------------------------------------ Accuracy
elif page == "Accuracy":
    st.header("Backtest accuracy")
    s = res.series[res.series["act"] > 0]
    bt = res.backtest
    holdout = int(info["config"].get("holdout", DEFAULTS["holdout"]))
    st.caption(f"Each series (base SKU x customer group x location) was forecast for its last {holdout} months without "
               "seeing them. WAPE = total absolute error / total shipments; bias = total error / total shipments "
               "(positive = over-forecast). For coarser levels, forecasts and actuals are added together to that level "
               "for each month before comparing, so errors in the pieces cancel as they would in a report. "
               "Hard-transition chains are listed under the successor base SKU.")
    if s.empty:
        st.info("No series has enough history to backtest.")
    elif bt is None:
        st.info("This forecast was run before error by level was added. Press Run forecast to see it.")
    else:
        fams = D["items"].groupby("base_sku")["family"].first().fillna("").to_dict()
        f1, f2, _ = st.columns([2, 2, 3])
        fam_list = sorted({fams.get(b, "") for b in bt["root_base"].unique()} - {""})
        fam = f1.selectbox("Family", ["All"] + fam_list, format_func=lambda f: "All families" if f == "All" else f)
        loc = f2.selectbox("Ship-from", ["All"] + sorted(bt["location"].unique()),
                           format_func=lambda l: "All locations" if l == "All" else l)
        v = bt
        if fam != "All":
            v = v[v["root_base"].map(fams) == fam]
        if loc != "All":
            v = v[v["location"] == loc]

        st.subheader("Error by level")
        lv = accuracy.by_level(v)
        cols = st.columns(len(lv))
        for col, r_ in zip(cols, lv.itertuples()):
            col.caption(r_.level)
            col.markdown(f"#### {r_.wape:.1f}% WAPE" if r_.wape == r_.wape else "#### n/a")
            col.caption(f"bias {r_.bias:+.1f}%" if r_.bias == r_.bias else "")
        st.caption(f"{v[['root_base', 'customer_group', 'location']].drop_duplicates().shape[0]:,} series, "
                   f"{fmt(v['act'].sum())} units shipped in the test months. Error falls at coarser levels because "
                   "over- and under-forecasts in the pieces cancel.")

        st.subheader("By base SKU")
        st.caption("Base SKU = all customers and locations added together. Each customer group column = that base SKU "
                   "for that group, locations added together.")
        base = accuracy.by_key(v, ["root_base"], ["root_base"])
        grp = accuracy.by_key(v, ["root_base", "customer_group"], ["root_base", "customer_group"])["wape"].unstack()
        tb = pd.concat([base["act"].rename("Units in test"), base["wape"].rename("Base SKU WAPE %"),
                        base["bias"].rename("Base SKU bias %"), grp], axis=1).sort_values("Units in test", ascending=False)
        tb.index.name = "Base SKU"
        st.dataframe(tb, width="stretch", height=min(38 * (len(tb) + 1) + 3, 520),
                     column_config={"Units in test": st.column_config.NumberColumn(format="localized"),
                                    **{c: st.column_config.NumberColumn(format="%.1f") for c in tb.columns if c != "Units in test"}})

        st.subheader("By customer group")
        st.caption("All items: the group's total across every item, the coarsest view. Item level: each base SKU for "
                   "the group, then added up, which is the level planners usually review.")
        tot = accuracy.by_key(v, ["customer_group"], ["customer_group"])
        itm = accuracy.by_key(v, ["customer_group"], ["root_base", "customer_group"])
        rows = pd.DataFrame({"Customer group": tot.index, "Units in test": tot["act"].values,
                             "WAPE %, all items": tot["wape"].values, "WAPE %, item level": itm["wape"].reindex(tot.index).values,
                             "Bias %": itm["bias"].reindex(tot.index).values})
        cg = rows.sort_values("Units in test", ascending=False)
        st.dataframe(cg, hide_index=True, width="stretch",
                     column_config={"Units in test": st.column_config.NumberColumn(format="localized"),
                                    "WAPE %, all items": st.column_config.NumberColumn(format="%.1f"),
                                    "WAPE %, item level": st.column_config.NumberColumn(format="%.1f"),
                                    "Bias %": st.column_config.NumberColumn(format="%+.1f")})

        st.subheader("Models chosen")
        mc = res.series["model"].value_counts().rename_axis("Model").reset_index(name="Series")
        st.dataframe(mc, hide_index=True)

# ------------------------------------------------------------------ Data
elif page == "Data":
    st.header("Data")
    c = st.columns(4)
    for col, (t, name) in zip(c, [("items", "Items"), ("customers", "Customers"), ("locations", "Locations"),
                                  ("shipments", "Shipment rows (monthly)")]):
        col.metric(name, fmt(db.count(t)))
    if not D["shipments"].empty:
        st.caption(f"Shipment history runs {label(D['shipments']['month'].min())} to {label(D['shipments']['month'].max())}.")

    rule_counts = {t: db.count(t) for t in ("version_transitions", "hard_transitions", "new_products", "split_settings", "manual_split")}
    st.caption("Rules loaded: " + " · ".join(f"{dataio.SPEC[t][1]} {n:,}" for t, n in rule_counts.items()))

    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ORDER = [s[0] for s in dataio.SPECS]

    def load_table(table, clean, mode="Replace everything"):
        if table == "shipments" and mode != "Replace everything":
            db.upsert_shipments(clean)
        else:
            db.write(table, clean)

    st.subheader("Load everything from one workbook")
    st.write("One Excel file with a sheet per table: items, customers, locations, shipments, version transitions, "
             "hard transitions, new products, split methods and manual split percentages. Sheets you leave out are not changed.")
    w1, w2, _ = st.columns([2, 2, 3])
    w1.download_button("Download blank template", dataio.template_workbook(), file_name="demand_planner_template.xlsx",
                       mime=XLSX, key="tpl_book")
    if not D["items"].empty:
        w2.download_button("Download what is loaded now", dataio.current_data_workbook(D),
                           file_name=f"demand_planner_data_{dt.date.today():%Y-%m-%d}.xlsx", mime=XLSX, key="cur_book",
                           help="Edit it in Excel and load it back here.")
    book = st.file_uploader("Workbook", type=["xlsx", "xlsm"], key="up_book")
    if book is not None:
        try:
            raw, other = dataio.read_workbook(book)
        except Exception as e:
            st.error(f"Could not read this workbook. {e}")
            raw, other = {}, []
        if other:
            st.caption("Sheets not recognised and ignored: " + ", ".join(other))
        if raw:
            ship_mode = "Replace everything"
            if "shipments" in raw:
                ship_mode = st.radio("Shipments in the workbook", ["Replace the months in the file, keep the rest",
                                                                   "Replace everything"], key="book_ship_mode")
            # Check in load order, so rules are checked against the masters in the same workbook.
            ref = dict(D)
            ready, summary, failed = {}, [], False
            for t in [t for t in ORDER if t in raw]:
                try:
                    clean, issues = dataio.normalize(t, raw[t])
                    if t in dataio.RULE_KEYS:
                        clean, warn = dataio.check_references(t, clean, ref)
                        issues += warn
                    ready[t] = clean
                    if t in ("items", "customers"):
                        ref[t] = clean
                    summary.append({"Sheet": dataio.SPEC[t][2], "Rows ready": len(clean),
                                    "Notes": " ".join(issues) or "OK"})
                except Exception as e:
                    failed = True
                    summary.append({"Sheet": dataio.SPEC[t][2], "Rows ready": 0, "Notes": f"Not loadable: {e}"})
            st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch",
                         column_config={"Notes": st.column_config.TextColumn(width="large")})
            if failed:
                st.warning("Fix the sheets marked Not loadable, or delete them from the workbook, then upload it again. "
                           "Nothing has been loaded.")
            elif st.button(f"Load {len(ready)} sheet{'s' if len(ready) > 1 else ''}", type="primary", key="load_book"):
                for t in ORDER:
                    if t in ready:
                        load_table(t, ready[t], ship_mode)
                st.session_state["flash"] = ("Loaded: " + ", ".join(f"{dataio.SPEC[t][1].lower()} {len(ready[t]):,}" for t in ORDER if t in ready)
                                             + ". Run the forecast to use it.")
                st.rerun()
        elif not other:
            st.error("No recognised sheets. Use the sheet names from the template.")

    st.subheader("Load one table at a time")
    st.write("CSV or Excel, first row as headers. Load the masters before shipments and rules. "
             "Daily shipment lines are summed to months. Common header names are recognised.")
    for tab, (table, title, sheet, req, opt) in zip(st.tabs([s[1] for s in dataio.SPECS]), dataio.SPECS):
        with tab:
            st.caption(f"Required: {req}." + (f" Optional: {opt}." if opt else ""))
            st.download_button("Download template", dataio.template_csv(table), file_name=f"{table}_template.csv",
                               mime="text/csv", key=f"tpl_{table}")
            mode = "Replace everything"
            if table == "shipments":
                mode = st.radio("How to load", ["Replace the months in the file, keep the rest", "Replace everything"],
                                key="ship_mode")
            elif table in dataio.RULE_KEYS:
                st.caption("Loading replaces every row of this table.")
            up = st.file_uploader(f"{title} file", type=["csv", "xlsx", "xls", "txt"], key=f"up_{table}")
            if up is not None:
                try:
                    clean, issues = dataio.normalize(table, dataio.read_upload(up))
                    if table in dataio.RULE_KEYS:
                        clean, warn = dataio.check_references(table, clean, D)
                        issues += warn
                except Exception as e:  # show the reason, load nothing
                    st.error(f"Could not read this file. {e}")
                else:
                    st.write(f"{len(clean):,} rows ready to load.")
                    for i in issues:
                        st.warning(i)
                    st.dataframe(clean.head(20), hide_index=True, width="stretch")
                    if st.button(f"Load {title.lower()}", type="primary", key=f"load_{table}"):
                        load_table(table, clean, mode)
                        st.session_state["flash"] = f"{title} loaded: {len(clean):,} rows. Run the forecast to use it."
                        st.rerun()

    if res is not None and len(res.exceptions):
        st.subheader("Exceptions from the last run")
        st.caption("Shipments that did not match master data were left out of history. Fix the master data and run again.")
        st.dataframe(res.exceptions.rename(columns=str.capitalize), hide_index=True, width="stretch")

    log = db.get_setting("run_log", [])
    if log:
        with st.expander("Run history"):
            st.dataframe(pd.DataFrame([{"When": r["when"], "Series": r["series"], "Forecast units": round(r["forecast_units"]),
                                        "Horizon": r["config"]["horizon"], "Seasonal": r["config"]["seasonal"]}
                                       for r in reversed(log)]), hide_index=True)

    st.subheader("Sample data and reset")
    if demo.available():
        st.write("Replace everything with the demo dataset: M5 Walmart food and household products (1,000 products, "
                 "10 stores as customers, 3 state DCs, Feb 2021 to May 2026) with a generated lifecycle layer.")
        sure_d = st.checkbox("Yes, replace what is loaded with the demo dataset", key="sure_demo", value=D["items"].empty)
        if st.button("Load demo dataset", disabled=not sure_d):
            load_demo()
            st.rerun()
    s1, s2 = st.columns(2)
    with s1:
        st.write("Replace everything with the sample range (6 base SKUs, 8 customers, 3 locations, 36 months).")
        sure = st.checkbox("Yes, replace what is loaded", key="sure_sample", value=D["items"].empty)
        if st.button("Load sample data", disabled=not sure):
            load_sample()
            st.rerun()
    with s2:
        st.write("Delete all data, rules, settings and results.")
        sure2 = st.checkbox("Yes, delete everything", key="sure_reset")
        if st.button("Clear the database", disabled=not sure2):
            db.reset()
            st.cache_data.clear()
            st.rerun()
