"""Forecast breakdown by account.

The engine forecasts per customer group: each key customer is its own group, and
non-key customers share an "All other" group. To show the forecast per account,
each group's forecast is shared across the accounts in it by their share of
shipments over the last 12 months, in integers that add back exactly.

Shares are looked for at the finest level that has shipments:
  1. same base SKU, customer group and location
  2. same base SKU and customer group, any location
  3. same customer group and location, any base SKU
  4. same customer group, anywhere
  5. equal across the group's active accounts
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import customer_groups


def account_history(shipments: pd.DataFrame, items: pd.DataFrame, customers: pd.DataFrame,
                    hist_months: list[str], nonkey_by_channel: bool = False) -> pd.DataFrame:
    """Shipments mapped to base SKU and customer group, kept at account level, for the run's history months."""
    v2b = dict(zip(items["version_sku"].astype(str), items["base_sku"].fillna("").astype(str)))
    grp = customer_groups(customers, nonkey_by_channel).to_dict() if len(customers) else {}
    h = shipments[shipments["month"].isin(hist_months)].copy()
    h["base_sku"] = h["version_sku"].astype(str).map(v2b)
    h["customer_group"] = h["account"].astype(str).map(grp)
    h = h[h["base_sku"].fillna("").ne("") & h["customer_group"].notna()]
    return h[["version_sku", "base_sku", "customer_group", "location", "account", "month", "qty"]]


def _shares(hist_acct: pd.DataFrame, customers: pd.DataFrame, nonkey_by_channel: bool, window_months: list[str]):
    recent = hist_acct[hist_acct["month"].isin(window_months)]
    recent = recent[recent["qty"] > 0]
    levels = [["base_sku", "customer_group", "location"], ["base_sku", "customer_group"],
              ["customer_group", "location"], ["customer_group"]]
    out = []
    for keys in levels:
        t = recent.groupby(keys + ["account"])["qty"].sum()
        out.append((keys, (t / t.groupby(level=keys).transform("sum")).rename("w").reset_index()))
    grp = customer_groups(customers, nonkey_by_channel)
    members = pd.DataFrame({"account": grp.index.astype(str), "customer_group": grp.values})
    if "active" in customers.columns:  # not in the schema today; kept for a future active flag
        members = members[customers["active"].astype(bool).values]
    eq = members.assign(w=1.0)
    eq["w"] = eq["w"] / eq.groupby("customer_group")["w"].transform("sum")
    return out, eq


def split_by_account(fc: pd.DataFrame, hist_acct: pd.DataFrame, customers: pd.DataFrame,
                     hist_months: list[str], nonkey_by_channel: bool = False, window: int = 12) -> pd.DataFrame:
    """fc: rows with base_sku, customer_group, location, month, qty (and optionally version_sku).

    Returns the same rows split by account, with integer quantities that add back to each input row.
    """
    if fc.empty:
        return fc.assign(account=pd.Series(dtype=str))
    levels, eq = _shares(hist_acct, customers, nonkey_by_channel, hist_months[-window:])
    fc = fc.reset_index(drop=True).copy()
    fc["_row"] = np.arange(len(fc))
    combos = fc[["base_sku", "customer_group", "location"]].drop_duplicates()
    pieces, done = [], pd.Series(False, index=combos.index)
    for keys, w in levels:
        todo = combos[~done]
        if todo.empty:
            break
        m = todo.reset_index().merge(w, on=keys, how="inner")
        if len(m):
            pieces.append(m[["base_sku", "customer_group", "location", "account", "w"]])
            done.loc[m["index"].unique()] = True
    todo = combos[~done]
    if len(todo):
        pieces.append(todo.merge(eq, on="customer_group", how="inner")[["base_sku", "customer_group", "location", "account", "w"]])
    weights = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["base_sku", "customer_group", "location", "account", "w"])

    x = fc.merge(weights, on=["base_sku", "customer_group", "location"], how="left")
    x["account"] = x["account"].fillna("(no account)")
    x["w"] = x["w"].fillna(1.0)
    # Integer split by largest remainder, per input row
    q = x["qty"].round().astype(np.int64)
    raw = q * x["w"]
    x["qty"] = np.floor(raw + 1e-9).astype(np.int64)
    x["_rem"] = raw - x["qty"]
    short = q.groupby(x["_row"]).transform("first") - x.groupby("_row")["qty"].transform("sum")
    x["_rank"] = x.groupby("_row")["_rem"].rank(method="first", ascending=False)
    x.loc[x["_rank"] <= short, "qty"] += 1
    x = x[x["qty"] != 0]
    return x.drop(columns=["_row", "_rem", "_rank", "w"]).reset_index(drop=True)
