"""Backtest accuracy at different levels.

Each series (base SKU x customer group x location) was forecast for its holdout months
without seeing them. To measure a coarser level, forecasts and actuals are first added
together to that level for each month, then compared, so over- and under-forecasts in the
pieces cancel the way they do in a report at that level.

    WAPE = sum over rows of |forecast - actual| / sum of actual
    bias = sum over rows of (forecast - actual) / sum of actual
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = {
    "Base SKU x customer group x location": ["root_base", "customer_group", "location"],
    "Base SKU x customer group": ["root_base", "customer_group"],
    "Base SKU": ["root_base"],
    "Customer group (all items)": ["customer_group"],
}


def measure(bt: pd.DataFrame, keys: list[str]) -> dict:
    """WAPE and bias after adding the backtest to `keys` x month."""
    if bt is None or bt.empty:
        return dict(wape=np.nan, bias=np.nan, act=0.0, rows=0)
    g = bt.groupby(keys + ["month"])[["fc", "act"]].sum()
    act = g["act"].sum()
    if act <= 0:
        return dict(wape=np.nan, bias=np.nan, act=0.0, rows=len(g))
    return dict(wape=float((g["fc"] - g["act"]).abs().sum() / act * 100),
                bias=float((g["fc"] - g["act"]).sum() / act * 100), act=float(act), rows=len(g))


def by_level(bt: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dict(level=name, **measure(bt, keys)) for name, keys in LEVELS.items()])


def by_key(bt: pd.DataFrame, report: list[str], level: list[str]) -> pd.DataFrame:
    """WAPE, bias and units per value of `report`, each measured at `level` (report must be part of level).

    One pass: add to level x month, take errors, then add errors up per report key.
    """
    g = bt.groupby(level + ["month"])[["fc", "act"]].sum()
    g["abs"] = (g["fc"] - g["act"]).abs()
    g["err"] = g["fc"] - g["act"]
    t = g.groupby(level=report)[["abs", "err", "act"]].sum()
    out = pd.DataFrame({"wape": t["abs"] / t["act"] * 100, "bias": t["err"] / t["act"] * 100, "act": t["act"]})
    return out.where(t["act"] > 0)
