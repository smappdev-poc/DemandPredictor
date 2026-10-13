"""Reading uploaded files, validating them, and building exports."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .db import SCHEMA

# Accepted header names (lower-case, spaces and dashes become underscores) -> our column.
ALIASES = {
    "items": {
        "version_sku": ["version_sku", "item", "sku", "item_number", "version", "item_code"],
        "description": ["description", "item_description", "desc"],
        "base_sku": ["base_sku", "base", "planning_item", "base_item", "item_family"],
        "base_description": ["base_description", "base_desc", "planning_item_description"],
        "version_seq": ["version_seq", "seq", "version_sequence", "version_number"],
        "family": ["family", "product_family", "category"],
        "forecastable": ["forecastable", "forecast_flag", "active"],
        "status": ["status", "lifecycle_status", "item_status"],
        "launch_month": ["launch_month", "launch_date", "start_date", "intro_date"],
        "discontinue_month": ["discontinue_month", "discontinue_date", "end_date", "disc_date"],
    },
    "customers": {
        "account": ["account", "customer", "customer_account", "customer_number", "account_number"],
        "name": ["name", "customer_name", "account_name"],
        "channel": ["channel", "sales_channel"],
        "key_customer": ["key_customer", "key", "key_flag", "is_key"],
        "customer_group": ["customer_group", "group", "planning_account", "planning_group"],
    },
    "locations": {
        "location": ["location", "location_code", "ship_from", "warehouse", "org", "organization_code"],
        "name": ["name", "location_name"],
        "type": ["type", "location_type"],
        "region": ["region"],
    },
    "shipments": {
        "ship_date": ["ship_date", "date", "month", "shipment_date", "ship_month", "period"],
        "version_sku": ["version_sku", "item", "sku", "item_number", "item_code"],
        "account": ["account", "customer", "customer_account", "customer_number", "account_number"],
        "location": ["location", "location_code", "ship_from", "warehouse", "org", "organization_code"],
        "qty": ["qty", "quantity", "shipped_quantity", "shipped_qty", "units"],
    },
}
REQUIRED = {
    "items": ["version_sku", "base_sku"],
    "customers": ["account"],
    "locations": ["location"],
    "shipments": ["ship_date", "version_sku", "account", "location", "qty"],
}
TRUE = {"y", "yes", "true", "1", "t", "x", "key", "active"}


def read_upload(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file, dtype=str)
    return pd.read_csv(file, dtype=str, sep=None, engine="python", encoding_errors="replace")


def _rename(df: pd.DataFrame, table: str) -> pd.DataFrame:
    norm = {c: str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    df = df.rename(columns=norm)
    out = {}
    for target, names in ALIASES[table].items():
        for n in names:
            if n in df.columns:
                out[target] = df[n]
                break
    return pd.DataFrame(out)


def _flag(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().isin(TRUE)


def _month(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return d.dt.strftime("%Y-%m").where(d.notna(), None)


def _clean_keys(df, cols):
    for c in cols:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def normalize(table: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean frame in our schema, list of plain-language issues)."""
    issues: list[str] = []
    df = _rename(raw, table)
    missing = [c for c in REQUIRED[table] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. "
                         f"Found: {', '.join(map(str, raw.columns))}")
    n0 = len(df)
    df = _clean_keys(df, [c for c in REQUIRED[table] if c not in ("qty", "ship_date")])

    if table == "shipments":
        df["month"] = _month(df["ship_date"])
        df["qty"] = pd.to_numeric(df["qty"].astype(str).str.replace(",", ""), errors="coerce")
        bad = df["month"].isna() | df["qty"].isna() | (df["version_sku"] == "") | (df["account"] == "") | (df["location"] == "")
        if bad.any():
            issues.append(f"{int(bad.sum()):,} rows skipped: unreadable date or quantity, or a blank item, customer or location.")
        df = df[~bad]
        out = df.groupby(["month", "version_sku", "account", "location"], as_index=False)["qty"].sum()
        if len(out) < len(df):
            issues.append(f"{len(df):,} rows summed to {len(out):,} monthly rows.")
        return out[SCHEMA["shipments"]], issues

    key = REQUIRED[table][0]
    blank = df[key] == ""
    if blank.any():
        issues.append(f"{int(blank.sum()):,} rows skipped: blank {key}.")
    df = df[~blank]
    dup = df.duplicated(key, keep="first")
    if dup.any():
        issues.append(f"{int(dup.sum()):,} duplicate {key} rows skipped (first one kept): "
                      + ", ".join(df.loc[dup, key].head(5)) + ("..." if dup.sum() > 5 else ""))
    df = df[~dup].copy()

    if table == "items":
        no_base = df["base_sku"] == ""
        if no_base.any():
            issues.append(f"{int(no_base.sum()):,} items have no base SKU and will be held out of the forecast.")
        for c in ("launch_month", "discontinue_month"):
            df[c] = _month(df[c]) if c in df.columns else None
        status = df["status"].fillna("").astype(str).str.lower() if "status" in df.columns else pd.Series("", index=df.index)
        discontinued = status.str.contains("disc|inactive|obsolete|phase.?out|end")
        if "forecastable" in df.columns and df["forecastable"].fillna("").astype(str).str.strip().ne("").any():
            df["forecastable"] = _flag(df["forecastable"])
        else:
            df["forecastable"] = ~discontinued
            issues.append("No forecastable column: every item is forecastable unless its status says discontinued.")
        if "version_seq" in df.columns:
            df["version_seq"] = pd.to_numeric(df["version_seq"], errors="coerce")
        else:
            df["version_seq"] = np.nan
        # Fill missing sequence by launch month, then item code, within each base SKU.
        order = df.sort_values(["base_sku", "launch_month", "version_sku"], na_position="first")
        auto = order.groupby("base_sku").cumcount() + 1
        df["version_seq"] = df["version_seq"].fillna(auto.reindex(df.index)).astype("Int64")
    elif table == "customers":
        if "key_customer" in df.columns:
            df["key_customer"] = _flag(df["key_customer"])
        else:
            df["key_customer"] = False
            issues.append("No key_customer column: every customer is non-key until you flag them under Master data.")

    for c in SCHEMA[table]:
        if c not in df.columns:
            df[c] = None
    if n0 and not len(df):
        raise ValueError("No usable rows in the file.")
    return df[SCHEMA[table]].reset_index(drop=True), issues


TEMPLATES = {
    "items": pd.DataFrame([
        ["B100-V3", "Tower Fan 40 in v3", "B100", "Tower Fan 40 in", 3, "Fans", "Y", "2025-06-01", ""],
        ["B100-V2", "Tower Fan 40 in v2", "B100", "Tower Fan 40 in", 2, "Fans", "N", "2024-06-01", "2025-08-01"]],
        columns=["version_sku", "description", "base_sku", "base_description", "version_seq", "family",
                 "forecastable", "launch_date", "discontinue_date"]),
    "customers": pd.DataFrame([["C01", "Northgate Stores", "Mass", "Y", ""], ["C05", "Harbor Hardware", "Hardware", "N", ""]],
                              columns=["account", "name", "channel", "key_customer", "customer_group"]),
    "locations": pd.DataFrame([["DC-E", "Eastern DC", "Distribution center", "East"]],
                              columns=["location", "name", "type", "region"]),
    "shipments": pd.DataFrame([["2026-09-14", "B100-V3", "C01", "DC-E", 480], ["2026-09-21", "B100-V3", "C05", "DC-E", 36]],
                              columns=["ship_date", "version_sku", "account", "location", "quantity"]),
}


def template_csv(table: str) -> bytes:
    return TEMPLATES[table].to_csv(index=False).encode()


def export_excel(res, items: pd.DataFrame) -> bytes:
    """Workbook with base SKU and version forecasts (months across), accuracy and exceptions."""
    desc = items.set_index("version_sku")["description"].to_dict()

    def wide(df, keys):
        if df.empty:
            return pd.DataFrame(columns=keys + res.fc_months)
        w = df.pivot_table(index=keys, columns="month", values="qty", aggfunc="sum", fill_value=0)
        w = w.reindex(columns=res.fc_months, fill_value=0)
        w["Total"] = w.sum(axis=1)
        return w.reset_index()

    ver = wide(res.version_fc, ["base_sku", "version_sku", "customer_group", "location"])
    if len(ver):
        ver.insert(2, "description", ver["version_sku"].map(desc))
    acc = res.series.copy()
    acc["wape_pct"] = np.where(acc["act"] > 0, acc["abs_err"] / acc["act"] * 100, np.nan).round(1)
    acc["bias_pct"] = np.where(acc["act"] > 0, acc["err"] / acc["act"] * 100, np.nan).round(1)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        wide(res.base_fc, ["base_sku", "customer_group", "location"]).to_excel(xl, sheet_name="Base SKU forecast", index=False)
        ver.to_excel(xl, sheet_name="Version forecast", index=False)
        wide(res.unallocated, ["base_sku", "customer_group", "location"]).to_excel(xl, sheet_name="Unallocated", index=False)
        acc.drop(columns=["abs_err", "err", "act"]).to_excel(xl, sheet_name="Models and accuracy", index=False)
        res.exceptions.to_excel(xl, sheet_name="Exceptions", index=False)
        for ws in xl.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(40, max(10, len(str(col[0].value or "")) + 2))
    return buf.getvalue()
