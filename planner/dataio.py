"""Reading uploaded files, validating them, and building exports."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .db import SCHEMA
from .engine import ALL, customer_groups

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
    "version_transitions": {
        "from_version": ["from_version", "from", "from_sku", "old_version", "old_sku", "old_item"],
        "to_version": ["to_version", "to", "to_sku", "new_version", "new_sku", "new_item"],
        "customer_group": ["customer_group", "group", "customer_scope", "scope", "customer", "planning_group"],
        "effective_month": ["effective_month", "effective_date", "cutover_month", "cutover_date", "effective", "date", "month"],
    },
    "hard_transitions": {
        "from_base": ["from_base", "from_base_sku", "from", "old_base", "predecessor"],
        "to_base": ["to_base", "to_base_sku", "to", "new_base", "successor"],
        "effective_month": ["effective_month", "effective_date", "cutover_month", "cutover_date", "effective", "date", "month"],
        "scale": ["scale", "volume_factor", "factor", "scaling_factor"],
    },
    "new_products": {
        "base_sku": ["base_sku", "new_base", "new_base_sku", "new_item", "base"],
        "like_base": ["like_base", "like_item", "like_base_sku", "like", "reference_item"],
        "launch_month": ["launch_month", "launch_date", "launch", "start_date", "date"],
        "scale": ["scale", "volume_factor", "factor", "scaling_factor"],
        "ramp": ["ramp", "ramp_curve", "ramp_up"],
    },
    "split_settings": {
        "base_sku": ["base_sku", "base"],
        "method": ["method", "split_method", "split"],
    },
    "manual_split": {
        "base_sku": ["base_sku", "base"],
        "version_sku": ["version_sku", "version", "item", "sku"],
        "pct": ["pct", "percent", "percentage", "share", "split_pct"],
    },
}
REQUIRED = {
    "items": ["version_sku", "base_sku"],
    "customers": ["account"],
    "locations": ["location"],
    "shipments": ["ship_date", "version_sku", "account", "location", "qty"],
    "version_transitions": ["from_version", "to_version", "effective_month"],
    "hard_transitions": ["from_base", "to_base", "effective_month"],
    "new_products": ["base_sku", "like_base", "launch_month"],
    "split_settings": ["base_sku", "method"],
    "manual_split": ["version_sku", "pct"],
}
RULE_KEYS = {
    "version_transitions": ["from_version", "to_version", "customer_group"],
    "hard_transitions": ["from_base"],
    "new_products": ["base_sku"],
    "split_settings": ["base_sku"],
    "manual_split": ["base_sku", "version_sku"],
}

# What each table is, in the order it should be loaded. (table, title, sheet name, required, optional)
SPECS = [
    ("items", "Item master", "Items", "version_sku, base_sku",
     "description, base_description, version_seq, family, forecastable (Y/N), status, launch_date, discontinue_date"),
    ("customers", "Customer master", "Customers", "account", "name, channel, key_customer (Y/N), customer_group"),
    ("locations", "Location master", "Locations", "location", "name, type, region"),
    ("shipments", "Shipments", "Shipments", "ship_date, version_sku, account, location, quantity", ""),
    ("version_transitions", "Version transitions", "Version transitions", "from_version, to_version, effective_month",
     "customer_group (a customer group name, or ALL; blank means ALL)"),
    ("hard_transitions", "Hard transitions", "Hard transitions", "from_base, to_base, effective_month",
     "scale (volume factor, default 1)"),
    ("new_products", "New products", "New products", "base_sku, like_base, launch_month",
     "scale (default 1), ramp (for example 0.3,0.6,0.85)"),
    ("split_settings", "Version split method", "Split settings", "base_sku, method (recent, equal or manual)", ""),
    ("manual_split", "Manual split percentages", "Manual split", "version_sku, pct",
     "base_sku (filled from the item master when blank)"),
]
SPEC = {s[0]: s for s in SPECS}

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
    d = pd.to_datetime(s, errors="coerce", format="mixed")
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
    df = _clean_keys(df, [c for c in REQUIRED[table]
                          if c not in ("qty", "ship_date", "pct", "effective_month", "launch_month")])

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

    if table in RULE_KEYS:
        return _normalize_rule(table, df, issues)

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


def _ramp(v) -> str:
    vals = []
    for x in str(v if v is not None and not (isinstance(v, float) and np.isnan(v)) else "").replace(";", ",").replace(" ", ",").split(","):
        x = x.strip().rstrip("%")
        if not x:
            continue
        try:
            f = float(x)
        except ValueError:
            continue
        vals.append(f / 100 if f > 1.5 else f)
    return ",".join(f"{f:g}" for f in vals)


def _normalize_rule(table: str, df: pd.DataFrame, issues: list[str]) -> tuple[pd.DataFrame, list[str]]:
    n0 = len(df)
    df = df.copy()
    for mcol in ("effective_month", "launch_month"):
        if mcol in df.columns:
            raw = df[mcol]
            df[mcol] = _month(raw)
            bad = df[mcol].isna()
            if bad.any():
                issues.append(f"{int(bad.sum()):,} rows skipped: {mcol.replace('_', ' ')} is blank or not a date.")
                df = df[~bad]
    for c in ("scale",):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0)
        elif table in ("hard_transitions", "new_products"):
            df[c] = 1.0
    if table == "version_transitions":
        g = df["customer_group"].fillna("").astype(str).str.strip() if "customer_group" in df.columns else pd.Series("", index=df.index)
        df["customer_group"] = g.where(~g.str.lower().isin(["", "all", "*", "all customers", "everyone", "all other"]), ALL)
    if table == "new_products":
        df["ramp"] = df["ramp"].map(_ramp) if "ramp" in df.columns else ""
    if table == "split_settings":
        m = df["method"].fillna("").astype(str).str.lower()
        df["method"] = np.select([m.str.contains("manual"), m.str.contains("equal"),
                                  m.str.contains("recent|mix|hist")], ["manual", "equal", "recent"], "")
        bad = df["method"] == ""
        if bad.any():
            issues.append(f"{int(bad.sum()):,} rows skipped: method must be recent, equal or manual.")
            df = df[~bad]
    if table == "manual_split":
        df["pct"] = pd.to_numeric(df["pct"].astype(str).str.rstrip("%").str.replace(",", ""), errors="coerce")
        bad = df["pct"].isna() | (df["pct"] < 0)
        if bad.any():
            issues.append(f"{int(bad.sum()):,} rows skipped: percentage missing or negative.")
            df = df[~bad]
        if "base_sku" not in df.columns:
            df["base_sku"] = ""
        df["base_sku"] = df["base_sku"].fillna("").astype(str).str.strip()
    blank_cols = [c for c in REQUIRED[table] if c in df.columns and df[c].dtype == object]
    blank = pd.Series(False, index=df.index)
    for c in blank_cols:
        blank |= df[c].fillna("").astype(str).str.strip() == ""
    if blank.any():
        issues.append(f"{int(blank.sum()):,} rows skipped: a required value is blank.")
        df = df[~blank]
    keys = [k for k in RULE_KEYS[table] if k in df.columns]
    dup = df.duplicated(keys, keep="last")
    if dup.any():
        issues.append(f"{int(dup.sum()):,} duplicate rows skipped (last one kept).")
        df = df[~dup]
    for c in SCHEMA[table]:
        if c not in df.columns:
            df[c] = None
    if n0 and not len(df):
        raise ValueError("No usable rows in the file.")
    return df[SCHEMA[table]].reset_index(drop=True), issues


def check_references(table: str, df: pd.DataFrame, current: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Warn about codes the masters do not know. Fills manual_split.base_sku from the item master."""
    items = current.get("items", pd.DataFrame(columns=SCHEMA["items"]))
    versions = set(items["version_sku"].astype(str))
    bases = set(items["base_sku"].dropna().astype(str))
    v2b = dict(zip(items["version_sku"].astype(str), items["base_sku"].astype(str)))
    out, warn = df.copy(), []

    def unknown(col, known, what):
        if not known or col not in out.columns:
            return
        miss = sorted(set(out[col].astype(str)) - known)
        if miss:
            warn.append(f"{len(miss)} {what} not in the item master yet: {', '.join(miss[:5])}{'...' if len(miss) > 5 else ''}. "
                        "Those rows load but are ignored by the forecast until the item exists.")

    if table == "manual_split":
        fill = out["base_sku"].fillna("").astype(str).str.strip() == ""
        out.loc[fill, "base_sku"] = out.loc[fill, "version_sku"].map(v2b)
        still = out["base_sku"].isna() | (out["base_sku"].astype(str) == "")
        if still.any():
            warn.append(f"{int(still.sum()):,} rows skipped: no base SKU given and the version is not in the item master.")
            out = out[~still]
        unknown("version_sku", versions, "versions")
    elif table == "version_transitions":
        unknown("from_version", versions, "versions")
        unknown("to_version", versions, "versions")
        cust = current.get("customers")
        if cust is not None and len(cust):
            groups = set(customer_groups(cust)) | {ALL}
            miss = sorted(set(out["customer_group"]) - groups)
            if miss:
                warn.append(f"Customer group not found: {', '.join(miss[:5])}. Use a key customer's name or customer group, "
                            "All other, or ALL.")
    elif table in ("hard_transitions",):
        unknown("from_base", bases, "base SKUs")
        unknown("to_base", bases, "base SKUs")
    elif table == "new_products":
        unknown("base_sku", bases, "base SKUs")
        unknown("like_base", bases, "like items")
    elif table == "split_settings":
        unknown("base_sku", bases, "base SKUs")
    return out.reset_index(drop=True), warn


def read_workbook(file) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read every recognised sheet. Returns ({table: raw frame}, names of sheets that were not recognised)."""
    xl = pd.ExcelFile(file)
    norm = lambda x: str(x).strip().lower().replace(" ", "_").replace("-", "_")  # noqa: E731
    lookup = {}
    for table, title, sheet, _, _ in SPECS:
        for n in (table, title, sheet):
            lookup[norm(n)] = table
    lookup.update({"item_master": "items", "customer_master": "customers", "location_master": "locations",
                   "split_method": "split_settings", "manual_split_percentages": "manual_split"})
    found, other = {}, []
    for name in xl.sheet_names:
        t = lookup.get(norm(name))
        if t and t not in found:
            found[t] = xl.parse(name, dtype=str)
        elif norm(name) not in ("instructions", "read_me", "readme"):
            other.append(name)
    return found, other


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


TEMPLATES.update({
    "version_transitions": pd.DataFrame([["B200-V2", "B200-V3", "Northgate Stores", "2026-12"],
                                         ["B200-V2", "B200-V3", "ALL", "2027-03"]],
                                        columns=["from_version", "to_version", "customer_group", "effective_month"]),
    "hard_transitions": pd.DataFrame([["B300", "B350", "2027-02", 1.1]],
                                     columns=["from_base", "to_base", "effective_month", "scale"]),
    "new_products": pd.DataFrame([["B500", "B400", "2027-01", 0.4, "0.3,0.6,0.85"]],
                                 columns=["base_sku", "like_base", "launch_month", "scale", "ramp"]),
    "split_settings": pd.DataFrame([["B100", "recent"], ["B200", "manual"]], columns=["base_sku", "method"]),
    "manual_split": pd.DataFrame([["B200", "B200-V2", 40], ["B200", "B200-V3", 60]], columns=["base_sku", "version_sku", "pct"]),
})


def template_csv(table: str) -> bytes:
    return TEMPLATES[table].to_csv(index=False).encode()


def _write_book(frames: dict[str, pd.DataFrame], with_instructions: bool) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        if with_instructions:
            pd.DataFrame([{"Sheet": sheet, "What it holds": title, "Required columns": req, "Optional columns": opt}
                          for _, title, sheet, req, opt in SPECS]
                         + [{"Sheet": "", "What it holds": "Notes", "Required columns":
                             "Keep the sheet names. Delete a sheet to leave that table as it is in the app. "
                             "Months can be any date in the month or YYYY-MM. Shipments over about a million rows: use a CSV file instead.",
                             "Optional columns": ""}]).to_excel(xl, sheet_name="Instructions", index=False)
        for table, _, sheet, _, _ in SPECS:
            if table in frames:
                frames[table].to_excel(xl, sheet_name=sheet, index=False)
        for ws in xl.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in list(col)[:200]) + 2
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width))
    return buf.getvalue()


def template_workbook() -> bytes:
    """Every input table in one workbook, with example rows and an Instructions sheet."""
    return _write_book(TEMPLATES, with_instructions=True)


def current_data_workbook(current: dict[str, pd.DataFrame]) -> bytes:
    """What is loaded now, in the template layout, so it can be edited in Excel and loaded back."""
    frames = {}
    for t, df in current.items():
        if t not in SPEC:
            continue
        out = df.copy()
        if t == "items":
            out = out.rename(columns={"launch_month": "launch_date", "discontinue_month": "discontinue_date"})
            out["forecastable"] = np.where(out["forecastable"].astype(bool), "Y", "N")
        elif t == "customers":
            out["key_customer"] = np.where(out["key_customer"].astype(bool), "Y", "N")
        elif t == "shipments":
            out = out.rename(columns={"month": "ship_date", "qty": "quantity"})
        frames[t] = out
    return _write_book(frames, with_instructions=True)


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
