"""SQLite storage. One file on disk holds inputs, rules, settings and the last forecast run."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

from .engine import DEFAULTS, Result

DB_PATH = Path(os.environ.get("PLANNER_DB", Path(__file__).resolve().parent.parent / "data" / "planner.db"))

# Input tables and their columns. Months are stored as 'YYYY-MM'.
SCHEMA = {
    "items": ["version_sku", "description", "base_sku", "base_description", "version_seq", "family",
              "forecastable", "launch_month", "discontinue_month"],
    "customers": ["account", "name", "channel", "key_customer", "customer_group"],
    "locations": ["location", "name", "type", "region"],
    "shipments": ["month", "version_sku", "account", "location", "qty"],
    "version_transitions": ["from_version", "to_version", "customer_group", "effective_month"],
    "hard_transitions": ["from_base", "to_base", "effective_month", "scale"],
    "new_products": ["base_sku", "like_base", "launch_month", "scale", "ramp"],
    "split_settings": ["base_sku", "method"],
    "manual_split": ["base_sku", "version_sku", "pct"],
}
RESULT_TABLES = ["hist", "base_fc", "version_fc", "unallocated", "series", "chains", "exceptions"]
NUMERIC = {"version_seq": "Int64", "qty": "float", "scale": "float", "pct": "float"}
BOOL = {"forecastable", "key_customer"}


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    return con


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in BOOL:
            df[c] = df[c].fillna(0).astype(int).astype(bool)
        elif c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(NUMERIC[c])
        elif c != "qty":
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def read(table: str) -> pd.DataFrame:
    with connect() as con:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", con)
        except Exception:
            df = pd.DataFrame(columns=SCHEMA.get(table, []))
    for c in SCHEMA.get(table, []):
        if c not in df.columns:
            df[c] = None
    if table in SCHEMA:
        df = _typed(df[SCHEMA[table]].copy())
    return df


def write(table: str, df: pd.DataFrame, touch: bool = True) -> None:
    out = df.copy()
    if table in SCHEMA:
        for c in SCHEMA[table]:
            if c not in out.columns:
                out[c] = None
        out = out[SCHEMA[table]]
        for c in BOOL & set(out.columns):
            out[c] = out[c].fillna(False).astype(bool).astype(int)
    with connect() as con:
        out.to_sql(table, con, if_exists="replace", index=False)
        if table == "shipments":
            con.execute("CREATE INDEX IF NOT EXISTS ix_ship_month ON shipments(month)")
    if touch and table in SCHEMA:
        bump_inputs()


def upsert_shipments(df: pd.DataFrame) -> int:
    """Replace the months present in df, keep every other month. Safe to re-run."""
    months = sorted(df["month"].unique())
    with connect() as con:
        exists = con.execute("SELECT name FROM sqlite_master WHERE name='shipments'").fetchone()
        if exists and months:
            con.executemany("DELETE FROM shipments WHERE month = ?", [(m,) for m in months])
        df[SCHEMA["shipments"]].to_sql("shipments", con, if_exists="append", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_ship_month ON shipments(month)")
    bump_inputs()
    return len(months)


def count(table: str) -> int:
    with connect() as con:
        try:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0


# ---- settings
def get_setting(key: str, default=None):
    with connect() as con:
        row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_setting(key: str, value) -> None:
    with connect() as con:
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)))


def get_config() -> dict:
    return {**DEFAULTS, **(get_setting("config", {}) or {})}


def set_config(cfg: dict) -> None:
    set_setting("config", cfg)
    bump_inputs()


def bump_inputs() -> None:
    set_setting("inputs_version", (get_setting("inputs_version", 0) or 0) + 1)


# ---- results of the last run
def save_result(res: Result, cfg: dict, when: str) -> None:
    for t in RESULT_TABLES:
        write("res_" + t, getattr(res, t), touch=False)
    set_setting("run_info", dict(when=when, config=cfg, hist_months=res.hist_months, fc_months=res.fc_months,
                                 inputs_version=get_setting("inputs_version", 0)))
    log = get_setting("run_log", []) or []
    log.append(dict(when=when, config=cfg, series=int(len(res.series)),
                    forecast_units=float(res.base_fc["qty"].sum())))
    set_setting("run_log", log[-50:])


def load_result() -> tuple[Result | None, dict | None]:
    info = get_setting("run_info")
    if not info:
        return None, None
    res = Result(hist_months=info["hist_months"], fc_months=info["fc_months"])
    with connect() as con:
        for t in RESULT_TABLES:
            setattr(res, t, pd.read_sql(f"SELECT * FROM res_{t}", con))
    return res, info


def is_stale() -> bool:
    info = get_setting("run_info")
    return bool(info) and info.get("inputs_version") != get_setting("inputs_version", 0)


def reset() -> None:
    with connect() as con:
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            con.execute(f"DROP TABLE IF EXISTS {name}")
