"""Demo dataset: M5 (Walmart) products with a generated product lifecycle layer. See DEMO_DATA.md."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import db

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "m5"
TABLES = ["items", "customers", "locations", "shipments", "version_transitions", "hard_transitions", "new_products"]
NOTE = ("Demo data: M5 Forecasting dataset (Walmart, CC BY 4.0) with a generated product lifecycle layer. "
        "Dates are moved forward 10 years.")


def is_demo_mode() -> bool:
    """True when the app was started with the demo launcher (or on a database named demo*)."""
    return os.environ.get("PLANNER_DEMO") == "1" or db.DB_PATH.stem.lower().startswith("demo")


def available() -> bool:
    return all((DEMO_DIR / f"{t}.parquet").exists() for t in TABLES)


def load() -> dict[str, int]:
    """Replace everything in the current database with the demo dataset. Returns row counts."""
    db.reset()
    counts = {}
    for t in TABLES:
        df = pd.read_parquet(DEMO_DIR / f"{t}.parquet")
        db.write(t, df)
        counts[t] = len(df)
    db.write("split_settings", pd.DataFrame(columns=db.SCHEMA["split_settings"]))
    db.write("manual_split", pd.DataFrame(columns=db.SCHEMA["manual_split"]))
    # ARIMA off keeps a full demo run to seconds; it can be switched on in Settings.
    db.set_config({**db.get_config(), "arima": False, "horizon": 18})
    db.set_setting("dataset", "demo-m5")
    return counts
