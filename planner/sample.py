"""Sample data: a small CPG range with version changes, a staggered transition,
a hard transition and a new product. History ends last month."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .engine import idx_to_month

H = 36


def build(today: dt.date | None = None) -> dict[str, pd.DataFrame]:
    today = today or dt.date.today()
    now = today.year * 12 + today.month - 1  # first forecast month
    t0 = now - H
    m = lambda t: None if t is None else idx_to_month(t0 + t)  # noqa: E731

    locations = pd.DataFrame([["DC-E", "Eastern DC", "Distribution center", "East", .40],
                              ["DC-C", "Central DC", "Distribution center", "Central", .35],
                              ["DC-W", "Western DC", "Third-party warehouse", "West", .25]],
                             columns=["location", "name", "type", "region", "share"])
    customers = pd.DataFrame([["C01", "Northgate Stores", "Mass", True, .28], ["C02", "Bellweather Club", "Club", True, .20],
                              ["C03", "Corvin Drug", "Drug", True, .14], ["C04", "Shoply Online", "E-commerce", True, .18],
                              ["C05", "Harbor Hardware", "Hardware", False, .07], ["C06", "Pine & Co", "Specialty", False, .05],
                              ["C07", "Midtown Grocers", "Grocery", False, .05], ["C08", "Alder Home", "Specialty", False, .03]],
                             columns=["account", "name", "channel", "key_customer", "share"])
    customers["customer_group"] = None
    # base: name, family, level, peak calendar month (0 = Jan), amplitude, yearly trend
    bases = {"B100": ("Tower Fan 40 in", "Fans", 4200, 5, .75, .03),
             "B200": ("Ceramic Heater 1500 W", "Heaters", 3000, 11, .80, .01),
             "B300": ("Cool Mist Humidifier 1 gal", "Humidifiers", 2200, 0, .60, -.02),
             "B350": ("Ultrasonic Humidifier 1.5 gal", "Humidifiers", None, 0, 0, 0),
             "B400": ("Insulated Bottle 24 oz", "Hydration", 5000, 6, .20, .08),
             "B500": ("Insulated Tumbler 30 oz", "Hydration", None, 0, 0, 0)}
    # version, base, seq, launch t, discontinue t, forecastable, has history
    vers = [("B100-V1", "B100", 1, None, 10, False, True), ("B100-V2", "B100", 2, 8, 22, False, True),
            ("B100-V3", "B100", 3, 20, None, True, True), ("B100-V4", "B100", 4, 31, None, True, True),
            ("B200-V1", "B200", 1, None, 14, False, True), ("B200-V2", "B200", 2, 12, None, True, True),
            ("B200-V3", "B200", 3, None, None, True, False),
            ("B300-V1", "B300", 1, None, 18, False, True), ("B300-V2", "B300", 2, 16, None, True, True),
            ("B350-V1", "B350", 1, None, None, True, False),
            ("B400-V1", "B400", 1, None, 26, False, True), ("B400-V2", "B400", 2, 24, None, True, True),
            ("B500-V1", "B500", 1, None, None, True, False)]
    items = pd.DataFrame([dict(version_sku=v, description=f"{bases[b][0]} v{s}", base_sku=b, base_description=bases[b][0],
                               version_seq=s, family=bases[b][1], forecastable=fc, launch_month=m(la), discontinue_month=m(di))
                          for v, b, s, la, di, fc, _ in vers])

    rng = np.random.default_rng(20261011)
    rows = []
    for b, (_, _, level, peak, amp, trend) in bases.items():
        if not level:
            continue
        vs = [v for v in vers if v[1] == b and v[6]]
        for c in customers.itertuples():
            for l in locations.itertuples():
                tilt = rng.uniform(.75, 1.25)
                for t in range(H):
                    cal = (t0 + t) % 12
                    season = 1 + amp * np.cos(2 * np.pi * (cal - peak) / 12)
                    q = (level * (1 + trend) ** (t / 12) * season * c.share * l.share * tilt
                         * np.exp(rng.normal(0, .12 if c.key_customer else .32)))
                    w = np.array([(1 if la is None else min(1, max(0, (t - la + 1) / 3)))
                                  * (1 if di is None else min(1, max(0, (di - t) / 3))) for _, _, _, la, di, _, _ in vs])
                    for v, share in zip(vs, w / (w.sum() or 1)):
                        if round(q * share) > 0:
                            rows.append((m(t), v[0], c.account, l.location, float(round(q * share))))
    shipments = pd.DataFrame(rows, columns=["month", "version_sku", "account", "location", "qty"])

    return dict(
        items=items, customers=customers.drop(columns="share"), locations=locations.drop(columns="share"),
        shipments=shipments,
        version_transitions=pd.DataFrame([["B200-V2", "B200-V3", "Northgate Stores", m(H + 2)],
                                          ["B200-V2", "B200-V3", "ALL", m(H + 5)]],
                                         columns=["from_version", "to_version", "customer_group", "effective_month"]),
        hard_transitions=pd.DataFrame([["B300", "B350", m(H + 4), 1.1]],
                                      columns=["from_base", "to_base", "effective_month", "scale"]),
        new_products=pd.DataFrame([["B500", "B400", m(H + 3), 0.4, "0.3,0.6,0.85"]],
                                  columns=["base_sku", "like_base", "launch_month", "scale", "ramp"]),
        split_settings=pd.DataFrame(columns=["base_sku", "method"]),
        manual_split=pd.DataFrame(columns=["base_sku", "version_sku", "pct"]),
    )
