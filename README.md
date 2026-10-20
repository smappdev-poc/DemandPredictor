# Base SKU Demand Planner

A demand forecasting app that runs on your own laptop. Shipments from every version
of an item roll up to its base SKU, the forecast is made at base SKU, customer group
and ship-from location, and the result is split to the versions that are active in
each month.

Everything stays on your machine. Data is kept in one file, `data/planner.db`.

## Start it

You need Python 3.10 or newer (python.org). Internet access is needed the first
time only, to download the packages.

- **Windows:** double-click `run_windows.bat`.
- **Mac or Linux:** in a terminal, `sh run_mac_linux.sh`.

The first start takes a few minutes. After that the app opens in your browser at
http://localhost:8501. Keep the black window open while you work; close it to stop.

On first open, press **Load sample data** to see the app working, then load your own
files from the **Data** page.

## Demo copy

`run_windows.bat` opens your own data (port 8501, `data/planner.db`).
`run_demo.bat` opens a separate demo copy (port 8502, `data/demo.db`) built on public
Walmart data from the M5 forecasting competition, with versions, transitions and new
product launches added. Both can run at the same time and share the same code. On the
demo's first start press **Load demo dataset**. Source, licence and changes are in
`DEMO_DATA.md`.

## Your files

Everything the app uses can be loaded from the **Data** page, either all at once
from one Excel workbook or one table at a time from CSV or Excel.

- **One workbook:** download the blank template (it has an Instructions sheet), fill
  the sheets you need, and upload it. Sheets you delete are left as they are in the
  app. "Download what is loaded now" gives you the current data in the same layout,
  so you can edit it in Excel and load it back.
- **One table at a time:** each table has its own template and upload.

Common header names are recognised (for example `item`, `sku`, `customer`,
`warehouse`, `quantity`, `cutover date`). Dates can be any date in the month or
`YYYY-MM`.

| Table (sheet) | Required columns | Optional columns |
| --- | --- | --- |
| Items | `version_sku`, `base_sku` | `description`, `base_description`, `version_seq`, `family`, `forecastable` (Y/N), `status`, `launch_date`, `discontinue_date` |
| Customers | `account` | `name`, `channel`, `key_customer` (Y/N), `customer_group` |
| Locations | `location` | `name`, `type`, `region` |
| Shipments | `ship_date`, `version_sku`, `account`, `location`, `quantity` | |
| Version transitions | `from_version`, `to_version`, `effective_month` | `customer_group` (a group name, or ALL; blank means ALL) |
| Hard transitions | `from_base`, `to_base`, `effective_month` | `scale` (default 1) |
| New products | `base_sku`, `like_base`, `launch_month` | `scale` (default 1), `ramp` (e.g. `0.3,0.6,0.85` or `30%,60%`) |
| Split settings | `base_sku`, `method` (recent, equal or manual) | |
| Manual split | `version_sku`, `pct` | `base_sku` (filled from the item master when blank) |

- Shipment lines can be daily; they are summed to months.
- Shipments can be reloaded for a date range. Months in the file replace the same
  months in the database; other months are kept.
- Loading a master or rule table replaces that whole table.
- Before anything is loaded, each file is checked and you see what will be skipped
  and why: blank or duplicate keys, unreadable dates, and codes the item or customer
  master does not know.
- Shipments whose item, customer or location is not in the masters are left out of
  the forecast and listed under Exceptions on the Data page.
- Quantities are assumed to be in one unit of measure. For more than about a million
  shipment rows, use a CSV file rather than the workbook.

## Explore page

Profiles the shipments loaded now, before any forecast is run, at base SKU, base SKU x
customer group, or the full forecast level (base SKU x customer group x location).

- **Data health:** history range, volume and year-on-year change, a monthly total chart
  with a 12-month moving average, and data checks: unmatched codes, items with no base
  SKU, negative months, missing months, an incomplete latest month, versions flagged
  wrongly, rules naming unknown codes.
- **Suggested changes:** settings and master-data changes worked out from the data, such
  as turning on Ignore the latest month, a shorter holdout, a longer recent-mix window,
  pooling, key customer flags, inactive or steeply declining series.
- **Demand patterns:** each series placed by average demand interval (ADI) and CV² of
  non-zero quantities: smooth, erratic, intermittent or lumpy (cut-offs 1.32 and 0.49).
  ABC by last-12-month volume and XYZ by variability after seasonality.
- **Trend and seasonality:** volume by trend class (slope of the deseasonalised last 36
  months; Up or Down beyond 10% a year) and by peak month, split by seasonal strength.
- **Series profiles:** one row per series with every measure, filterable, downloadable.
- **One series:** monthly history with its trend line, outliers ringed, and its seasonal
  profile. An outlier is a month more than 3.5 robust standard deviations and 25% away
  from the three months either side, after seasonality.

## How the forecast works

1. **Roll-up.** Version to base SKU; customer to customer group. A key customer is
   its own group. Non-key customers are pooled as "All other" (or one pool per
   channel, in Settings).
2. **Models.** Each series tries exponential smoothing, seasonal naive, seasonal with
   damped trend, ARIMA, SARIMA, and Croston for sparse series. ARIMA picks the best of
   (0,1,1), (1,1,0) and (1,1,1) by AIC; SARIMA the best of the same with a seasonal
   (0,1,1)12 part. ARIMA needs 18 months after the holdout, SARIMA 30, and neither is
   tried on sparse series. A forecast above 3x the highest month is rejected. Turn them
   off in Settings if you don't need them.
   ARIMA fits run on the pre-differenced series with at most 50 optimiser steps, which is
   about 4x faster than a full fit for almost the same result. From 100 series up, series
   are spread over worker processes (all cores but one; set in Settings). The sidebar
   shows how long the last run took. The winner is the one with the lowest
   error on the last six months, which are held out. Seasonal models need 24 months
   of history; shorter series borrow the seasonal profile of their product family.
3. **Hard transition.** The predecessor's history is added to the successor's (times
   the volume factor). The predecessor gets the forecast before the effective month,
   the successor from it. Works for transitions in the past as well as the future.
4. **New product.** Borrows the like item's forecast times a factor, with an optional
   ramp, until it has six months of its own history.
5. **Split to versions.** A version receives forecast in a month if it is
   forecastable, the month is inside its launch and discontinue months, and no
   version transition rule excludes it for that customer group. Shares come from the
   recent shipment mix, an equal split, or manual percentages. Version forecasts
   always add back to the base SKU forecast; anything with no eligible version is
   held as Unallocated and flagged.
6. **Staggered transition.** Add one version transition row per customer group that
   moves early, and one row with `ALL` for everyone else.

After changing master data, rules or settings, press **Run forecast**.

## Not included yet

Planner overrides, history cleansing (outliers, stockouts), promotions, unit of
measure conversion, pipeline fill, one-to-many hard transitions, weekly buckets,
and multi-user access.

## Files

- `app.py` - the screens
- `planner/engine.py` - the forecast logic
- `planner/db.py` - storage
- `planner/dataio.py` - file import and Excel export
- `planner/explore.py` - data profiling, checks and suggestions for the Explore page
- `planner/sample.py` - sample data
- `tests/` - run with `python -m pytest tests` (needs `pip install pytest`)

To start again from nothing, use Clear the database on the Data page, or delete
`data/planner.db`.
