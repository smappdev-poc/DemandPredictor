# Demo dataset

The demo copy of Demand Predictor (`run_demo.bat`, port 8502, database `data/demo.db`)
uses public data only.

## Source and licence

**M5 Forecasting - Accuracy** (Walmart unit sales), Makridakis, Spiliotis and Gilliland /
University of Nicosia, licensed **CC BY 4.0**: https://doi.org/10.5281/zenodo.12636070.
The copy used was downloaded from Nixtla's mirror:
https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip

CC BY 4.0 allows any use, including commercial demos, provided the source is credited and
changes are indicated. Keep this credit with any demo you show or share.

## Changes made to the original data

- Kept the 1,000 best-selling FOODS and HOUSEHOLD products (HOBBIES dropped).
- Each of the 10 Walmart stores is treated as a customer account; the largest store in
  each state, plus the second largest in California, are key customers (CA_1, CA_3,
  TX_2, WI_2). Each state is treated as a ship-from DC (DC-CA, DC-TX, DC-WI).
- Daily units summed to months; partial first and last months dropped.
- **All dates moved forward exactly 10 years**: history runs Feb 2021 to May 2026
  (original: Feb 2011 to May 2016).
- **Product lifecycle data is generated, not real** (M5 has none), with a fixed seed:
  - 120 products split into V1 and V2 at a past date, with two months of overlap;
  - 30 of those get a future V3, introduced at Store CA 1 first and everyone else three months later;
  - 15 products replaced by a new base SKU (suffix -N) at a past date;
  - 5 products get a future replacement (hard transition, volume factor 1.05);
  - 8 new products (e.g. FOODS_3_NEW01) launch in the future, borrowing a like item's forecast.
- Product descriptions are generic ("Foods 3 product 090"); the original data has none.

## Rebuilding it

`python tools/build_demo_m5.py <folder with the unzipped M5 csv files>` rewrites
`demo/m5/*.parquet`. The result is the same every time.
