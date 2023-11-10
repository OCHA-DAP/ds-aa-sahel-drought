---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: ds-aa-sahel-drought
    language: python
    name: ds-aa-sahel-drought
---

# CHIRPS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import datetime

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr
from ochanticipy import (
    CodAB,
    GeoBoundingBox,
    IriForecastProb,
    create_country_config,
    create_custom_country_config,
    ChirpsMonthly,
)
from scipy.stats import zscore, linregress
from tqdm.notebook import tqdm
```

```python
from src import utils
```

```python
aoi = utils.load_codab(aoi_only=True)
```

## Load CHIRPS

```python
chirps = utils.load_chirps()
```

```python
chirps
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
aoi.boundary.plot(ax=ax, color="k", linewidth=0.5)
chirps.isel(valid_date=8)["actual_precipitation"].plot(ax=ax)
ax.set_xlim(aoi.total_bounds[[0, 2]])
ax.set_ylim(aoi.total_bounds[[1, 3]])
```

## Load ECMWF

```python
ec = utils.load_ecmwf()
ec_df = ec.to_dataframe()
ec_df = ec_df.reset_index()
ec_df["forecast_precipitation"] = ec_df["tprate"] * 3600 * 24 * 365 / 12 * 1000
ec_df = ec_df.drop(columns="tprate")
ec_df["valid_date"] = ec_df["time"] + ec_df["leadtime"].apply(
    lambda x: pd.DateOffset(months=x - 1)
)
ec_df = ec_df.rename(columns={"time": "pub_date"})
ec_df = ec_df.groupby(
    ["valid_date", "leadtime", "latitude", "longitude"]
).first()
ec_da = ec_df.to_xarray()
```

```python
ec
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
aoi.boundary.plot(ax=ax, color="k", linewidth=0.5)
ec.isel(time=3, leadtime=5).plot(ax=ax)
ax.set_xlim(aoi.total_bounds[[0, 2]])
ax.set_ylim(aoi.total_bounds[[1, 3]])
```

```python
ec_interp = ec_da["forecast_precipitation"].interp_like(
    chirps,
    method="nearest",
)
```

```python
ec_interp.rio.write_crs(4326, inplace=True)
ec_clip = ec_interp.rio.clip(aoi.geometry, all_touched=True)
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
aoi.boundary.plot(ax=ax, color="k", linewidth=0.5)
ec_clip.isel(valid_date=8, leadtime=5).plot(ax=ax)
ax.set_xlim(aoi.total_bounds[[0, 2]])
ax.set_ylim(aoi.total_bounds[[1, 3]])
```

```python
fig, ax = plt.subplots()
chirps["actual_precipitation"].mean(dim=["latitude", "longitude"]).isel(
    valid_date=slice(400, 450)
).plot(ax=ax)
ec_interp.mean(dim=["latitude", "longitude"]).sel(leadtime=1).isel(
    valid_date=slice(400, 450)
).plot(ax=ax)
ax.legend(["CHIRPS", "ECMWF"])
```

## Correlation

```python
das = []

for valid_month in range(1, 13):
    da = xr.corr(
        ec_interp.isel(
            valid_date=(ec_interp.valid_date.dt.month == valid_month)
        ),
        ch_da["actual_precipitation"].isel(
            valid_date=(ch_da.valid_date.dt.month == valid_month)
        ),
        dim=["valid_date"],
    )
    da["valid_month"] = valid_month
    das.append(da)

corr = xr.concat(das, dim="valid_month")
```

```python
corr.mean(dim=["latitude", "longitude"]).plot.line(x="leadtime")
```

```python
season = utils.load_asap_inseason_allmonths()
season = season.rename({"month": "valid_month"})
```

```python
corr_inseason = corr.interp_like(season, method="nearest") * season.where(
    season == 1
)
```

```python
corr_inseason.mean(dim=["latitude", "longitude"]).plot.line(x="leadtime")
```

```python
fig, ax = plt.subplots()
corr.where(corr.valid_month.isin([7, 8, 9])).mean(
    dim=["latitude", "longitude", "valid_month"]
).plot(ax=ax)
corr_inseason.where(corr.valid_month.isin([7, 8, 9])).mean(
    dim=["latitude", "longitude", "valid_month"]
).plot(ax=ax)
ax.legend(["all area", "in season only"])
ax.set_title("correlation between CHIRPS and ECMWF for Jul, Aug, Sep")
```

```python
valid_month = 8
ch_inseason = ch_da["actual_precipitation"].isel(
    valid_date=(ch_da.valid_date.dt.month == valid_month)
)
ch_inseason = ch_inseason.interp_like
```

```python
season
```

```python
ch_da["valid_month"] = ch_da["valid_date"].dt.month
```

```python
ch_da.assign_coords({"valid_month": ch_da["valid_date"].dt.month})
```

```python
da = utils.load_chirps_inseason("2020-08-01")
```

```python
da.plot()
```

```python
utils.load_chirps_inseason(valid_date="2021-09-01", variable="zscore").plot()
```

```python
dfs = []
for year in tqdm(range(1981, 2023)):
    das = []
    for month in range(1, 13):
        da_in = utils.load_chirps_inseason(
            valid_date=f"{year}-{month:02}-01", variable="zscore"
        )
        da_in["month"] = month
        das.append(da_in)
    da = xr.concat(das, dim="month")
    df_in = da.oap.compute_raster_stats(aoi, feature_col="ADM0_CODE")
    df_in["year"] = year
    dfs.append(df_in)

df = pd.concat(dfs, ignore_index=True)
```

```python
ds = df.set_index(["ADM0_CODE", "year", "month"]).to_xarray()
```

```python
da_scores = ds["mean"].weighted(ds["count"]).mean(dim="month")
```

```python
df_scores = da_scores.to_dataframe().reset_index()
```

```python
df_scores.pivot_table(index="year", columns="ADM0_CODE").plot()
```

```python
grouped = df_scores.groupby("ADM0_CODE")


def calculate_trendline(group):
    slope, intercept, _, _, _ = linregress(group["year"], group["mean"])
    group["pred"] = intercept + slope * group["year"]
    return group


df_scores = grouped.apply(calculate_trendline).reset_index(drop=True)
```

```python
df_scores["normalized"] = df_scores["mean"] - df_scores["pred"]
```

```python
df_scores.pivot_table(
    index="year", columns="ADM0_CODE", values="normalized"
).plot()
```

```python
def is_worst_third(group):
    group["worst_third"] = group["normalized"] <= group["normalized"].quantile(
        1 / 3
    )
    return group


df_scores = (
    df_scores.groupby("ADM0_CODE").apply(is_worst_third).reset_index(drop=True)
)
```

```python
df_scores[df_scores["worst_third"]]
```

```python
filename = "chirps_score_allmonths_inseason_adm0.csv"
df_scores.to_csv(utils.PROC_CHIRPS_DIR / filename, index=False)
```
