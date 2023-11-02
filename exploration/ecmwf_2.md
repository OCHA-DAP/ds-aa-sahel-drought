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

# ECMWF 2

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import datetime

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
utils.process_ecmwf()
```

```python
aoi = utils.load_codab(aoi_only=True)
```

```python
da = utils.load_ecmwf()
ds = da.to_dataset()
da["valid_month"] = (da["time"].dt.month + da["leadtime"] - 1) % 12 + 1
df = da.to_dataframe().reset_index()
df["rank"] = (
    df.groupby(["valid_month", "latitude", "longitude"])["tprate"]
    .rank()
    .astype("float32")
)
ds["rank"] = (
    df.groupby(["leadtime", "time", "latitude", "longitude"])
    .mean()
    .to_xarray()["rank"]
)
ds = ds.assign_coords(
    {"valid_time": ds["time"] + ds["leadtime"] * 31 * 24 * 3600 * 1000000000}
)
rank = ds["rank"]
rank.rio.write_crs(4326, inplace=True)
ds
```

```python
pctile = (rank - 1) / (rank.max() - 1)
pctile.name = "percentile"
```

```python
valid_month = 7
valid_year = 1981

valid_date = pd.to_datetime(datetime.date(valid_year, valid_month, 1))
leadtimes = range(1, 7)
forecast_dates = [valid_date - pd.DateOffset(months=x) for x in leadtimes]

for leadtime, forecast_date in zip(leadtimes, forecast_dates):
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.axis("off")
    aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
    pctile.sel(time=forecast_date, leadtime=leadtime).plot(
        ax=ax, vmin=0, vmax=1
    )
    ax.set_title(
        f"Produced: {forecast_date:%B %Y}; "
        f"Leadtime: {leadtime} months; "
        f"Valid: {valid_date:%B %Y}"
    )
```

```python
stats = pctile.oap.compute_raster_stats(gdf=aoi, feature_col="ADM0_CODE")
stats["valid_time"] = stats.apply(
    lambda row: row["time"] + pd.DateOffset(months=row["leadtime"]), axis=1
)
```

```python
rel_months = [7, 8, 9]
rel_years = range(2017, 2100)
for adm0 in stats["ADM0_CODE"].unique():
    print(adm0)
    fig, ax = plt.subplots()
    dff = (
        stats[
            (stats["ADM0_CODE"] == adm0)
            & (stats["valid_time"].dt.month.isin(rel_months))
            & (stats["valid_time"].dt.year.isin(rel_years))
        ].groupby(stats["valid_time"].dt.date)
        # .agg(["min", "max"])
    )
    dff.boxplot(column="mean", subplots=False, rot=90)
    # for leadtime in range(1, 5):
    #     dff[dff["leadtime"] == leadtime].plot(
    #         ax=ax, x="valid_time", y="mean", linestyle="None", marker="."
    #     )
    # dff["valid_time", "mean"].plot(ax=ax)

    # display(dff.sort_values("mean")[:5])
```

```python
ds["tprate"].sel(leadtime=4).isel(time=0).plot()
```

```python
leadtime = 4
filename = f"ecmwf-total-leadtime-{leadtime}_sys5.nc"
test = xr.open_dataset(utils.RAW_ECMWF_DIR / filename, engine="cfgrib")
```

```python
for step in test.step:
    fig, ax = plt.subplots()
    test["tprate"].sel(step=step).isel(number=0, time=0).plot(ax=ax)
```
