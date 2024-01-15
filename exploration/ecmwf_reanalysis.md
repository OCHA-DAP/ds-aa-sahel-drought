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

# ECMWF reanalysis

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

from src import utils
```

```python
aoi = utils.load_codab(aoi_only=True)
```

```python
# utils.download_ecmwf_reanalysis()
# utils.process_ecmwf_reanalysis()
# utils.process_ecmwf_inseason("reanalysis", "tp")
```

```python
da = utils.load_ecmwf_inseason(
    product="reanalysis", variable="tp", publication_date="2020-08-01"
)
vmin, vmax = da.min(), da.max()
```

```python
filename = "ecmwf-reanalysis-monthly-precipitation_processed.nc"
ec = xr.load_dataset(utils.PROC_ECMWF_DIR / filename)
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
aoi.boundary.plot(ax=ax, color="k")
ec.sel(valid_time="2020-08-01")["tp"].plot(ax=ax, vmin=vmin, vmax=vmax)
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
aoi.boundary.plot(ax=ax, color="k")
da.plot(ax=ax, vmin=vmin, vmax=vmax)
```

```python
dfs = []
for year in tqdm(range(1981, 2024)):
    das = []
    for month in range(1, 13):
        da_in = utils.load_ecmwf_inseason(
            product="reanalysis",
            publication_date=f"{year}-{month:02}-01",
            variable="tp",
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
# calculate mean rainfall per year
yearly = df.groupby(["ADM0_CODE", "year"]).sum()
yearly["monthly_mean"] = yearly["sum"] / yearly["count"] * 1000
yearly = yearly["monthly_mean"].reset_index()
yearly.pivot(columns="ADM0_CODE", index="year").plot()
```

```python
filename = "ecmwf_reanalysis_score_allmonths_inseason_adm0.csv"
yearly.to_csv(utils.PROC_ECMWF_DIR / filename, index=False)
```

```python
yearly
```

```python
(da.sel(month=8) * 1000 * 60).plot()
```

```python
ds["rank"].mean(dim=["latitude", "longitude"]).groupby(
    "valid_time.year"
).mean().plot()
```

```python
ch = utils.load_chirps()
```

```python
ch
```

```python
LAT, LON = 13.5116, 2.1254
fig, ax = plt.subplots()
(
    ds.sel(latitude=LAT, longitude=LON, method="nearest")
    .groupby("valid_time.month")
    .mean()["tp"]
    * 1000
    * 30
).plot(ax=ax)
ch.sel(latitude=LAT, longitude=LON, method="nearest").groupby(
    "valid_date.month"
).mean()["actual_precipitation"].plot(ax=ax)
```

```python
filename = "/Users/tdowning/Downloads/adaptor.mars.internal-1700508667.8038206-22833-2-d609af6f-6cb0-4872-8300-ed6f35284d26.grib"
test = xr.load_dataset(filename)
```

```python
test["tp"].plot()
```
