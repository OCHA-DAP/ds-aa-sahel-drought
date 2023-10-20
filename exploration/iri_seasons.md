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

# IRI with ASAP seasons

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import cftime
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray as rxr
import xarray as xr
from tqdm.notebook import tqdm

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
# utils.clip_asap_inseason_trimester(start_month=11)
```

```python
aoi = utils.load_codab_aoi()
iri = utils.load_iri()
da = utils.load_asap_inseason("dekad", 19)
da = da.where(da < 251)
da = da.squeeze(drop=True)
da = da.rename({"y": "Y", "x": "X"})
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
iri.isel(L=0, F=0).plot(ax=ax)
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
da.plot(ax=ax)
```

```python
# load inseason

da_ins = []

for month in range(1, 13):
    da_in = utils.load_asap_inseason("trimester", month, agg="any")
    da_in["start_month"] = month
    da_ins.append(da_in)

tri = xr.concat(da_ins, dim="start_month")
tri = tri.where(tri < 251)
```

```python
# get intersection of inseason and IRI

for F in tqdm(iri.F.values):
    start_months = [
        x if x < 13 else x - 12 for x in range(F.month + 1, F.month + 5)
    ]
    tri_l = tri.sel(start_month=start_months)
    tri_l = tri_l.rename({"start_month": "L", "x": "X", "y": "Y"})
    tri_l["L"] = iri.L
    da_iri = tri_l * iri.sel(F=F).interp_like(tri_l, method="nearest")
    da_iri = da_iri.where(da_iri > 0)
    da_iri = da_iri.transpose("L", "Y", "X")
    da_iri = da_iri.rio.set_spatial_dims(x_dim="X", y_dim="Y")
    save_dir = DATA_DIR / "private/processed/sah/iri/aoi_inseason_tif"
    filename = f"sah_iri_lowtercileprob_aoi_inseason_{F.isoformat().split('T')[0]}.tif"
    da_iri.rio.to_raster(save_dir / filename, driver="COG")
```

```python
# check plots

da = utils.load_iri_inseason(forecast_date="2020-02-16")

for L in range(1, 5):
    fig, ax = plt.subplots(figsize=(25, 5))
    da.sel(L=L).plot(ax=ax)
```

```python
type(iri.F.values[0])
isinstance(iri.F.values[0], cftime.Datetime360Day)
```

```python
cftime.Datetime360Day
```

```python
# calc raster stats

df_ins = []
percentiles = range(10, 100, 20)
percetile_cols = [f"{x}quant" for x in percentiles]

for F in tqdm(iri.F.values):
    da = utils.load_iri_inseason(forecast_date=F)
    df_in = da.oap.compute_raster_stats(
        gdf=aoi, feature_col="ADM0_CODE", percentile_list=percentiles
    )
    df_in["F"] = F
    df_ins.append(df_in)

stats = pd.concat(df_ins, ignore_index=True)
stats["F"] = pd.to_datetime(stats["F"])

stats["rel_month1"] = stats["F"].apply(lambda x: x.month) + stats["L"].astype(
    int
)
stats["rel_month1"] = stats["rel_month1"].apply(lambda x: x if x < 13 else x - 12)
stats["F_year"] = stats["F"].apply(lambda x: x.year)

save_dir = DATA_DIR / "private/processed/sah/iri"
filename = "iri_stats_adm0_any_inseason.csv"

stats.to_csv(save_dir / filename, index=False)
```

```python

```

```python
stats["rel_month1"].value_counts()
```

```python
rel_months = [6, 7, 8]
total_months = [x for x in range(min(rel_months), max(rel_months) + 3)]
plot_quants = [10, 50, 90]
plot_quant_cols = [f"{100 - x}quant" for x in percentiles if x in plot_quants]
max_per_year = (
    stats[stats["rel_month1"].isin(rel_months)]
    .groupby(["ADM0_CODE", "F_year"])[percetile_cols]
    .max()
    .reset_index()
)

for adm0 in max_per_year["ADM0_CODE"].unique():
    dff = max_per_year[max_per_year["ADM0_CODE"] == adm0][plot_quant_cols]
    for col in dff.columns:
        dff[col] = dff[col].sort_values().values
    dff["return_period"] = [7 / x for x in range(7, 0, -1)]
    fig, ax = plt.subplots()
    dff.plot(x="return_period", ax=ax)
    ax.set_title(
        f"{adm0}: IRI low tercile return period (since 2017)\n"
        f"All leadtimes, Relevant months: {total_months}"
    )
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Prob. of low tercile")
    ax.legend(title="Frac. of inseason AOI", labels=plot_quants)
```

```python
# check typical values by leadtime, month

agg_cols = ["L", "rel_month1"]
plot_cols = ["30quant", "count"]
for agg_col in agg_cols:
    dff = stats.groupby(agg_col)[plot_cols].agg(["max", "mean", "min"])
    for plot_col in plot_cols:
        fig, ax = plt.subplots()
        dff[plot_col].plot(ax=ax)
        ax.set_ylabel(plot_col)
```

```python
stats.groupby(["F_year", "rel_month1"])["count"].sum().reset_index().groupby(
    "rel_month1"
)["count"].min()
```
