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
# utils.clip_asap_inseason_trimester()
```

## Load data

We have IRI at 1 degree resolution and "inseason" at 0.01 degree resolution.

```python
aoi = utils.load_codab_aoi()
iri = utils.load_iri()
da = utils.load_asap_inseason("dekad", 19)
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

## Process data

Not the most computationally efficient, but works for this size of data.

We just interpolate the IRI forecast at the coordinates of the "inseason".
Then we can re-aggregate however we want (adm0, adm1, livelihoods zone, etc).

```python
# utils.process_iri_inseason()
```

```python
# check plots

da = utils.load_iri_inseason(forecast_date="2020-02-16")

for L in range(1, 5):
    fig, ax = plt.subplots(figsize=(25, 5))
    da.sel(L=L).plot(ax=ax)
```

```python
# utils.calculate_iri_inseason_stats()
```

```python
stats = utils.load_iri_inseason_stats()
stats["rel_month1"] = stats["rel_month1"].apply(
    lambda x: x if x < 13 else x - 12
)

# calculate fraction of pixels in AOI that might have a season
# i.e., ones that don't have an error code ( > 250) in ASAP rasters

# note: must do as float, otherwise compute_raster_stats doesn't work
s_pos_da = xr.where(utils.load_asap_inseason("dekad", 1) < 251, 1.0, 0.0)
s_pos = s_pos_da.oap.compute_raster_stats(aoi, feature_col="ADM0_CODE")
s_pos = s_pos.rename(
    columns={"sum": "possible_season_pixels", "count": "total_pixels"}
)

# fraction with possible season is close to 100% for BFA, but less for TCD and NER
display(s_pos)

stats = stats.merge(
    s_pos[["ADM0_CODE", "possible_season_pixels", "total_pixels"]],
    on="ADM0_CODE",
)
stats["frac_reporting"] = stats["count"] / stats["possible_season_pixels"]
display(stats)
```

## Selecting relevant dates

We can select relevant dates either by:

- Just picking which months we care about
  - Based on previous frameworks this would be Jun - Oct
- Only monitoring once the fraction of "reporting" pixels
goes above a certain level, say 50%

```python
# check typical values by leadtime, month

agg_cols = ["L", "rel_month1"]
plot_cols = ["30quant", "frac_reporting"]

for agg_col in agg_cols:
    year_group = stats.groupby(["F_year", agg_col])[plot_cols].mean()
    dff = year_group.groupby(agg_col)[plot_cols].agg(["max", "mean", "min"])
    for plot_col in plot_cols:
        fig, ax = plt.subplots()
        dff[plot_col].plot(ax=ax)
        ax.set_ylabel(plot_col)

# note that rel_month1 is the first relevant month of the forecast

# things are pretty stable by leadtime, except fewer tend to be reporting
# at L=4

# we see that if we do not select which months we care about,
# we will often end up triggering in October
```

Unfortunately `frac_reporting` is still pretty high in September.
This means that we could trigger for a forecast relevant for Sep-Oct-Noc,
which doesn't seem right.

Basically this means that we need to either:

- Threshold based on how many dekads within the trimester each pixel is in season
  - We are currently just using `any()` (i.e. the trimester is inseason
if any of its dekads are inseason)
- Use ECMWF which forecasts monthly instead of trimesterially

Here we are doing the land area fraction as `pixels_above_threshold / pixels_reporting`,
but this might not make sense. Based on previous frameworks, the analogue
here would be more like `pixels_above_theshold / pixels_with_possible_season`.
That way the denominator stays the same.

```python
# plot return period for different methods of selecting relevant dates

frac_reporting_threshold = 0.5
rel_months = [6, 7, 8]
total_months = [x for x in range(min(rel_months), max(rel_months) + 3)]
plot_quants = [10, 50, 90]
plot_quant_cols = [f"{100 - x}quant" for x in percentiles if x in plot_quants]

for adm0 in max_per_year["ADM0_CODE"].unique():
    for method in ["months", "frac_reporting"]:
        if method == "months":
            rel_stats = stats[stats["rel_month1"].isin(rel_months)]
            title_text = f"Relevant months: {total_months}"
        else:
            rel_stats = stats[
                stats["frac_reporting"] > frac_reporting_threshold
            ]
            title_text = f"Fraction Reporting > {frac_reporting_threshold}"
        max_per_year = (
            rel_stats.groupby(["ADM0_CODE", "F_year"])[percetile_cols]
            .max()
            .reset_index()
        )
        dff = max_per_year[max_per_year["ADM0_CODE"] == adm0][plot_quant_cols]
        for col in dff.columns:
            dff[col] = dff[col].sort_values().values
        dff["return_period"] = [7 / x for x in range(7, 0, -1)]
        fig, ax = plt.subplots()
        dff.plot(x="return_period", ax=ax)
        ax.set_title(
            f"{adm0}: IRI low tercile return period (since 2017)\n"
            f"All leadtimes, {title_text}"
        )
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Prob. of low tercile")
        ax.legend(title="Frac. of inseason AOI", labels=plot_quants)
```
