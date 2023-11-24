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

# Seasons by Adm

Look at the seasonality per admin, and how we can choose which months of the
ECMWF forecast we care about.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

from src import utils
```

```python
ADM_LEVEL = "ADM0_CODE"
TIME_INTERVAL = "month"
```

## Load data

Note that we are loading the default ASAP inseason files per month.
These aggregate the inseason dekads with `any()`.
In other words, a month is considered "in season" if any of its dekads are
"in season" (considered on a pixel-by-pixel basis).

```python
das = []
end = 13 if TIME_INTERVAL == "month" else 37
for d in range(1, end):
    da = utils.load_asap_inseason(TIME_INTERVAL, d)
    da[TIME_INTERVAL] = d
    das.append(da)
season = xr.concat(das, dim=TIME_INTERVAL)
```

```python
aoi = utils.load_codab(aoi_only=True)
```

## Process Data

Here we count the pixels in each admin:

- `relevant_pixels`: the number of pixels that could be in season
(i.e. that ASAP does not have an error code for, so they must be
in season at some point)
- `total_pixels`: the total number of 0.1 deg pixels in the admin

```python
total_pixels = (
    season.where(season != 254)
    .isel({TIME_INTERVAL: 0})
    .oap.compute_raster_stats(aoi, feature_col=ADM_LEVEL)
)
relevant_pixels = (
    season.where(season < 251)
    .isel({TIME_INTERVAL: 0})
    .oap.compute_raster_stats(aoi, feature_col=ADM_LEVEL)
)
total_pixels = total_pixels.merge(
    relevant_pixels[[ADM_LEVEL, "count"]],
    on=ADM_LEVEL,
    suffixes=["_total", ""],
)
total_pixels["frac_relevant"] = (
    total_pixels["count"] / total_pixels["count_total"]
)
total_pixels = total_pixels[
    [ADM_LEVEL, "count", "count_total", "frac_relevant"]
]
total_pixels = total_pixels.rename(
    columns={"count": "relevant_pixels", "count_total": "total_pixels"}
)
```

Then we can count the number of pixels that are in season, per admin and per
month. We are sticking to month for now because that's the granulatiry of
ECMWF forecasts. We then calculate `frac_inseason` as the fraction of pixels
that _could_ be in season (i.e. the same as `relevant_pixels` above) that are
_actually_ in season.

```python
season_stats = season.where(season < 251).oap.compute_raster_stats(
    aoi, feature_col=ADM_LEVEL
)
season_stats = season_stats.merge(
    total_pixels,
    on=ADM_LEVEL,
)
season_stats["frac_inseason"] = season_stats["sum"] / season_stats["count"]
```

## Plot seasonality per admin, per month

To decide which months to monitor for each country,
we only want to look at those where a sufficient fraction of pixels
are in season (i.e., where `frac_inseason` is above a certain threshold).

We can see how to set the `inseason_threshold` parameter based on what
months would be considered in season for each country.

From the plot below, we see that if we set `inseason_threshold = 0.5`,
then Burkina would have 5 months in season (Jun to Oct inclusive),
whereas Niger and Chad would only have 3 months (Jul to Sep inclusive).

The existing Burkina framework includes only 3 months (Jul to Sep),
so it probably makes sense to set `inseason_threshold > 0.7`,
since this is approximately the point at which Burkina goes to
only having 3 months in season.
Fortunately, setting this threshold does not reduce the months in season for
Niger or Chad, since

```python
dfs = []
for inseason_threshold in np.linspace(0, 1, 500):
    df = (
        season_stats[season_stats["frac_inseason"] > inseason_threshold]
        .groupby(ADM_LEVEL)["month"]
        .agg(["min", "max", "size"])
    )
    df["inseason_threshold"] = inseason_threshold
    dfs.append(df)

months_inseason = pd.concat(dfs).reset_index()

fig, axs = plt.subplots(
    3, 1, sharex=True, constrained_layout=True, figsize=(5, 5)
)
for ax, adm0 in zip(axs, ["BFA", "NER", "TCD"]):
    months_inseason[months_inseason[ADM_LEVEL] == adm0].plot(
        ax=ax,
        x="inseason_threshold",
        y=["min", "max"],
    )
    ax.set_title(adm0)
    ax.set_ybound([5, 11])
    ax.set_yticks(range(5, 12))
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xbound([0, 1])
    ax.set_ylabel("months\nin season")
    ax.get_legend().remove()
```

```python

```
