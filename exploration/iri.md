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

# IRI

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from rasterio.enums import Resampling

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

## Load and process

Load IRI low tercile and AOI. Up-sample IRI and clip to AOI.

```python
utils.process_iri_aoi_lowtercile()
iri = utils.load_iri()
aoi = utils.load_codab(aoi_only=True)

resolution = 0.1
iri_01 = utils.approx_mask_raster(iri, "X", "Y", resolution)
iri_01 = iri_01.rio.clip(aoi.geometry, all_touched=True)
```

Check that resolution of clip seems reasonable - fits boundaries decently well.

```python
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
iri_01.isel(L=0, F=0).plot(ax=ax)
plt.show()
```

## Calculate stats

Calculate basic stats for IRI forecast, by country.

```python
percentiles = range(10, 100, 10)
percetile_cols = [f"{x}quant" for x in percentiles]
stats = iri_01.oap.compute_raster_stats(
    gdf=aoi, feature_col="ADM0_CODE", percentile_list=percentiles
)
stats["rel_month1"] = stats["F"].apply(lambda x: x.month) + stats["L"].astype(
    int
)
stats["F_year"] = stats["F"].apply(lambda x: x.year)
```

Here we can plot the return period of the low tercile probability by country.
It's more or less what we expect, and is similar to the analysis previously
done for the frameworks.
Note that this doesn't include any seasonality,
other than only looking at Jun-Oct.

The mean is pretty close to the 50% quantile, so the distribution isn't too skewed.

```python
rel_months = [6, 7, 8]
total_months = [x for x in range(min(rel_months), max(rel_months) + 3)]
plot_quants = [10, 30, 50, 70, 90]
plot_quant_cols = [f"{100 - x}quant" for x in percentiles if x in plot_quants]
max_per_year = (
    stats[stats["rel_month1"].isin(rel_months)]
    .groupby(["ADM0_CODE", "F_year"])[percetile_cols + ["mean"]]
    .max()
    .reset_index()
)

for adm0 in max_per_year["ADM0_CODE"].unique():
    dff = max_per_year[max_per_year["ADM0_CODE"] == adm0][
        plot_quant_cols + ["mean"]
    ]
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
    ax.legend(title="Frac. of AOI", labels=plot_quants + ["mean"])
```

```python

```
