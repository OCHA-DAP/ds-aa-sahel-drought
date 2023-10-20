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
from ochanticipy import (
    GeoBoundingBox,
    IriForecastProb,
    IriForecastDominant,
    CodAB,
    create_country_config,
    create_custom_country_config,
)
from rasterio.enums import Resampling

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
# set up config
cod_all = utils.load_codab_all()
aoi = utils.load_codab_aoi()
country_config = create_custom_country_config("../sah.yaml")
geobb = GeoBoundingBox.from_shape(cod_all)

iri_prob = IriForecastProb(
    country_config=country_config, geo_bounding_box=geobb
)
```

```python
# Note: iri_prob.download() requires Python 3.9

# iri_prob.download()
# iri_prob.process()
da = iri_prob.load()["prob"]
da = da.sel(C=0).squeeze(drop=True)
```

```python
da
```

```python
da_aoi_1 = da.rio.clip(aoi.geometry, all_touched=True)
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
da_aoi_1.isel(L=0, F=0).plot(ax=ax)

save_dir = DATA_DIR / "private/processed/sah/iri"
filename = "sah_iri_lowtercileprob_aoi.nc"
filepath = save_dir / filename
# below lines may be needed due to Xarray bug
# if filepath.exists():
#     filepath.unlink()
da_aoi_1.to_netcdf(filepath)
```

```python

```

```python
resolution = 0.1
da_aoi_01 = utils.approx_mask_raster(da_aoi_1, "X", "Y", resolution)
da_aoi_01 = da_aoi_01.rio.clip(aoi.geometry, all_touched=True)
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
da_aoi_01.isel(L=0, F=0).plot(ax=ax)
```

```python
da_aoi_01
```

```python
percentiles = range(10, 100, 10)
percetile_cols = [f"{x}quant" for x in percentiles]
stats = da_aoi_01.oap.compute_raster_stats(
    gdf=aoi, feature_col="ADM0_CODE", percentile_list=percentiles
)
stats["rel_month1"] = stats["F"].apply(lambda x: x.month) + stats["L"].astype(
    int
)
stats["F_year"] = stats["F"].apply(lambda x: x.year)
```

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
max_per_year
```
