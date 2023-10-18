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

# ASAP phenology

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
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
aoi = utils.load_codab_aoi()
```

```python
# utils.process_asap_raw()
```

```python
da = utils.load_asap_sos_eos()
da = da.rio.clip(aoi.geometry, all_touched=True)
da = da.where(da < 251)
da = da.assign_attrs({"_FillValue": np.nan})
# bring all into two years
da = da.where(da < 37, da - 36)
# da = da.where(da < 37, da - 36)
```

```python
def dekad_to_month(dekad, pos=None):
    if dekad < 1:
        return None
    dekad = np.round(dekad).astype(int)
    dekad = dekad - 72 if dekad > 72 else dekad
    dekad = dekad - 36 if dekad > 36 else dekad
    month = datetime.datetime(2023, (dekad - 1) // 3 + 1, 1).strftime("%b")
    num = (dekad - 1) % 3 + 1
    return f"{month} {num}"
```

```python
dekad_to_month(25)
```

```python
s_es = ["s", "e", "sen"]
s_e_labels = ["start", "end", "senescence"]
for s_e, s_e_label in zip(s_es, s_e_labels):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.axis("off")
    aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
    contour = da.sel(season=1, s_e=s_e).plot.contourf(ax=ax, vmin=25, vmax=37)
    contour.colorbar.formatter = dekad_to_month
    contour.colorbar.set_label("dekad")
    ax.set_title(f"Season 1 {s_e_label}")
    plt.show()
```

```python
fig, ax = plt.subplots()
for s_e, s_e_label in zip(s_es, s_e_labels):
    da.sel(season=1, s_e=s_e).plot.hist(
        bins=[x - 0.5 for x in range(1, 73)], ax=ax, alpha=0.5, label=s_e_label
    )


ax.set_xticks(range(1, 73, 3))
ax.xaxis.set_major_formatter(dekad_to_month)
ax.set_xlim([12, 43])
ax.set_title("Season 1 distribution")
ax.set_xlabel("Dekad")
ax.set_ylabel("Pixel count (1km resolution)")
ax.legend()
plt.show()
```

```python
da.sel(season=2, s_e="s").plot.hist()
```

```python
da.sel(season=2, s_e="s").plot(cmap="hsv")
```

```python
da.sel(season=2, s_e="s").plot(cmap="hsv")
```

```python
s1_len = da.sel(season=1, s_e="e") - da.sel(season=1, s_e="s")
s1_len.where(s1_len > 0).plot()
```

```python
s2_len = da.sel(season=2, s_e="e") - da.sel(season=2, s_e="s")
s2_len.where(s2_len > 0).plot()
```

```python
longest = xr.where(s2_len > s1_len, da.sel(season=2), da.sel(season=1))
longest = longest.where(longest < 72, longest - 36)
longest = longest.where(longest < 36, longest - 36)
longest.sel(s_e="e").plot(cmap="hsv")
```

```python
filename = "phenos1_v03.tif"
sos1 = rxr.open_rasterio(load_dir / filename)
sos1 = sos1.assign_attrs({"_FillValue": np.nan})
sos1 = sos1.where(sos1 < 251, drop=True)
sos1 = sos1.rio.clip(codab["geometry"], all_touched=True)
sos1.plot()
```

```python
filename = "phenoe1_v03.tif"
soe1 = rxr.open_rasterio(load_dir / filename)
soe1 = soe1.assign_attrs({"_FillValue": np.nan})
soe1 = soe1.where(soe1 < 251, drop=True)
soe1 = soe1.rio.clip(codab["geometry"], all_touched=True)
soe1.plot()
```

```python
# check season 2 - basically nowhere is bimodal, so we can ignore
filename = "phenos2_v03.tif"
sos2 = rxr.open_rasterio(load_dir / filename)
sos2 = sos2.assign_attrs({"_FillValue": np.nan})
sos2 = sos2.where(sos2 < 251, drop=True)
sos2 = sos2.rio.clip(codab["geometry"], all_touched=True)
sos2.plot()
```
