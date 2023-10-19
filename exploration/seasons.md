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
from tqdm.notebook import tqdm

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
aoi = utils.load_codab_aoi()
```

```python
test = rxr.open_rasterio(save_dir / f"inseason_dekad1_sen_aoi.tif")
test.plot()
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
s_es = ["s", "e", "sen"]
s_e_labels = ["start", "end", "senescence"]
for s_e, s_e_label in zip(s_es, s_e_labels):
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.axis("off")
    aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
    if s_e == "s":
        contour = da.sel(season=1, s_e=s_e).plot.contourf(ax=ax, robust=True)
    else:
        contour = da.sel(season=1, s_e=s_e).plot.contourf(
            ax=ax, vmin=25, vmax=37
        )
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
len_e = da.sel(s_e="e") - da.sel(s_e="s")
len_sen = da.sel(s_e="sen") - da.sel(s_e="s")
```

```python
fig, ax = plt.subplots()
for da_len, label in zip([len_e, len_sen], ["end", "senescence"]):
    da_len.sel(season=1).plot.hist(
        ax=ax, alpha=0.5, label=label, bins=[x - 0.5 for x in range(6, 25)]
    )
ax.set_xticks(range(4, 25, 2))
ax.set_title("Season 1 length distribution")
ax.set_xlabel("Dekads")
ax.set_ylabel("Pixel count (1km resolution)")
ax.legend()
plt.show()
```

```python
for da_len, label in zip([len_e, len_sen], ["end", "senescence"]):
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.axis("off")
    aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
    contour = da_len.sel(season=1).plot.contourf(ax=ax, vmin=5, vmax=25)
    contour.colorbar.formatter = lambda x, _: x.astype(int)
    contour.colorbar.set_label("dekads")
    ax.set_title(f"Season 1 length, '{label}' as EOS")
    plt.show()
```

```python
# check where season 2 exists - basically nowhere
da.sel(season=2, s_e="s").plot()
```

```python
da.sel(season=2, s_e="s").plot.hist()
```

```python
eos_dif = da.sel(s_e="e") - da.sel(s_e="sen")
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
contour = eos_dif.sel(season=1).plot.contourf(ax=ax, robust=True)
contour.colorbar.set_label("dekads")
ax.set_title("Dekads between senscence and season end")
```
