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

# ECMWF by ASAP season

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from tqdm.notebook import tqdm

from src import utils
```

```python
# utils.clip_asap_inseason_month()
```

```python
filename = "ecmwf_total-precipitation_sah_zscore.nc"
ec = xr.load_dataset(utils.PROC_ECMWF_DIR / filename)["zscore_lt"]
```

```python
aoi = utils.load_codab(aoi_only=True)
```

```python
for month in range(1, 13):
    season = utils.load_asap_inseason(interval="month", number=month)
    fig, ax = plt.subplots()
    season.plot(ax=ax)
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
season.where(season < 251).plot(ax=ax)
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
ax.axis("off")
```

```python
ec.interp_like(season, method="nearest")
```
