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

import geopandas as gpd
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
codab = utils.load_codab_all()
```

```python
# utils.process_asap_raw()
```

```python
da = utils.load_asap_sos_eos()
da = da.where(da < 251)
```

```python
da
```

```python
da.sel(season=1, s_e="s").plot(cmap="hsv")
```

```python
da.sel(season=1, s_e="s").plot.hist()
```

```python
da.sel(season=2, s_e="s").plot.hist()
```

```python
da.sel(season=1, s_e="e").plot(cmap="hsv")
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
