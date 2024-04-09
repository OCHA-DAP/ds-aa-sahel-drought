---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: ds-aa-sahel-drought
    language: python
    name: ds-aa-sahel-drought
---

# LAC 2024 primera support

LAC support for checking ECMWF grid

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr

from src import utils
```

```python
LAC_AOI_DIR = utils.DATA_DIR / "public" / "processed" / "lac"
LAC_EC_EXP_DIR = utils.DATA_DIR / "public" / "exploration" / "lac" / "ecmwf"
```

```python
filename = "adm0_central_ameria_dry_corridor_simp.shp"
lac_aoi = gpd.read_file(LAC_AOI_DIR / filename)
```

```python
lac_aoi.plot()
```

```python
lac_aoi.total_bounds
```

```python
lac_aoi.total_bounds.astype(int)
```

```python
filename = "test_noround.grib"
no_round = xr.load_dataset(LAC_EC_EXP_DIR / filename)
```

```python
fig, ax = plt.subplots()
lac_aoi.boundary.plot(ax=ax)
no_round["tprate"].isel(number=0).plot(ax=ax)
```

```python
filename = "test_round.grib"
rounded = xr.load_dataset(LAC_EC_EXP_DIR / filename)
```

```python
fig, ax = plt.subplots()
lac_aoi.boundary.plot(ax=ax)
rounded["tprate"].isel(number=0).plot(ax=ax)
```

```python
no_round.isel(number=0, latitude=0, longitude=0)["tprate"].values
```

```python
rounded.isel(number=0, latitude=0, longitude=0)["tprate"].values
```
