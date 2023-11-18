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

# ECMWF reanalysis

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import matplotlib.pyplot as plt

from src import utils
```

```python
def calc_zscore(x):
    return (x - x.mean()) / x.std()


def calc_abs_anom(x):
    return x - x.mean()
```

```python
aoi = utils.load_codab(aoi_only=True)
```

```python
# utils.download_ecmwf_reanalysis()
# utils.process_ecmwf_reanalysis()
```

```python
ds = xr.load_dataset(
    utils.PROC_ECMWF_DIR
    / "ecmwf-reanalysis-monthly-precipitation_processed.nc"
)
```

```python
ds
```

```python
utils.process_ecmwf_inseason("reanalysis", "tp")
```

```python
ds["rank"].mean(dim=["latitude", "longitude"]).groupby(
    "valid_time.year"
).mean().plot()
```

```python
ch = utils.load_chirps()
```

```python
ch
```

```python
LAT, LON = 13.5116, 2.1254
fig, ax = plt.subplots()
(
    ds.sel(latitude=LAT, longitude=LON, method="nearest")
    .groupby("valid_time.month")
    .mean()["tp"]
    * 1000
    * 30
).plot(ax=ax)
ch.sel(latitude=LAT, longitude=LON, method="nearest").groupby(
    "valid_date.month"
).mean()["actual_precipitation"].plot(ax=ax)
```
