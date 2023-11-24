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

# ECMWF HRES seasonal forecast

Verifying the date ranges of the ECMWF seasonal forecasts.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import xarray as xr

from src import utils
```

```python
filename = "ecmwf_seasonalforecast_daily_2023-11-01.grib"
ds = xr.load_dataset(utils.RAW_ECMWF_DIR / filename)
```

```python
ds
```

```python
ds["tp"].mean(dim=["latitude", "longitude", "number"]).plot()
```

```python
(ds["valid_time"].max() - ds["valid_time"].min()).values.astype(
    "timedelta64[D]"
)
```

```python
ds["valid_time"]
```

```python
ds["valid_time"].min().values
```

```python
ds["valid_time"].max().values
```

```python
215 - 185
```

```python

```
