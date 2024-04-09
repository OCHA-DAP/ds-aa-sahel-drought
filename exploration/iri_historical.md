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

# IRI historical

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR_NEW"))
IRI_RAW_PATH = DATA_DIR / "public" / "raw" / "glb" / "iri" / "iri.nc"
```

```python
def upsample_dataarray(
    da: xr.DataArray,
    resolution: float = 0.1,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    new_lat = np.arange(
        da[lat_dim].min() - 1, da[lat_dim].max() + 1, resolution
    )
    new_lon = np.arange(
        da[lon_dim].min() - 1, da[lon_dim].max() + 1, resolution
    )
    return da.interp(
        coords={
            lat_dim: new_lat,
            lon_dim: new_lon,
        },
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )
```

```python
aoi_all = utils.load_codab(aoi_only=True)
```

```python
aoi_all
```

```python
aoi_all.plot()
```

```python
iri_all = xr.load_dataset(IRI_RAW_PATH, decode_times=False, drop_variables="C")
iri_all.F.attrs["calendar"] = "360_day"
iri_all = xr.decode_cf(iri_all)
iri_all = iri_all.rio.write_crs(4326)
```

```python
iri_sah = iri_all.rio.clip(aoi_all.geometry, all_touched=True)["prob"].isel(
    C=0
)
iri_sah_up = upsample_dataarray(iri_sah, lat_dim="Y", lon_dim="X")
```

```python
countries = [
    {
        "iso3": "BFA",
        "thresh": {"prob": 40, "frac": 0.1},
        "windows": [
            {
                "number": 1,
                "monitoring_points": [{"forecast_month": 3, "leadtime": 3}],
            },
            {
                "number": 2,
                "monitoring_points": [{"forecast_month": 7, "leadtime": 1}],
            },
        ],
    },
    {
        "iso3": "TCD",
        "thresh": {"prob": 42.5, "frac": 0.2},
        "windows": [
            {
                "number": 1,
                "monitoring_points": [
                    {"forecast_month": 3, "leadtime": 4},
                    {"forecast_month": 4, "leadtime": 3},
                ],
            },
            {
                "number": 2,
                "monitoring_points": [
                    {"forecast_month": 5, "leadtime": 2},
                    {"forecast_month": 6, "leadtime": 1},
                ],
            },
        ],
    },
]
```

```python
years = range(2017, 2025)
dicts = []
for country in countries:
    aoi = aoi_all[aoi_all["ADM0_CODE"] == country["iso3"]]
    iri_country = iri_sah_up.rio.clip(aoi.geometry, all_touched=True)
    thresh = country["thresh"]
    for year in years:
        for window in country["windows"]:
            for monitoring_point in window["monitoring_points"]:
                date_str = f'{year}-{monitoring_point["forecast_month"]:02}-16'
                if date_str not in iri_country.F.dt.strftime("%Y-%m-%d"):
                    print(f"no forecast for {date_str}")
                    continue
                iri_point = iri_country.sel(
                    F=date_str,
                    L=monitoring_point["leadtime"],
                )
                count_above_threshold = (
                    (iri_point.where(iri_point >= thresh["prob"]))
                    .count()
                    .item()
                )
                total_non_null = iri_point.count().item()
                fraction_above_threshold = (
                    count_above_threshold / total_non_null
                )
                quantile = iri_point.quantile(1 - thresh["frac"]).item()
                triggered = fraction_above_threshold >= thresh["frac"]
                dicts.append(
                    {
                        "iso3": country["iso3"],
                        "year": year,
                        "window": window["number"],
                        "monitoring_point": f'm{monitoring_point["forecast_month"]}l{monitoring_point["leadtime"]}',
                        "triggered": triggered,
                        "frac": fraction_above_threshold,
                        "quantile": quantile,
                    }
                )
```

```python
df = pd.DataFrame(dicts)
```

```python
1 / len(years)
```

```python
df
```

```python
iri_point.plot()
```

```python

```
