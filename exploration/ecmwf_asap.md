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
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm

from src import utils
```

```python
# utils.clip_asap_inseason_month()
# utils.process_ecmwf_zscore()
utils.process_ecmwf_inseason()
```

```python
aoi = utils.load_codab(aoi_only=True)
```

```python
aoi_0 = aoi.dissolve("ADM0_CODE").reset_index()
```

```python
aoi_0
```

```python
months = range(1, 13)
years = range(1981, 2023)


dfs = []
for year in tqdm(years):
    for month in months:
        pub_date_str = f"{year}-{month:02d}-01"
        # December is missing for 2019, 2020, and 2021, so skip
        try:
            da_in = utils.load_ecmwf_inseason(pub_date_str)
        except:
            print(f"couldn't open {pub_date_str}")
            continue
        df = da_in.oap.compute_raster_stats(aoi, feature_col="ADM0_CODE")
        df["pub_date"] = pub_date_str
        dfs.append(df)

stats = pd.concat(dfs)
```

```python
stats["pub_date"] = pd.to_datetime(stats["pub_date"])
# note that for leadtime = 1, valid_date = pub_date
# this is because of ECMWF forecast indexing
stats["valid_date"] = pd.to_datetime(
    stats["pub_date"]
    + stats["leadtime"].apply(lambda x: pd.DateOffset(months=x - 1))
)
```

```python
filename = "ecmwf_inseason_adm0_rasterstats.csv"
stats.to_csv(utils.PROC_ECMWF_DIR / filename, index=False)
```

```python
stats["mean"].hist(bins=100)
```

```python
stats
```

```python
stats_f = stats[stats["valid_date"].dt.month.isin([7, 8, 9])]
```

```python
years = range(1981, 2023)
raster_col = "mean"

for adm0 in stats_f["ADM0_CODE"].unique():
    dff = stats_f[stats_f["ADM0_CODE"] == adm0]

    # any valid month, any leadtime, no consecutive
    dff.groupby(dff["valid_date"].dt.year)[raster_col].min().sort_values()

    # must be all months reporting, any leadtime
    # only Mar - Jun forecasts
    df_all = pd.DataFrame()
    for pub_month in range(3, 7):
        dfff = dff[dff["pub_date"].dt.month == pub_month]
        df_all[pub_month] = dfff.groupby(dfff["valid_date"].dt.year)[
            raster_col
        ].mean()
    df_all = df_all.min(axis=1).sort_values()
```
