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

# VSI

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import datetime
import os
from io import BytesIO

import rioxarray as rxr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
import statsmodels.api as sm
from owslib.wms import WebMapService
from shapely.geometry import box
from tqdm.notebook import tqdm
from statsmodels.tsa.stattools import adfuller

from src import utils
```

```python
utils.download_vhi()
```

```python
utils.process_vhi()
```

```python
raw_files = os.listdir(utils.RAW_VHI_DIR)

das = []
for raw_filename in tqdm(raw_files):
    year, month = (
        raw_filename.removeprefix("VHI_M_").removesuffix(".tif").split("-")
    )

    da_in = rxr.open_rasterio(utils.RAW_VHI_DIR / raw_filename).squeeze(
        drop=True
    )
    da_in = da_in.rename({"x": "longitude", "y": "latitude"})
    da_in["date"] = datetime.datetime(int(year), int(month), 1)
    da_in = da_in.fillna(255)
    da_in = da_in.astype("uint8")
    das.append(da_in)

da = xr.concat(das, dim="date")
da = da.sortby("date")
```

```python
ds = da.to_dataset(name="vhi")
```

```python
vhi = xr.load_dataarray(utils.PROC_VHI_DIR / "VHI_M_all_sah.nc")
```

```python
vhi = vhi.astype("uint8")
```

```python
vhi
```

```python
vhi_fit = vhi.polyfit(dim="date", deg=1)
```

```python
vhi_fit
```

```python
vhi_fit.sel(degree=0)["polyfit_coefficients"].plot()
```

```python
vhi_fit.sel(degree=1)["polyfit_coefficients"].plot()
```

```python
vhi_trend = vhi_fit.sel(degree=0) + vhi_fit.sel(degree=1) * vhi["date"].astype(
    int
)
```

```python
vhi_trend.mean(dim=["latitude", "longitude"])["polyfit_coefficients"].plot()
```

```python
vhi_trend
```

```python
vhi
```

```python
vhi_ds = vhi.to_dataset(name="raw_vhi")
```

```python
vhi_ds["vhi_trend"] = vhi_fit.sel(degree=1)["polyfit_coefficients"]
vhi_ds["vhi_intercept"] = vhi_fit.sel(degree=0)["polyfit_coefficients"]
```

```python
vhi_ds["vhi_pred"] = vhi_trend["polyfit_coefficients"]
```

```python
vhi_ds["vhi_anomaly"] = vhi_ds["raw_vhi"] - vhi_ds["vhi_pred"]
```

```python
vhi_ds["vhi_anomaly"].drop_vars(["degree"]).to_netcdf(
    utils.PROC_VHI_DIR / "VHI_M_all_anomaly_sah.nc"
)
```

```python
vhi_anom = xr.load_dataarray(utils.PROC_VHI_DIR / "VHI_M_all_anomaly_sah.nc")
```

```python
vhi_anom
```

```python
utils.process_vhi_anom_inseason()
```

```python
vhi_anom_mean = vhi_anom.mean(dim=["longitude", "latitude"])
```

```python
vhi_anom_mean.plot()
```

```python
print(
    sm.OLS(vhi_mean_df["mean"], sm.add_constant(vhi_mean_df["index"]))
    .fit()
    .summary()
)
```

```python
vhi
```

```python
utils.process_vhi_inseason()
```

```python
utils.calculate_vhi_raster_stats(variable="anom")
```

```python
filename = "vhi_anom_inseason_adm0_rasterstats.csv"
anom_stats = pd.read_csv(utils.PROC_VHI_DIR / "anom" / filename)
anom_stats["date"] = pd.to_datetime(
    anom_stats.apply(
        lambda row: datetime.date(row["year"], row["month"], 1), axis=1
    )
)
anom_scores = (
    anom_stats.groupby(["ADM0_CODE", "year"])
    .mean()[["mean", "sum", "count"]]
    .reset_index()
)
anom_scores["weighted_mean"] = anom_scores["sum"] / anom_scores["count"]
anom_scores = anom_scores.sort_values("year")
anom_scores
```

```python
anom_scores.pivot(
    index="year", columns="ADM0_CODE", values="weighted_mean"
).plot()
```

```python
filename = "vhi_anom_scores_adm0.csv"
anom_scores.to_csv(utils.PROC_VHI_DIR / "anom" / filename, index=False)
```

```python
filename = "vhi_inseason_adm0_rasterstats.csv"
stats = pd.read_csv(utils.PROC_VHI_DIR / filename)
stats["date"] = pd.to_datetime(
    stats.apply(
        lambda row: datetime.date(row["year"], row["month"], 1), axis=1
    )
)
```

```python
stats
```

```python
stats.pivot(index="date", columns="ADM0_CODE", values="mean").plot()
```

```python
scores = stats.groupby(["ADM0_CODE", "year"]).mean().reset_index()
scores["weighted_mean"] = scores["sum"] / scores["count"]
filename = "vhi_actual_scores_adm0.csv"
scores.to_csv(utils.PROC_VHI_DIR / "actual" / filename, index=False)
```

```python
for adm, group in scores.groupby("ADM0_CODE"):
    result = adfuller(group["weighted_mean"])
    print(f"{adm}: {result[1]}")
```

```python
for adm, group in scores.groupby("ADM0_CODE"):
    print(adm)
    print(
        sm.OLS(group["weighted_mean"], sm.add_constant(group["year"]))
        .fit()
        .summary()
    )
```

```python
sns.lmplot(scores, x="year", col="ADM0_CODE", y="weighted_mean")
```

```python
sns.lmplot(anom_scores, x="year", col="ADM0_CODE", y="weighted_mean")
```
