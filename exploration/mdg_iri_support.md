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

# Madagascar IRI historical analysis

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import cftime
import geopandas as gpd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from ochanticipy import (
    GeoBoundingBox,
    IriForecastProb,
    IriForecastDominant,
    CodAB,
    create_country_config,
    create_custom_country_config,
)
from tqdm.notebook import tqdm

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
level = 1
load_dir = DATA_DIR / "public/raw/glb/asap/reference_data"
filename = f"gaul{level}_asap_v04.zip"
zip_path = "zip:/" / load_dir / filename
cod = gpd.read_file(zip_path)
mdg = cod[cod["name0"] == "Madagascar"]
```

```python
mdg
```

```python
country_config = create_custom_country_config("../mdg.yaml")
geobb = GeoBoundingBox.from_shape(mdg)

iri_prob = IriForecastProb(
    country_config=country_config, geo_bounding_box=geobb
)
```

```python
# iri_prob.download()
# iri_prob.process()
da_all = iri_prob.load()
iri = da_all["prob"].isel(C=0)
```

```python
iri
```

```python
df_ins = []
percentiles = range(10, 100, 20)

for F in tqdm(iri.F.values):
    da = iri.sel(F=F)
    df_in = da.oap.compute_raster_stats(
        gdf=mdg, feature_col="name1", percentile_list=percentiles
    )
    df_in["F"] = F
    df_ins.append(df_in)

stats = pd.concat(df_ins, ignore_index=True)
```

```python
# stats["F"] = pd.to_datetime(stats["F"])
stats["rel_month1"] = stats["F"].apply(lambda x: x.month) + stats["L"].astype(
    int
)
stats["rel_month1"] = stats["rel_month1"].apply(
    lambda x: x if x < 13 else x - 12
)
stats["F_year"] = stats["F"].apply(lambda x: x.year)
```

```python
stats
```

```python
stats[
    (stats["F_year"] == 2023) & (stats["rel_month1"] == 10) & (stats["L"] == 1)
]
```

```python
frac_reporting_threshold = 0.5
rel_months = [10]
total_months = [x for x in range(min(rel_months), max(rel_months) + 3)]
plot_quants = [10, 50, 90]
plot_quant_cols = [
    f"{100 - x}quant" for x in percentiles if x in plot_quants
] + ["mean"]
plot_quants = ["mean"]
plot_quant_cols = ["mean"]

rel_stats = stats[stats["rel_month1"].isin(rel_months)]
max_per_year = (
    rel_stats.groupby(["name1", "F_year"])[plot_quant_cols].max().reset_index()
)

f_mon = ["Jun", "Jul", "Aug", "Sep"]
f_mon_L_num = [4, 3, 2, 1]

oct_fcast = stats[(stats["F_year"] == 2023) & (stats["rel_month1"] == 10)]

for adm0 in max_per_year["name1"].unique():
    if adm0 != "Anosy":
        pass
    title_text = f"Relevant months: OND"
    dff = max_per_year[max_per_year["name1"] == adm0][plot_quant_cols]
    for col in dff.columns:
        dff[col] = dff[col].sort_values().values
    dff["return_period"] = [7 / x for x in range(7, 0, -1)]
    fig, ax = plt.subplots()
    dff.plot(x="return_period", ax=ax)
    ax.set_title(
        f"{adm0}: IRI low tercile return period (since 2017)\n"
        f"All leadtimes, {title_text}"
    )
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Prob. of low tercile")
    for L in f_mon_L_num:
        oct_fcast_mean = oct_fcast[
            (oct_fcast["name1"] == adm0) & (oct_fcast["L"] == L)
        ]["mean"]
        oct_fcast_rp = np.interp(
            oct_fcast_mean, dff["mean"], dff["return_period"]
        )
        # print(oct_fcast_rp)
        ax.plot([oct_fcast_rp], [oct_fcast_mean.values[0]], marker=".", l)
    ax.legend(
        title="Mean prob.",
        labels=["historical", *[f"{x} 2023 forecast" for x in f_mon]],
    )
```

```python

```
