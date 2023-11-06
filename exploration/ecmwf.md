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

# ECMWF

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path
import datetime

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
# utils.download_ecmwf()
# utils.process_ecmwf()
```

```python
aoi = utils.load_codab(aoi_only=True)
```

## Process data

Calculate the rank and zscore for `tprate`. This is done over `valid_month`
(i.e. the month that the forecast is aiming to predict),
`latitude` and `longitude`.
This is because we want to compare the forecast against the climatology,
which is by pixel and by month.

This is done in `pandas` instead of `xarray` because aggregating over
non-dimensions (i.e. `valid_month`) is kind of tricky in `xarray` but much
easier in `pandas`.

The `valid_time` is also set in the Dataset, although this implementation isn't
ideal as we're just adding months as seconds.
Again, kind of tricky to do in `xarray`.

```python
da = utils.load_ecmwf()
ds = da.to_dataset()
da["valid_month"] = (da["time"].dt.month + da["leadtime"] - 1) % 12 + 1
df = da.to_dataframe().reset_index()
df["valid_time"] = df.apply(
    lambda row: row["time"] + pd.DateOffset(months=row["leadtime"]), axis=1
)
df_groupby = df.groupby(["valid_month", "latitude", "longitude"])["tprate"]
df["rank"] = df_groupby.rank().astype("float32")
df["zscore"] = df_groupby.transform(lambda x: zscore(x))
# assign back to variable in Dataset
# groupby in this case is just to set the index
ds[["rank", "zscore"]] = (
    df.groupby(["leadtime", "time", "latitude", "longitude"])
    .mean()
    .to_xarray()[["rank", "zscore"]]
)
ds = ds.assign_coords(
    {"valid_time": ds["time"] + ds["leadtime"] * 31 * 24 * 3600 * 1000000000}
)
rank = ds["rank"]
rank.rio.write_crs(4326, inplace=True)
ds
```

```python
pctile = (rank - 1) / (rank.max() - 1)
pctile.name = "percentile"
```

## Visualize

We can have a look at the spread of zscores.
As expected, there is a longer tail on the high end because precipitation
can't be negative.

```python
ds["zscore"].plot(bins=20)
```

We can also look at bias.
It looks like ECMWF forecasts are biased by leadtime,
because the precipitation at shorter leadtimes is higher
(by quite a bit, like 50%!).

```python
df.groupby("leadtime")["tprate"].mean().plot()
```

This is somewhat consistent across valid months too.

```python
df.groupby(["valid_month", "leadtime"]).mean().groupby("valid_month")[
    "tprate"
].transform(lambda x: x / x.mean()).reset_index().pivot_table(
    columns="valid_month", index="leadtime"
).plot(
    cmap="Paired"
)
```

And across years.

```python
df.groupby([df["time"].dt.year, "leadtime"])[
    "norm_month"
].mean().reset_index().pivot_table(
    index="time", columns="leadtime", values="norm_month"
).plot(
    xlabel="year",
    ylabel="Forecasted precipitation,\nnormalized by valid month",
)
```

We can see that the shape of the leadtime bias is similar across decades,
which seems to suggest that this is not due to model drift?

```python
df["norm_month"] = df.groupby("valid_month")["tprate"].transform(
    lambda x: (x - x.mean()) / x.std()
)
fig, ax = plt.subplots()
for d in range(1980, 2021, 10):
    years = range(d, d + 11)
    dff = df[df["time"].dt.year.isin(years)]
    dff.groupby("leadtime")["norm_month"].mean().plot(ax=ax, label=f"{d}'s")

ax.set_ylabel("Forecasted precipitation,\nnormalized by valid month")
ax.set_xlabel("Leadtime (months)")
ax.legend(loc="upper right")
```

We can also have a look at examples of this by plotting the forecasts leading
up to a certain month. For many specific months, the forecast at `leadtime=1`
is wetter than the other ones.

```python
valid_month = 8
valid_year = 2020

valid_date = pd.to_datetime(datetime.date(valid_year, valid_month, 1))
leadtimes = range(1, 7)
forecast_dates = [valid_date - pd.DateOffset(months=x) for x in leadtimes]

for leadtime, forecast_date in zip(leadtimes, forecast_dates):
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.axis("off")
    aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
    ds["zscore"].sel(time=forecast_date, leadtime=leadtime).plot(
        ax=ax, vmin=-3, vmax=3, cmap="BrBG"
    )
    ax.set_title(
        f"Produced: {forecast_date:%B %Y}; "
        f"Leadtime: {leadtime} months; "
        f"Valid: {valid_date:%B %Y}"
    )
```

## Process by leadtime

We can correct for this bias by instead calculating the rank and zscore by
`leadtime` as well as `valid_month`, `latitude`, and `longitude`.

```python
df_groupby = df.groupby(["leadtime", "valid_month", "latitude", "longitude"])[
    "tprate"
]
df["rank_lt"] = df_groupby.rank().astype("float32")
df["zscore_lt"] = df_groupby.transform(lambda x: zscore(x))
# assign back to variable in Dataset
# groupby in this case is just to set the index
ds[["rank_lt", "zscore_lt"]] = (
    df.groupby(["leadtime", "time", "latitude", "longitude"])
    .mean()
    .to_xarray()[["rank_lt", "zscore_lt"]]
)
```

This means that we now have a zscore that is not biased by leadtime
(note y-axis scale). Now we can treat all leadtimes the same way.

```python
for x in ["zscore", "zscore_lt"]:
    fig, ax = plt.subplots()
    ds[x].mean(dim=["latitude", "longitude", "time"]).plot(ax=ax)
```

Here we can just verify again that using `zscore_lt` properly normalizes the
distribution.
The first plot shows `zscore` with the lower leadtimes skewing higher.
The second plot shows all leadtimes with the same distribution for `zscore_lt`.

```python
for zscore in ["zscore", "zscore_lt"]:
    fig, ax = plt.subplots()
    for lt in range(1, 7):
        df[df["leadtime"] == lt][zscore].hist(
            histtype="step", bins=100, label=lt, ax=ax
        )
    ax.legend(title="Leadtime")
    ax.set_xlabel(zscore)
```

```python
valid_month = 8
valid_year = 2020

valid_date = pd.to_datetime(datetime.date(valid_year, valid_month, 1))
leadtimes = range(1, 7)
forecast_dates = [valid_date - pd.DateOffset(months=x) for x in leadtimes]

for leadtime, forecast_date in zip(leadtimes, forecast_dates):
    fig, ax = plt.subplots(figsize=(25, 5))
    ax.axis("off")
    aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
    ds["zscore_lt"].sel(time=forecast_date, leadtime=leadtime).plot(
        ax=ax, vmin=-3, vmax=3, cmap="BrBG"
    )
    ax.set_title(
        f"Produced: {forecast_date:%B %Y}; "
        f"Leadtime: {leadtime} months; "
        f"Valid: {valid_date:%B %Y}"
    )
```

Now that we've normalized by leadtime, we can compare the forecasts from
different leadtimes for the same valid time.
Each point in the plot below is one valid time.
There is more disagreement in the forecasts (high `std`) for relatively
high precipitation forecasts (high `mean`).

```python
df.groupby("valid_time")["zscore_lt"].agg(["mean", "std"]).reset_index().plot(
    x="mean",
    y="std",
    linestyle="None",
    marker=".",
    ylabel="Std of forecasts for specific valid time",
    xlabel="Mean of forecasts for specific valid time",
)
```

The disagreement between forecasts does not really vary over time.

```python
df.groupby("valid_time")["zscore_lt"].agg(["mean", "std"]).reset_index().plot(
    x="valid_time",
    y="std",
    linestyle="None",
    marker=".",
    ylabel="Std of forecasts for specific valid time",
    xlabel="Valid time",
)
```

## Raster stats

```python
z_stats = (
    ds["zscore_lt"]
    .rio.write_crs(4326)
    .oap.compute_raster_stats(gdf=aoi, feature_col="ADM0_CODE")
)
z_stats["valid_time"] = z_stats.apply(
    lambda row: row["time"] + pd.DateOffset(months=row["leadtime"]), axis=1
)
```

We can crudely look at the average forecast for a subset of relevant months
(`rel_months`) for each year. By grouping by `z_stats["valid_time"].dt.year`,
we just take the average (normalized) forecast for those months
(averaged over all leadtimes and relevant months).

Note that in reality, we wouldn't trigger this way.
To get all the forecasts to average over,
we would need to wait until the month before the final relevant month,
which isn't very "anticipatory".

```python
rel_months = [7, 8, 9]
rel_years = range(1998, 2100)
plot_lts = False

total_years = z_stats[z_stats["valid_time"].dt.year.isin(rel_years)][
    "valid_time"
].dt.year.unique()

for adm0 in z_stats["ADM0_CODE"].unique():
    fig, ax = plt.subplots()
    dff = z_stats[
        (z_stats["ADM0_CODE"] == adm0)
        & (z_stats["valid_time"].dt.month.isin(rel_months))
        & (z_stats["valid_time"].dt.year.isin(rel_years))
    ]
    dff_group = (
        dff.groupby(z_stats["valid_time"].dt.year)
        # .agg(["min", "max"])
    )

    dff_group.boxplot(column="mean", subplots=False, rot=90, ax=ax)

    ax.set_title(adm0)
    if plot_lts:
        fig, ax = plt.subplots()
        for leadtime in range(1, 7):
            dff[dff["leadtime"] == leadtime].plot(
                ax=ax,
                label=leadtime,
                x="valid_time",
                y="mean",
                linestyle="None",
                marker=".",
            )
    # dff["valid_time", "mean"].plot(ax=ax)
    print(
        f"{adm0} worst third of years, {min(total_years)} to {max(total_years)}"
    )
    display(
        dff_group["mean"]
        .mean()
        .reset_index()
        .sort_values("mean")[["valid_time", "mean"]][
            : int(len(total_years) / 3)
        ]
    )
```

```python
filename = "ecmwf_total-precipitation_sah_zscore.nc"
ds.to_netcdf(utils.PROC_ECMWF_DIR / filename)
```
