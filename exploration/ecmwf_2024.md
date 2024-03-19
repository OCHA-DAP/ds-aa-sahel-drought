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

# ECMWF 2024

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from src import utils
```

```python

```

```python
adm = utils.load_codab(aoi_only=True)
```

```python
adm_bfa = adm[adm["ADM0_CODE"] == "BFA"]
```

```python
adm_bfa
```

```python
# utils.download_ecmwf(start_year=2023, end_year=2023)
```

```python
# utils.download_ecmwf(start_year=2024, end_year=2024)
```

```python
utils.process_ecmwf()
```

```python
ec_all = utils.load_ecmwf()
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
for iso in ["BFA", "TCD"]:
    if iso == "BFA":
        leadtimes = [4, 5, 6]
        monthsname = "juin-juillet-août"
        fullname = "Burkina Faso"
    else:
        leadtimes = [5, 6]
        monthsname = "juillet-août"
        fullname = "Tchad"

    adm_bfa = adm[adm["ADM0_CODE"] == iso]

    ec_bfa = ec_all.sel(leadtime=leadtimes).sel(
        time=(ec_all["time"].dt.month == 3)
    )
    ec_bfa = upsample_dataarray(ec_bfa)
    ec_bfa = ec_bfa.rio.clip(adm_bfa.geometry)
    ec_bfa = ec_bfa.groupby("time.year").sum().sum(dim="leadtime")
    ec_bfa = ec_bfa.where(ec_bfa != 0, np.nan)
    ec_bfa = ec_bfa * 3600 * 24 * 1000 * 30

    fig, ax = plt.subplots(figsize=(10, 5))
    adm_bfa.boundary.plot(ax=ax, color="white", linewidth=0.5)
    ec_bfa.sel(year=2024).plot(
        ax=ax, cbar_kwargs={"label": "Précipitations totales prévues (mm)"}
    )
    ax.axis("off")
    ax.set_title(
        f"Prévisions ECMWF 2024 {fullname}\n"
        f"mois de publication: mars, période d'interêt: {monthsname}"
    )

    df_bfa = (
        ec_bfa.mean(dim=["latitude", "longitude"])
        .to_dataframe()["tprate"]
        .reset_index()
    )

    thresh = df_bfa["tprate"].quantile(1 / 3)

    fig, ax = plt.subplots()
    df_bfa.plot(x="year", y="tprate", ax=ax, legend=False, linewidth=1)
    ax.plot([2024], [df_bfa.iloc[-1]["tprate"]], ".r")
    ax.annotate(
        " 2024",
        xy=(2024, df_bfa.iloc[-1]["tprate"]),
        color="red",
        ha="left",
        va="center",
    )
    ax.axhline(y=thresh, color="grey", linestyle="--")
    ax.annotate(
        " seuil\n 1-an-sur-3",
        xy=(2026, thresh),
        color="grey",
        ha="left",
        va="center",
    )
    for year, row in df_bfa.set_index("year").iterrows():
        tp = row["tprate"]
        if tp < thresh:
            ax.annotate(
                year,
                xy=(year, tp),
                color="grey",
                ha="center",
                va="top",
            )
    ax.set_xlabel("Année")
    ax.set_ylabel("Précipitations totales prévues (mm)")
    ax.set_title(
        f"Prévisions ECMWF historiques {fullname}\n"
        f"mois de publication: mars, période d'interêt: {monthsname}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
```

```python

```
