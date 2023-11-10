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

# LAC support

Support for ECMWF analysis in LAC. Actual processing steps are also in LAC repo.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from src import utils
```

```python
LAC_AOI_DIR = utils.DATA_DIR / "public" / "processed" / "lac"
```

```python
filename = "adm0_central_ameria_dry_corridor_simp.shp"
lac_aoi = gpd.read_file(LAC_AOI_DIR / filename)
```

```python
lac_aoi.plot()
```

```python
LAC_DIR = (
    utils.DATA_DIR
    / "private"
    / "processed"
    / "lac"
    / "ecmwf_seasonal"
    / "seas51"
    / "nc"
)
```

```python
LAC_RAW_DIR = (
    utils.DATA_DIR / "private" / "raw" / "lac" / "ecmwf_seasonal" / "seas51"
)
LAC_RAW_GRIB_DIR = LAC_RAW_DIR / "grib"
LAC_RAW_GRIB_CUR_DIR = LAC_RAW_DIR / "grib" / "current_year"
LAC_PROC_CLIP_TIF_DIR = (
    utils.DATA_DIR
    / "private"
    / "processed"
    / "lac"
    / "ecmwf_seasonal"
    / "seas51"
    / "tif"
)
```

```python
# read in GRIBs

dap_s = []
for period, period_dir in zip(
    ["2023", "lte2022"], ["current_year", "historical"]
):
    das = []
    for leadtime in range(1, 7):
        filename = f"ecmwf_forecast_{period}_lt{leadtime}.grib"
        da_in = xr.load_dataset(LAC_RAW_GRIB_DIR / period_dir / filename)
        da_in["leadtime"] = leadtime
        da_in = da_in.mean(dim=["number", "step"])
        das.append(da_in)
    dap_s.append(xr.concat(das, dim="leadtime"))

da = xr.concat(dap_s, dim="time")
da = da.sortby("time")
da = da.rio.write_crs(4326)
```

```python
# clip
da = da.rio.clip(lac_aoi.geometry, all_touched=True)
```

```python
# output TIFs
for pub_date in da["time"]:
    da_out = da.sel(time=pub_date)["tprate"]
    filename = f"ecmwf_forecast_{pub_date.dt.date.values}_aoi.tif"
    print(filename)
    da_out.rio.to_raster(LAC_PROC_CLIP_TIF_DIR / filename, driver="COG")
```

```python
da.isel(time=0, leadtime=0)["tprate"].plot()
```

```python
for pub_date in da["time"]:
    da_out = da.sel(time=pub_date)["tprate"]
    filename = f"ecmwf_forecast_{pub_date.dt.date.values}_aoi.tif"
    print(filename)
    da_out.rio.to_raster(LAC_PROC_CLIP_TIF_DIR / filename, driver="COG")
    # display(da_out)
```

```python
da_out.isel(leadtime=5).plot()
```

```python
test = rxr.open_rasterio(
    LAC_PROC_CLIP_TIF_DIR / "ecmwf_forecast_2022-04-01_aoi.tif"
)
```

```python
test.isel(band=5).plot()
```

```python
test
```

```python
LAC_AOI_DIR = utils.DATA_DIR / "public" / "processed" / "lac"
```

```python
filename = "adm0_central_ameria_dry_corridor_simp.shp"
lac_aoi = gpd.read_file(LAC_AOI_DIR / filename)
```

```python
lac_aoi.plot()
```

```python
LAC_DIR = (
    utils.DATA_DIR
    / "private"
    / "processed"
    / "lac"
    / "ecmwf_seasonal"
    / "seas51"
    / "nc"
)
```

```python
LAC_RAW_DIR = (
    utils.DATA_DIR / "private" / "raw" / "lac" / "ecmwf_seasonal" / "seas51"
)
LAC_RAW_GRIB_DIR = LAC_RAW_DIR / "grib" / "historical"
LAC_RAW_GRIB_CUR_DIR = LAC_RAW_DIR / "grib" / "current_year"
```

```python
leadtime = 2
filename = f"ecmwf_forecast_lte2022_lt{leadtime}.grib"
lac_raw = xr.load_dataset(LAC_RAW_GRIB_DIR / filename, engine="cfgrib")
lac_raw.rio.write_crs(4326, inplace=True)
lac_raw
lac_raw_clip = lac_raw.rio.clip(lac_aoi.geometry, all_touched=True)
lac_raw_clip.isel(number=0, time=0, step=0)["tprate"].plot()
```

```python
lac_raw_clip
```

```python
for leadtime in range(2, 3):
    fig, ax = plt.subplots()
    filename = f"ecmwf_forecast_lte2022_lt{leadtime}.grib"
    lac_raw = xr.load_dataset(LAC_RAW_GRIB_DIR / filename)
    lac_raw.rio.write_crs(4326, inplace=True)
    lac_raw
    lac_raw_clip = lac_raw.rio.clip(lac_aoi.geometry, all_touched=True)
    lac_raw_clip.isel(number=0, time=0)["tprate"].plot(ax=ax)
```

```python
lac_raw_clip.isel(number=0, time=0, step=0)["tprate"].plot()
```

```python
for leadtime in range(1, 5):
    fig, ax = plt.subplots()
    filename = f"ecmwf_forecast_2023_lt{leadtime}.nc"
    lac_raw = xr.load_dataset(LAC_RAW_DIR / filename)
    lac_raw.rio.write_crs(4326, inplace=True)
    lac_raw
    lac_raw_clip = lac_raw.rio.clip(lac_aoi.geometry, all_touched=True)
    lac_raw_clip.isel(number=0, time=0)["tprate"].plot(ax=ax)
```

```python
lac_raw["tprate"].isel(number=0, time=0).plot()
```

```python

```

```python

```

```python
filename = "ecmwf_total-precipitation_cadc.nc"
lac = xr.load_dataset(LAC_DIR / filename)
```

```python
lac.isel(time=0, leadtime=0)["tprate"].plot()
```

```python
lac.longitude
```

```python
filename = "clipped_ecmwf_total-precipitation_cadc.nc"
lac_clip = xr.load_dataset(LAC_DIR / filename)
```

```python
lac_clip.isel(time=0, leadtime=0)["tprate"].plot()
```

```python
test.isel(band=0)["tprate"].plot()
```

```python
leadtime = 2
filename = f"ecmwf_forecast_lte2022_lt{leadtime}.grib"
lac_raw = xr.load_dataset(LAC_RAW_GRIB_DIR / filename, engine="cfgrib")
lac_raw.rio.write_crs(4326, inplace=True)
lac_raw
lac_raw_clip = lac_raw.rio.clip(lac_aoi.geometry, all_touched=True)
lac_raw_clip.isel(number=0, time=0, step=0)["tprate"].plot()
```

```python
lac_raw_clip
```

```python
for leadtime in range(2, 3):
    fig, ax = plt.subplots()
    filename = f"ecmwf_forecast_lte2022_lt{leadtime}.grib"
    lac_raw = xr.load_dataset(LAC_RAW_GRIB_DIR / filename)
    lac_raw.rio.write_crs(4326, inplace=True)
    lac_raw
    lac_raw_clip = lac_raw.rio.clip(lac_aoi.geometry, all_touched=True)
    lac_raw_clip.isel(number=0, time=0)["tprate"].plot(ax=ax)
```

```python
lac_raw_clip.isel(number=0, time=0, step=0)["tprate"].plot()
```

```python
for leadtime in range(1, 5):
    fig, ax = plt.subplots()
    filename = f"ecmwf_forecast_2023_lt{leadtime}.nc"
    lac_raw = xr.load_dataset(LAC_RAW_DIR / filename)
    lac_raw.rio.write_crs(4326, inplace=True)
    lac_raw
    lac_raw_clip = lac_raw.rio.clip(lac_aoi.geometry, all_touched=True)
    lac_raw_clip.isel(number=0, time=0)["tprate"].plot(ax=ax)
```

```python
lac_raw["tprate"].isel(number=0, time=0).plot()
```

```python
filename = "ecmwf_total-precipitation_cadc.nc"
lac = xr.load_dataset(LAC_DIR / filename)
```

```python
lac.isel(time=0, leadtime=0)["tprate"].plot()
```

```python
lac.longitude
```

```python
filename = "clipped_ecmwf_total-precipitation_cadc.nc"
lac_clip = xr.load_dataset(LAC_DIR / filename)
```

```python
lac_clip.isel(time=0, leadtime=0)["tprate"].plot()
```

```python

```
