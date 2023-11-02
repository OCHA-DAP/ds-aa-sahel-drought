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
import xarray as xr
import matplotlib.pyplot as plt

from src import utils
```

```python
utils.download_ecmwf()
```

```python
aoi = utils.load_codab(aoi_only=True)

aoi.total_bounds
```

```python
# sample_path = "/Users/tdowning/Downloads/adaptor.mars.external-1697662778.5430737-11636-3-c03f3856-3c8b-4132-9868-b9ad7cc43b0d.grib"
sample_path = "/Users/tdowning/Downloads/adaptor.mars.external-1698791926.0763922-27285-14-ea4e3d6b-d1cf-4ae1-9acc-27cd8df27938.grib"
ds = xr.load_dataset(sample_path, engine="cfgrib")
display(ds)
```

```python
da = ds["tprate"]
da = da.assign_coords(longitude=((da.longitude - 180) % 360) - 180).sortby(
    "longitude"
)
da.rio.write_crs(4326, inplace=True)

da_clip = da.rio.clip(aoi.geometry, all_touched=True)

fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
aoi.boundary.plot(linewidth=0.2, ax=ax, color="black")
da_clip.isel(time=0, step=2).plot(ax=ax)
```

```python
da_clip.step.dt.days / 30
```

```python
ds.valid_time.isel(time=0, step=2)
```
