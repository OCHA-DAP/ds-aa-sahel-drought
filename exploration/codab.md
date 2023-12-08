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

# CODAB

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from src import utils
```

```python
# utils.process_codab_all()
cod = utils.load_codab()
# utils.process_clip_cod_to_aoi()
cod_aoi = utils.load_codab(aoi_only=True)
```

## Plotting

Check that CODAB for entire countries includes Burkina, Niger, Chad.

```python
cod.plot()
```

Check that CODAB for Area of Interest includes:

- Burkina: Boucle du Mouhoun, Nord, Centre-Nord, Sahel
- Niger: everything below 17 N
- Chad: Lac, Kanem, Barh-El-Gazel, Batha, Wadi Fira

These are the Areas of Interest from the existing frameworks for the three countries.

```python
cod_aoi.plot()
```

```python
cod_aoi_0 = cod_aoi.dissolve("ADM0_CODE")
cod_aoi_1 = cod_aoi.dissolve("ADM1_CODE")
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
ax.axis("off")
cod_aoi_1.plot(ax=ax, column="ADM0_NAME", legend=True)
cod_aoi_1.apply(
    lambda x: ax.annotate(
        text=x["ADM1_NAME"], xy=x.geometry.centroid.coords[0], ha="center"
    ),
    axis=1,
)
```
