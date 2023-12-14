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
import os
from io import BytesIO

import rioxarray as rxr
import matplotlib.pyplot as plt
import pandas as pd
from owslib.wms import WebMapService
from shapely.geometry import box
from tqdm.notebook import tqdm

from src import utils
```

```python
utils.download_vhi()
```

```python
utils.process_vhi_inseason()
```

```python
utils.calculate_vhi_raster_stats()
```

```python
filename = "vhi_inseason_adm0_rasterstats.csv"
stats = pd.read_csv(utils.PROC_VHI_DIR / filename)
```

```python
stats.pivot(index="")
```
