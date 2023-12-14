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

```

```python
test
```

```python
fig, ax = plt.subplots(figsize=(25, 5))
aoi.boundary.plot(ax=ax, linewidth=0.5, color="k")
ax.axis("off")
test.plot(ax=ax)
```

```python
s = utils.load_asap_inseason(interval="dekad", number=1)
```

```python
s
```
