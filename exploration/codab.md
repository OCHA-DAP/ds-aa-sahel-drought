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
from ochanticipy import CodAB, create_country_config

from src import utils
```

```python
# utils.process_codab_all()
cod = utils.load_codab_all()
```

```python
cod
```

```python
cod.plot()
```

```python
cod[cod["ADM0_CODE"] == "NER"].plot()
```
