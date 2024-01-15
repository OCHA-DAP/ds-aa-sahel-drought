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

# BFA 2023 analysis

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import matplotlib.pyplot as plt

from src import utils
```

```python
filename = "ecmwf_reanalysis_score_allmonths_inseason_adm0.csv"
ec_re = pd.read_csv(utils.PROC_ECMWF_DIR / filename)
ec_re = ec_re.rename(columns={"monthly_mean": "ec_re_mean"})
ec_re_bfa = ec_re[ec_re["ADM0_CODE"] == "BFA"]
```

```python
ec_re_bfa
```

```python
ec_re_bfa["rank"] = ec_re_bfa["ec_re_mean"].rank()
ec_re_bfa["rp"] = len(ec_re_bfa) / ec_re_bfa["rank"]
```

```python
ec_re_bfa.set_index("year").loc[2023]
```

```python
fig, ax = plt.subplots(figsize=(10, 5))
xmin, xmax = 0, 20
ymin, ymax = ec_re_bfa["ec_re_mean"].min(), ec_re_bfa["ec_re_mean"].max()
ec_re_bfa.sort_values("rank").plot(x="rp", y="ec_re_mean", ax=ax)
ax.set_xlabel("Return period (years)")
ax.set_ylabel("Mean daily precipitation during season (mm)")
for rp in [3, 5]:
    ax.plot([rp, rp], [ymin, ymax], "k--")
    ax.annotate(f"\n {rp}-yr", (rp, ymax), ha="left", va="top")
for year in [2023]:
    precip = ec_re_bfa.set_index("year").loc[2023, "ec_re_mean"]
    rp = ec_re_bfa.set_index("year").loc[2023, "rp"]
    ax.plot(rp, precip, "r.")
    ax.annotate(f" {year} ", (rp, precip), ha="right", va="top", color="r")
ax.set_ylim(ymin, ymax)
ax.set_xlim(xmin, xmax)
ax.set_title("Burkina Faso precipitation return period - ERA5 reanalysis")
```

```python
ec_re_bfa.plot(x="year", y="ec_re_mean")
```

```python
filename = "vhi_actual_scores_adm0.csv"
vhi_actual = (
    pd.read_csv(utils.PROC_VHI_DIR / "actual" / filename)
    .rename(columns={"weighted_mean": "vhi_actual"})
    .drop(columns=["mean", "sum", "count", "max", "month", "std", "min"])
)
```

```python
filename = "vhi_anom_scores_adm0.csv"
vhi_anom = (
    pd.read_csv(utils.PROC_VHI_DIR / "anom" / filename)
    .rename(columns={"weighted_mean": "vhi_anom"})
    .drop(columns=["mean", "sum", "count"])
)
```

```python
vhi_bfa = vhi_anom[vhi_anom["ADM0_CODE"] == "BFA"]
vhi_bfa.plot(x="year", y="vhi_anom")
```

```python
vhi_bfa["rank"] = vhi_bfa["vhi_anom"].rank()
vhi_bfa["rp"] = len(vhi_bfa) / vhi_bfa["rank"]
```

```python
vhi_bfa
```

```python
fig, ax = plt.subplots(figsize=(10, 5))
xmin, xmax = 0, 20
ymin, ymax = vhi_bfa["vhi_anom"].min(), vhi_bfa["vhi_anom"].max()
vhi_bfa.sort_values("rank").plot(x="rp", y="vhi_anom", ax=ax)
ax.set_xlabel("Return period (years)")
ax.set_ylabel("Mean daily precipitation during season (mm)")
for rp in [3, 5]:
    ax.plot([rp, rp], [ymin, ymax], "k--")
    ax.annotate(f"\n {rp}-yr", (rp, ymax), ha="left", va="top")
for year in [2023]:
    precip = vhi_bfa.set_index("year").loc[2023, "vhi_anom"]
    rp = vhi_bfa.set_index("year").loc[2023, "rp"]
    ax.plot(rp, precip, "r.")
    ax.annotate(f" {year} ", (rp, precip), ha="right", va="top", color="r")
ax.set_ylim(ymin, ymax)
ax.set_xlim(xmin, xmax)
ax.set_title("Burkina Faso precipitation return period")
```

```python

```
