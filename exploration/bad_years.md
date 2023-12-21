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

# Bad Years

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src import utils
```

```python
filename = "ecmwf_reanalysis_score_allmonths_inseason_adm0.csv"
ec_re = pd.read_csv(utils.PROC_ECMWF_DIR / filename)
ec_re = ec_re.rename(columns={"monthly_mean": "ec_re_mean"})
```

```python
filename = "chirps_score_allmonths_inseason_adm0.csv"
ch = pd.read_csv(utils.PROC_CHIRPS_DIR / filename)
ch = ch.rename(
    columns={"normalized": "ch_mean_corrected", "mean": "ch_mean_uncorrected"}
).drop(columns=["pred", "worst_third"])
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
filename = "vhi_actual_scores_adm0.csv"
vhi_actual = (
    pd.read_csv(utils.PROC_VHI_DIR / "actual" / filename)
    .rename(columns={"weighted_mean": "vhi_actual"})
    .drop(columns=["mean", "sum", "count", "max", "month", "std", "min"])
)
```

```python
bad_years = utils.load_bad_years()
```

```python
for n in [2, 3]:
    bad_years[f"v{n}_bool"] = bad_years[f"v{n}_rank"] < 32
```

```python
for n in [1, 2, 3]:
    bad_years[f"v{n}_bool_num"] = -bad_years[f"v{n}_bool"].astype(float)
```

```python
compare = (
    bad_years.merge(ec_re, on=["ADM0_CODE", "year"])
    .merge(ch, on=["ADM0_CODE", "year"])
    .merge(vhi_actual, on=["ADM0_CODE", "year"])
    .merge(vhi_anom, on=["ADM0_CODE", "year"])
)
```

```python
compare = compare[
    [
        col
        for col in compare.columns
        if not (col.endswith("_bool") or col.endswith("_rank"))
    ]
]
```

```python
compare
```

```python
corr = compare.set_index(["year", "ADM0_CODE"]).corr(numeric_only=True)
```

```python
sns.heatmap(corr, annot=True)
```

```python
compare_all = (
    ec_re.merge(ch, on=["ADM0_CODE", "year"])
    .merge(vhi_actual, on=["ADM0_CODE", "year"])
    .merge(vhi_anom, on=["ADM0_CODE", "year"])
)
compare_all = compare_all[
    [
        col
        for col in compare_all.columns
        if not (col.endswith("_bool") or col.endswith("_rank"))
    ]
]
```

```python
for adm, group in compare_all.groupby("ADM0_CODE"):
    fig, ax = plt.subplots()
    corr = group.set_index(["year", "ADM0_CODE"]).corr(numeric_only=True)
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title(adm)
```
