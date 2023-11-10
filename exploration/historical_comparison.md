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

# Historical TPR etc

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src import utils
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

```python
filename = "chirps_score_allmonths_inseason_adm0.csv"
ch = pd.read_csv(utils.PROC_CHIRPS_DIR / filename)
ch = ch.rename(
    columns={"normalized": "ch_mean_corrected", "mean": "ch_mean_uncorrected"}
).drop(columns=["pred"])
```

```python
ch
```

```python
filename = "ecmwf_inseason_adm0_rasterstats.csv"
ec = pd.read_csv(
    utils.PROC_ECMWF_DIR / filename, parse_dates=["pub_date", "valid_date"]
)
ec = ec[ec["valid_date"].dt.month.isin([7, 8, 9])]
```

```python
ec
```

```python
def is_in_quantile(group, q, col):
    group[f"under_{q}"] = group[col] <= group[col].quantile(q)
    return group
```

```python
# any valid month, no consecutive


ec_year = (
    ec.groupby(["ADM0_CODE", ec["valid_date"].dt.year])["mean"]
    .min()
    .reset_index()
    .rename(columns={"valid_date": "year", "mean": "ec_mean"})
)
ec_year = (
    ec_year.groupby("ADM0_CODE")
    .apply(lambda x: is_in_quantile(x, 0.33, "ec_mean"))
    .reset_index(drop=True)
)
```

```python
ec_year
```

```python
ec_year_lt = (
    ec.groupby(["ADM0_CODE", ec["valid_date"].dt.year, "leadtime"])["mean"]
    .min()
    .reset_index()
    .rename(columns={"valid_date": "year", "mean": "ec_mean"})
)
```

```python
ec
```

```python
dfs = []
for (adm, year), group in ec.groupby(["ADM0_CODE", ec["valid_date"].dt.year]):
    for leadtime in range(1, 7):
        dff = group[group["leadtime"] <= leadtime]
        x = dff["mean"].min()
        df_add = pd.DataFrame(
            {
                "ADM0_CODE": adm,
                "year": year,
                "ec_mean": x,
                "leadtime": leadtime,
            },
            index=[0],
        )
        dfs.append(df_add)

ec_year_lt = pd.concat(dfs, ignore_index=True)
```

```python
years = range(1981, 2023)

compare = ch.merge(ec_year_lt, on=["ADM0_CODE", "year"])
```

```python
compare
```

```python
# Group by 'ADM0_CODE' and 'leadtime' and calculate ROC curve for each group
roc_auc_values = []

# Get unique ADM0_CODE values
unique_adm_codes = compare["ADM0_CODE"].unique()

# Create subplots
fig, axs = plt.subplots(
    len(unique_adm_codes), 1, figsize=(10, 15), sharex=True, sharey=True
)
fig.suptitle(
    "Receiver Operating Characteristic (ROC) Curve by ADM0_CODE", y=1.02
)

for i, adm_code in enumerate(unique_adm_codes):
    ax = axs[i]
    adm_group = compare[compare["ADM0_CODE"] == adm_code]

    for leadtime, group in adm_group.groupby("leadtime"):
        fpr, tpr, thresholds = roc_curve(
            group["worst_third"], -group["ec_mean"]
        )
        roc_auc = auc(fpr, tpr)
        roc_auc_values.append(((adm_code, leadtime), roc_auc))

        # Plot ROC curve for each group
        ax.plot(
            fpr, tpr, lw=2, label=f"Leadtime {leadtime} (AUC = {roc_auc:.2f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Guess",
    )
    ax.set_title(f"ADM0_CODE {adm_code}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

# Set axes limits
for ax in axs:
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

# Display AUC values
# print("AUC values by ADM0_CODE and Leadtime:")
# for (adm_code, leadtime), auc_value in roc_auc_values:
#     print(f"{adm_code}, Leadtime {leadtime}: {auc_value:.2f}")

plt.tight_layout()
plt.show()
```

```python
thresholds
```
