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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.exceptions import UndefinedMetricWarning

from src import utils
```

```python
def calc_quantile(df, q, col, agg_cols):
    df = (
        df.groupby(agg_cols)
        .apply(lambda g: is_in_quantile(g, q, col))
        .reset_index(drop=True)
    )
    return df


def is_in_quantile(group, q, col):
    q_str = int(q * 100)
    group[f"{col}_q{q_str}"] = group[col] <= group[col].quantile(q)
    return group


def calc_zscore(group, col, negative: bool = False):
    group[f"{col}_zscore"] = (group[col] - group[col].mean()) / group[
        col
    ].std()
    if negative:
        group[f"{col}_zscore"] *= -1
    return group


def highlight_true(s):
    return ["background-color: red" if v else "" for v in s]
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
```

## Load data

### CHIRPS

```python
filename = "chirps_score_allmonths_inseason_adm0.csv"
ch = pd.read_csv(utils.PROC_CHIRPS_DIR / filename)
ch = ch.rename(
    columns={"normalized": "ch_mean_corrected", "mean": "ch_mean_uncorrected"}
).drop(columns=["pred"])
```

### ECMWF reanalysis

```python
filename = "ecmwf_reanalysis_score_allmonths_inseason_adm0.csv"
ec_re = pd.read_csv(utils.PROC_ECMWF_DIR / filename)
ec_re = ec_re.rename(columns={"monthly_mean": "ec_re_mean"})
```

### ECMWF

```python
filename = "ecmwf_inseason_adm0_rasterstats.csv"
ec = pd.read_csv(
    utils.PROC_ECMWF_DIR / filename, parse_dates=["pub_date", "valid_date"]
)
ec = ec[ec["valid_date"].dt.month.isin([7, 8, 9])]
ec = ec.rename(
    columns={"mean": "ec_mean", "sum": "ec_sum", "count": "ec_count"}
).drop(columns=["std", "min", "max"])
```

```python
ec.groupby(ec["valid_date"].dt.year)["ec_mean"].min().plot()
```

### IRI

```python
iri = utils.load_iri_inseason_stats()
iri = iri[iri["rel_month1"] == 7]
cols = ["mean", "L", "F", "rel_month1", "F_year", "ADM0_CODE"]
iri = iri[cols]
iri = iri.rename(
    columns={"mean": "low_ter_prob_mean", "F_year": "year", "L": "leadtime"}
)
iri["rel_month1"] = (iri["rel_month1"] - 1) % 12 + 1
iri = (
    iri.groupby(["leadtime", "ADM0_CODE"])
    .apply(lambda g: calc_zscore(g, "low_ter_prob_mean", negative=True))
    .reset_index(drop=True)
)
iri = iri.rename(columns={"low_ter_prob_mean_zscore": "iri_mean"})
```

## Process data

### IRI process

#### Specific leadtimes

Comparing the performance of each leadtime individually.

```python
iri_year_leadtime_eq = (
    iri.groupby(["ADM0_CODE", "year", "leadtime"])["iri_mean"]
    .max()
    .reset_index()
)
iri_year_leadtime_eq = calc_quantile(
    iri_year_leadtime_eq, 1 / 3, "iri_mean", ["ADM0_CODE", "leadtime"]
)
```

### ECMWF process

#### Leadtime LTE

Taking any leadtimes `<=` cutoff value.

E.g. `leadtime = 4` corresponds to only considering forecasts with `leadtime <= 4`

```python
dfs = []
for leadtime in range(1, 7):
    dff = ec[ec["leadtime"] <= leadtime]
    grouped = (
        dff.groupby(["ADM0_CODE", dff["valid_date"].dt.year])["ec_mean"]
        .min()
        .reset_index()
        .rename(columns={"valid_date": "year"})
    )
    grouped["leadtime"] = leadtime
    dfs.append(grouped)

ec_year_leadtime_lte = pd.concat(dfs, ignore_index=True)
ec_year_leadtime_lte = (
    ec_year_leadtime_lte.groupby(["ADM0_CODE", "leadtime"])
    .apply(lambda g: is_in_quantile(g, 0.33, "ec_mean"))
    .reset_index(drop=True)
)
```

#### Specific leadtimes EC

Comparing the performance of each leadtime individually.

```python
ec_year_leadtime_eq = (
    ec.groupby(["ADM0_CODE", ec["valid_date"].dt.year, "leadtime"])["ec_mean"]
    .min()
    .reset_index()
    .rename(columns={"valid_date": "year"})
)
for q in [1 / 2, 1 / 3]:
    ec_year_leadtime_eq = calc_quantile(
        ec_year_leadtime_eq, q, "ec_mean", ["ADM0_CODE", "leadtime"]
    )
```

#### By the whole season

Taking only forecasts for which the whole season is forecasted (Jul, Aug, Sep).
So, this includes only forecasts published in Apr, May, Jun, Jul.

```python
pub_months = [4, 5, 6, 7]
ec_year_season_lt = (
    ec[ec["pub_date"].dt.month.isin(pub_months)]
    .groupby(["ADM0_CODE", ec["valid_date"].dt.year, ec["pub_date"].dt.month])[
        ["ec_sum", "ec_count"]
    ]
    .sum()
    .reset_index()
    .rename(columns={"valid_date": "year", "pub_date": "pub_month"})
)
ec_year_season_lt["ec_season_mean"] = (
    ec_year_season_lt["ec_sum"] / ec_year_season_lt["ec_count"]
)
ec_year_season_lt["season_lt"] = 8 - ec_year_season_lt["pub_month"]
ec_year_season_lt = calc_quantile(
    ec_year_season_lt, 1 / 3, "ec_season_mean", ["ADM0_CODE", "season_lt"]
)
ec_year_season_lt
```

## Evaluate performance

### Merge

Merge dataframes for comparison

```python
years = range(1981, 2023)

compare = ec_re.merge(ec_year_season_lt, on=["ADM0_CODE", "year"])
# compare = ec_re.merge(ec_year_leadtime_eq, on=["ADM0_CODE", "year"])
# compare = ch.merge(ec_year_leadtime_eq, on=["ADM0_CODE", "year"])
# compare = ch.merge(iri_year_leadtime_eq, on=["ADM0_CODE", "year"])
# compare = calc_quantile(compare, 1 / 3, "ch_mean_corrected", ["ADM0_CODE"])
compare = calc_quantile(compare, 1 / 3, "ec_re_mean", ["ADM0_CODE"])
```

```python
compare
```

```python
col = "ec_re_mean_q33"
actual = (
    compare.groupby(["ADM0_CODE", "year"])[col]
    .first()
    .reset_index()
    .pivot(columns="ADM0_CODE", index="year")
    .swaplevel(axis=1)
    .rename(columns={col: "actual"})
)
pred_col = "ec_mean_q33"

pred = compare.pivot(
    columns=["ADM0_CODE", "leadtime"], values=[pred_col], index="year"
)[pred_col]
```

### Calculate performance metrics

For fixed threshold (e.g. 33% quantile, correponding to 3-year return period of trigger).
Note that when trigger is adjusted to have the same return period as the actual event,
the precision, recall, and F1 score will always be equal to each other.

```python
actual_name = "ec_re_mean"
pred_name = "ec_season_mean"
q_pred = 33
leadtime_name = "season_lt"

actual_bool = f"{actual_name}_q33"
pred_bool = f"{pred_name}_q{q_pred}"

dfs = []
for (lt, adm), group in compare.groupby([leadtime_name, "ADM0_CODE"]):
    class_rep = classification_report(
        group[actual_bool], group[pred_bool], output_dict=True
    ).get("True")
    corr = group[actual_name].corr(group[pred_name])
    df_add = pd.DataFrame(class_rep, index=[0])
    df_add[[leadtime_name, "ADM0_CODE", "corr"]] = lt, adm, corr
    dfs.append(df_add)

scores = pd.concat(dfs, ignore_index=True)

metric = "precision"
scores.pivot_table(
    index=leadtime_name, columns="ADM0_CODE", values=metric
).plot(title=f"{metric} between {actual_name} and {pred_name}, q={q_pred}")

metric = "corr"
scores.pivot_table(
    index=leadtime_name, columns="ADM0_CODE", values=metric
).plot(title=f"{metric} between {actual_name} and {pred_name}")
```

```python
threshs = np.linspace(compare[pred_name].min(), compare[pred_name].max())

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
for (lt, adm), group in compare.groupby([leadtime_name, "ADM0_CODE"]):
    for thresh in threshs:
        class_rep = classification_report(
            group[actual_bool], group[pred_name] < thresh, output_dict=True
        ).get("True")
        df_add = pd.DataFrame(class_rep, index=[0])
        df_add[[leadtime_name, "ADM0_CODE", "thresh"]] = lt, adm, thresh
        dfs.append(df_add)

scores_by_thresh = pd.concat(dfs, ignore_index=True)
```

```python
for adm, group in scores_by_thresh.groupby("ADM0_CODE"):
    group.pivot_table(
        columns=leadtime_name, index="thresh", values="precision"
    ).plot(title=adm)
```

```python
def plot_roc(df, actual_col, pred_col, colors_param, plots_param):

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

        for leadtime, group in adm_group.groupby(colors_param):
            fpr, tpr, thresholds = roc_curve(
                group[actual_col], -group[pred_col]
            )
            roc_auc = auc(fpr, tpr)
            roc_auc_values.append(((adm_code, leadtime), roc_auc))

            # Plot ROC curve for each group
            ax.plot(
                fpr,
                tpr,
                lw=2,
                label=f"Leadtime {leadtime} (AUC = {roc_auc:.2f})",
            )

        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Guess",
        )
        ax.set_title(adm_code)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal", "box")

    # Display AUC values
    # print("AUC values by ADM0_CODE and Leadtime:")
    # for (adm_code, leadtime), auc_value in roc_auc_values:
    #     print(f"{adm_code}, Leadtime {leadtime}: {auc_value:.2f}")

    plt.tight_layout()
    plt.show()


plot_roc(compare, "ec_re_mean_q33", "ec_season_mean", "season_lt", "")
```

```python

```
