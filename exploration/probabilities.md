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

# Probabilities

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
```

```python
def prob_single_event(n, p_any):
    return 1 - (1 - p_any) ** (1 / n)


def rp_single_event(n, rp_any):
    return 1 / prob_single_event(n, 1 / rp_any)


def p_any_corr_2(corr, p_single):
    return (corr - 1) * p_single**2 + (2 - corr) * p_single


def p_single_corr_2(corr, p_any):
    pm = np.array([1, -1])
    return ((corr - 2) + np.sqrt((2 - corr) ** 2 + 4 * (corr - 1) * p_any)) / (
        2 * (corr - 1)
    )
```

```python
rps = np.linspace(2, 4)
ns = np.linspace(1, 8)

rp_single = rp_single_event2(ns[:, np.newaxis], rps)

da = xr.DataArray(
    rp_single.T, coords={"rp_any": rps, "n": ns}, dims=["rp_any", "n"]
)

contour = da.T.plot.contour(figsize=(5, 5), levels=np.arange(3, 100, 1))
ax = plt.gca()  # Get the current axis
ax.clabel(contour, inline=True, fontsize=8)
```

```python
da.sel(rp_any=3, n=2, method="nearest")
```

```python
1 / (1 - (1 - 1 / 3) ** (1 / 2))
```

```python
ns = np.linspace(1, 8)
plt.plot(ns, rp_single_event(ns, 3))
```

```python
1 / p_any_corr_2(0.5, 1 / 3)
```

```python
p_single_corr_2(0.01, 0.01)
```
