import numpy as np
import pandas as pd
from src.evaluation import score


def regime_labels(returns, window=20, q=0.75, min_history=250):
  """Day t is 'stressed' if the volatility of the `window` days before t is in the
  top (1-q) of all such values seen up to t. Uses nothing from day t onward."""
  r = returns.dropna()
  rv = r.rolling(window).std().shift(1)                      # vol of days t-window .. t-1
  thr = rv.expanding(min_periods=min_history).quantile(q)    # cutoff from the past only
  lab = pd.Series(np.where(rv > thr, "stressed", "calm"), index=r.index, dtype=object)
  lab[rv.isna() | thr.isna()] = np.nan                       # warm-up days get no label
  return lab


def score_by_regime(out, regime, fn=score):
  """`fn` (MAE/RMSE/QLIKE by default) separately for calm and stressed days,
  one row per (regime, model)."""
  reg = regime.reindex(out.index)
  tables = {}
  for g in ["calm", "stressed"]:
    mask = (reg == g).values
    if mask.any():
      t = fn(out[mask])
      t.insert(0, "n_days", mask.sum())
      tables[g] = t
  return pd.concat(tables, names=["regime", "model"])
