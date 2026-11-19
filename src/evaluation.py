import numpy as np
import pandas as pd


def _clean(forecast, proxy):
  """Float arrays with any NaN/inf day dropped from both sides."""
  f = np.asarray(forecast, float)
  p = np.asarray(proxy, float)

  if f.shape != p.shape:
    raise ValueError(f"forecast has shape {f.shape} but proxy has shape {p.shape}")

  ok = np.isfinite(f) & np.isfinite(p)
  if not ok.any():
    raise ValueError("no valid (forecast, proxy) pairs")

  return f[ok], p[ok]


def mae(forecast, proxy):
  f, p = _clean(forecast, proxy)
  return np.mean(np.abs(p - f))


def rmse(forecast, proxy):
  f, p = _clean(forecast, proxy)
  return np.sqrt(np.mean((p - f) ** 2))


def qlike(forecast, proxy, eps=1e-12):
  # mean(log f + p / f): same model ranking as p/f - log(p/f) - 1,
  # but finite on the ~55 days per pair where the return is exactly 0
  f, p = _clean(forecast, proxy)
  f = np.maximum(f, eps)
  return np.mean(np.log(f) + p / f)


METRICS = {"MAE": mae, "RMSE": rmse, "QLIKE": qlike}


def score(out):
  """MAE, RMSE and QLIKE for every model in a get_forecasts() output, one row per model.
  The proxy for each day's true variance is that day's squared return. Lower is better."""
  proxy = out["realized"] ** 2
  models = out.columns.drop("realized")
  return pd.DataFrame(
    {name: [fn(out[m], proxy) for m in models] for name, fn in METRICS.items()},
    index=models,
  )
