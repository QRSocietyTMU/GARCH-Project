import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def qlike_loss(f, p, eps=1e-12):
  # daily log f + p / f: same model ranking as p/f - log(p/f) - 1,
  # but finite on the ~55 days per pair where the return is exactly 0
  f = np.maximum(f, eps)
  return np.log(f) + p / f


def qlike(forecast, proxy, eps=1e-12):
  f, p = _clean(forecast, proxy)
  return np.mean(qlike_loss(f, p, eps))


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


def dm_test(loss_a, loss_b):
  """Diebold-Mariano test on two models' daily losses: is the average gap real or noise?
  DM_stat < 0 means model a has the lower loss. Newey-West errors, since loss gaps cluster."""
  d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
  d = d[np.isfinite(d)]
  lags = int(4 * (len(d) / 100) ** (2 / 9))  # Newey-West rule of thumb
  res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
  return res.tvalues[0], res.pvalues[0]


def dm_table(out, base="gjr_garch"):
  """DM test of `base` against every other model on QLIKE, one row per model.
  DM_stat < 0 = `base` is better, p_value < 0.05 = the difference is significant."""
  proxy = out["realized"] ** 2
  others = out.columns.drop(["realized", base])
  return pd.DataFrame(
    [dm_test(qlike_loss(out[base], proxy), qlike_loss(out[m], proxy)) for m in others],
    index=others,
    columns=["DM_stat", "p_value"],
  )
