import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def var_es(sigma2, alpha=0.01, dist="normal", nu=None, mu=0.0):
  """1-day VaR and ES (positive losses, in %) from a variance forecast."""
  sigma = np.sqrt(np.asarray(sigma2, float))
  if dist == "normal":
    z = stats.norm.ppf(alpha)
    var = -(mu + sigma * z)
    es = -mu + sigma * stats.norm.pdf(z) / alpha
  elif dist == "t":
    if nu is None or nu <= 2:
      raise ValueError("Student-t needs nu > 2")
    scale = np.sqrt((nu - 2) / nu)  # rescale t to variance 1
    q = stats.t.ppf(alpha, nu)
    var = -(mu + sigma * scale * q)
    es = -mu + sigma * scale * (nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / alpha
  else:
    raise ValueError(f"unknown dist {dist!r}")
  return var, es


def var_hits(realized, var):
  """1 on days the loss beat VaR (return < -VaR), else 0."""
  return (np.asarray(realized, float) < -np.asarray(var, float)).astype(int)


def kupiec(hits, alpha):
  """Is the hit rate equal to alpha?"""
  h = np.asarray(hits, int)
  n, x = len(h), h.sum()
  pi = x / n
  ll_null = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
  ll_alt = xlogy(n - x, 1 - pi) + xlogy(x, pi)
  lr = -2 * (ll_null - ll_alt)
  return {"n": n, "hits": int(x), "hit_rate": pi, "LR_uc": lr, "p_uc": stats.chi2.sf(lr, 1)}


def christoffersen(hits, alpha):
  """Do hits cluster (independence)? Plus conditional coverage = Kupiec + independence."""
  h = np.asarray(hits, int)
  prev, curr = h[:-1], h[1:]
  n00 = np.sum((prev == 0) & (curr == 0))
  n01 = np.sum((prev == 0) & (curr == 1))
  n10 = np.sum((prev == 1) & (curr == 0))
  n11 = np.sum((prev == 1) & (curr == 1))
  pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
  pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
  pi = (n01 + n11) / (n00 + n01 + n10 + n11)
  ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
  ll_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
  lr_ind = -2 * (ll_null - ll_alt)
  uc = kupiec(h, alpha)
  lr_cc = uc["LR_uc"] + lr_ind
  return {**uc, "LR_ind": lr_ind, "p_ind": stats.chi2.sf(lr_ind, 1),
          "LR_cc": lr_cc, "p_cc": stats.chi2.sf(lr_cc, 2)}


RISK_COLS = ["hits", "hit_rate", "p_uc", "p_ind", "p_cc", "avg_VaR", "avg_ES"]


def backtest(out, alphas=(0.01, 0.05), dist="normal", nu=None):
  """VaR backtest for every model in a get_forecasts() output: one row per (model, alpha).
  p < 0.05 means the model's VaR is rejected."""
  rows = {}
  for m in out.columns.drop("realized"):
    ok = out[m].notna() & out["realized"].notna()
    f, r = out.loc[ok, m], out.loc[ok, "realized"]
    for a in alphas:
      var, es = var_es(f, alpha=a, dist=dist, nu=nu)
      res = christoffersen(var_hits(r, var), a)
      rows[(m, a)] = [res["hits"], res["hit_rate"], res["p_uc"], res["p_ind"], res["p_cc"],
                      np.mean(var), np.mean(es)]
  index = pd.MultiIndex.from_tuples(rows.keys(), names=["model", "alpha"])
  return pd.DataFrame(list(rows.values()), index=index, columns=RISK_COLS)
