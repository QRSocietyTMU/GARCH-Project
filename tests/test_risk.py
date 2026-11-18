import numpy as np
from src.risk import var_es, var_hits, kupiec, christoffersen


def test_var_es_values():
  v, e = var_es(1.0, 0.01)
  assert np.isclose(v, 2.3263, atol=1e-4) and np.isclose(e, 2.6652, atol=1e-4)
  v, e = var_es(1.0, 0.05)
  assert np.isclose(v, 1.6449, atol=1e-4) and np.isclose(e, 2.0627, atol=1e-4)
  assert np.isclose(var_es(4.0, 0.01)[0], 2 * var_es(1.0, 0.01)[0])
  for d, nu in [("normal", None), ("t", 5)]:
    v1, e1 = var_es(1.0, 0.01, d, nu)
    v5, e5 = var_es(1.0, 0.05, d, nu)
    assert e1 > v1 and e5 > v5 and v1 > v5
  assert np.allclose(var_es(1.0, 0.01, "t", 1e6), var_es(1.0, 0.01), atol=1e-3)
  assert var_es(1.0, 0.01, "t", 4)[0] > var_es(1.0, 0.01)[0]
  rng = np.random.default_rng(0)
  for d, nu in [("normal", None), ("t", 6)]:
    x = rng.standard_normal(2_000_000) if d == "normal" else \
      rng.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
    v, e = var_es(1.0, 0.01, d, nu)
    q = -np.quantile(x, 0.01)
    assert np.isclose(v, q, rtol=0.01) and np.isclose(e, -x[x <= -q].mean(), rtol=0.02)
  assert list(var_hits([-3.0, -1.0, 2.0, -2.0], [2.0] * 4)) == [1, 0, 0, 0]


def test_backtests_values():
  k = kupiec([1, 1] + [0] * 8, 0.1)
  expected = -2 * (8 * np.log(0.9) + 2 * np.log(0.1) - 8 * np.log(0.8) - 2 * np.log(0.2))
  assert k["hits"] == 2 and np.isclose(k["LR_uc"], expected)
  assert np.isfinite(kupiec(np.zeros(500), 0.01)["LR_uc"])
  assert np.isfinite(christoffersen(np.zeros(500), 0.01)["LR_cc"])
  assert np.isclose(kupiec([1] + [0] * 99, 0.01)["LR_uc"], 0.0)
  rng = np.random.default_rng(42)
  n = 2000
  res = christoffersen((rng.random(n) < 0.01).astype(int), 0.01)          # correct model
  assert res["p_uc"] > 0.05 and res["p_ind"] > 0.05 and res["p_cc"] > 0.05
  assert kupiec((rng.random(n) < 0.03).astype(int), 0.01)["p_uc"] < 0.01  # too many hits
  clustered = np.zeros(n, int)
  for s in rng.choice(np.arange(0, n - 5, 5), size=4, replace=False):
    clustered[s:s + 5] = 1                                                # hits in runs of 5
  assert christoffersen(clustered, 0.01)["p_ind"] < 0.01
  sigma2 = np.exp(rng.normal(0, 0.5, n)) * 0.4
  r = rng.standard_normal(n) * np.sqrt(sigma2)
  assert kupiec(var_hits(r, var_es(sigma2, 0.01)[0]), 0.01)["p_uc"] > 0.05
  assert kupiec(var_hits(r, var_es(sigma2 * 0.5, 0.01)[0]), 0.01)["p_uc"] < 0.01
