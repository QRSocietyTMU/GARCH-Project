import numpy as np
import pandas as pd
from src.data import fetch_returns
from src.regimes import regime_labels


def test_regime_labels_values():
  rng = np.random.default_rng(3)
  idx = pd.bdate_range("2010-01-01", periods=1000)
  r = pd.Series(np.r_[rng.normal(0, 0.5, 700), rng.normal(0, 2.0, 300)], index=idx)
  lab = regime_labels(r)
  # stressed block is detected once the 20-day window has filled up
  assert (lab.iloc[720:820] == "stressed").mean() > 0.9
  # calm block: roughly the top quarter gets flagged, by construction
  assert 0.05 < (lab.iloc[300:700] == "stressed").mean() < 0.45


def test_regime_labels_ignores_future():
  rng = np.random.default_rng(3)
  idx = pd.bdate_range("2010-01-01", periods=1000)
  r = pd.Series(np.r_[rng.normal(0, 0.5, 700), rng.normal(0, 2.0, 300)], index=idx)
  lab = regime_labels(r)

  # junk in the future leaves earlier labels alone
  k = 600
  poisoned = r.copy()
  poisoned.iloc[k:] = 999.0
  pd.testing.assert_series_equal(lab.iloc[:k + 1], regime_labels(poisoned).iloc[:k + 1])

  # a shock on day j can only change labels AFTER day j
  j = 500
  shocked = r.copy()
  shocked.iloc[j] = 50.0
  lab2 = regime_labels(shocked)
  pd.testing.assert_series_equal(lab.iloc[:j + 1], lab2.iloc[:j + 1])
  assert (lab2.iloc[j + 1] == "stressed") and (lab.iloc[j + 1] != lab2.iloc[j + 1])


def test_regime_labels_ignores_future_on_real_data():
  returns = fetch_returns()
  for c in returns.columns:
    rc = returns[c]
    lc = regime_labels(rc)
    cut = len(rc) // 2
    p = rc.copy()
    p.iloc[cut:] = 999.0
    pd.testing.assert_series_equal(lc.iloc[:cut + 1], regime_labels(p).iloc[:cut + 1])
