import pytest
import pandas as pd
import numpy as np
from src.evaluation import mae, rmse, qlike, score, dm_test


def test_metrics_values():
  f = np.array([1.0, 2.0, 3.0])
  p = np.array([2.0, 2.0, 5.0])
  assert np.isclose(mae(f, p), 1.0)                           # (1 + 0 + 2) / 3
  assert np.isclose(rmse(f, p), np.sqrt(5 / 3))               # sqrt((1 + 0 + 4) / 3)
  assert np.isclose(qlike(f, p), np.mean(np.log(f) + p / f))
  assert mae(p, p) == 0 and rmse(p, p) == 0                   # perfect forecast
  assert rmse(f, p) >= mae(f, p)


def test_qlike_behaviour():
  assert qlike([0.5], [1.0]) > qlike([2.0], [1.0])            # under-forecasting costs more
  rng = np.random.default_rng(1)
  prox = rng.chisquare(1, 200_000)                            # squared N(0,1) draws
  q_true = qlike(np.ones_like(prox), prox)
  assert q_true < qlike(np.full_like(prox, 1.3), prox)        # lowest at the true variance
  assert q_true < qlike(np.full_like(prox, 0.7), prox)
  assert np.isfinite(qlike([1.0, 1.0], [0.0, 1.0]))           # zero-return day is fine


def test_metrics_input_handling():
  assert np.isclose(mae([1.0, np.nan, 3.0], [2.0, 5.0, 5.0]), 1.5)   # NaN day dropped
  assert np.isclose(mae(pd.Series([1.0, 2, 3]), pd.Series([2.0, 2, 5])), 1.0)
  with pytest.raises(ValueError):
    mae([1.0, 2.0], [1.0])                                  # mismatched lengths


def test_score_values():
  fake = pd.DataFrame({
    "realized": [1.0, -2.0, 0.0],
    "hist_30": [1.0, 4.0, 1.0],
    "ewma": [2.0, 2.0, 2.0]
  })
  t = score(fake)
  assert list(t.index) == ["hist_30", "ewma"]
  assert np.isclose(t.loc["hist_30", "MAE"], 1 / 3)
  assert np.isclose(t.loc["ewma", "RMSE"], np.sqrt((1 + 4 + 4) / 3))


def test_dm_test_values():
  rng = np.random.default_rng(0)
  a = rng.normal(1.0, 1, 2000)
  b = rng.normal(1.5, 1, 2000)                                # b's loss is clearly higher
  stat, p = dm_test(a, b)
  assert stat < 0 and p < 0.01
  assert np.isclose(dm_test(b, a)[0], -stat)                  # swapping flips the sign
  assert dm_test(a, rng.normal(1.0, 1, 2000))[1] > 0.05       # same average loss -> not significant
