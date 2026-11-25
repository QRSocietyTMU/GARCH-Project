import pytest
import pandas as pd
import numpy as np
from src.data import fetch_returns
from src.pipeline import get_forecasts, build_models, run_all, MODEL_NAMES

MIN_TRAIN = 50


@pytest.fixture
def fake_returns():
  rng = np.random.default_rng(0)
  return pd.Series(rng.normal(0, 1, 300), index=pd.bdate_range("2020-01-01", periods=300))


def test_get_forecasts_uses_only_past(fake_returns):
  seen = []
  # fake model: records the last date it was given
  def spy(train):
    seen.append(train.index[-1])
    return 0.0

  out = get_forecasts(fake_returns, {"spy": spy}, min_train=MIN_TRAIN)

  assert len(seen) == len(out.index)
  assert all(s < day for s, day in zip(seen, out.index))


def test_get_forecasts_ignores_future(fake_returns):
  # fake model: forecast = most recent return
  last_value = {"last": lambda train: train.iloc[-1]}

  cut = fake_returns.index[150]
  poisoned = fake_returns.copy()
  poisoned.loc[cut:] = 999.0

  a = get_forecasts(fake_returns, last_value, min_train=MIN_TRAIN)
  b = get_forecasts(poisoned, last_value, min_train=MIN_TRAIN)

  # forecasts up to day `cut` only used data before it -> identical
  pd.testing.assert_series_equal(a.loc[:cut, "last"], b.loc[:cut, "last"])
  # after the cut the junk must show up, otherwise this test couldn't catch anything
  assert not np.allclose(a.loc[a.index > cut, "last"], b.loc[b.index > cut, "last"])


def test_get_forecasts_dates_align(fake_returns):
  out = get_forecasts(fake_returns, {"last": lambda train: train.iloc[-1]}, min_train=MIN_TRAIN)
  # one row per day after the training window, nothing missing
  assert out.index.equals(fake_returns.index[MIN_TRAIN:])
  assert out.notna().all().all()
  # each row holds that day's actual return and the forecast made the day before
  assert out["realized"].equals(fake_returns.iloc[MIN_TRAIN:])
  assert out["last"].equals(fake_returns.shift(1).iloc[MIN_TRAIN:])


def test_run_all_final_checks():
  """The original notebook's final checks. Runs every model on every pair (~1.5 min)."""
  returns = fetch_returns()
  run = run_all(returns)
  outs = run["outs"]
  # every pair produced a forecast for every model on every day, all positive
  for c, out in outs.items():
    assert list(out.columns) == ["realized"] + MODEL_NAMES, c
    f = out[MODEL_NAMES]
    assert np.isfinite(f.values).all() and (f.values > 0).all(), c
    # forecasts are on the right scale: average forecast vs average squared return
    ratio = f.mean() / (out["realized"] ** 2).mean()
    assert ratio.between(0.5, 2.0).all(), (c, ratio.round(2).to_dict())
    assert out.index.equals(returns[c].index[250:]), c
  # every results/risk cell is filled
  assert run["results"].notna().all().all()
  assert run["risk"].notna().all().all()
  assert run["dm"].notna().all().all()
  assert run["next_day_vol"].notna().all().all()
  # the loop forecast for the last day matches calling the model directly (no hidden state bugs)
  for c in returns.columns:
    r = returns[c]
    direct = {m: fn(r.iloc[:-1]) for m, fn in build_models().items()
              if m in ["hist_30", "hist_60", "hist_250", "ewma"]}
    for m, v in direct.items():
      assert np.isclose(outs[c][m].iloc[-1], v), (c, m)
