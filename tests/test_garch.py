import numpy as np
import pandas as pd
from arch import arch_model
from src.data import fetch_returns
from src.garch import RollingGarch
from src.pipeline import get_forecasts


def test_rolling_garch_matches_arch():
  r = fetch_returns()["EUR"].iloc[:1500]
  for o in (0, 1):
    g = RollingGarch(o=o, refit_every=20)
    # on a refit day the forecast equals arch's own 1-day forecast
    f0 = g(r.iloc[:1000])
    res = arch_model(r.iloc[:1000], mean="Constant", vol="GARCH", p=1, o=o, q=1).fit(disp="off")
    assert np.isclose(f0, res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    # the next day uses the same parameters and the GARCH recursion by hand
    f1 = g(r.iloc[:1001])
    p = g.params
    e = r.iloc[1000] - p["mu"]
    by_hand = (p["omega"] + p["alpha[1]"] * e ** 2 + p.get("gamma[1]", 0) * e ** 2 * (e < 0)
               + p["beta[1]"] * f0)
    assert np.isclose(f1, by_hand), (f1, by_hand)
    assert g.n_fits == 1
    for k in range(1002, 1041):
      g(r.iloc[:k])
    assert g.n_fits == 3  # refit at 1000, 1020, 1040


def test_rolling_garch_tracks_true_variance():
  # simulated GARCH data: forecasts track the true variance closely
  sim = arch_model(None, p=1, q=1)
  data = sim.simulate([0.0, 0.02, 0.08, 0.9], 1500, burn=500)
  rs = pd.Series(data["data"].values, index=pd.bdate_range("2000-01-03", periods=1500))
  g = RollingGarch(o=0, refit_every=20)
  fc = [g(rs.iloc[:i]) for i in range(500, 1500)]
  true_var = data["volatility"].values[500:] ** 2
  assert np.corrcoef(fc, true_var)[0, 1] > 0.9


def test_rolling_garch_ignores_future():
  # no lookahead on a short window of real data
  r = fetch_returns()["EUR"].iloc[-600:]
  models = lambda: {"garch11": RollingGarch(0), "gjr_garch": RollingGarch(1)}

  cut = r.index[len(r) // 2]
  poisoned = r.copy()
  poisoned.loc[cut:] = 999.0

  a = get_forecasts(r, models())
  b = get_forecasts(poisoned, models())
  cols = ["garch11", "gjr_garch"]

  # forecasts up to day `cut` only used data before it -> identical
  pd.testing.assert_frame_equal(a.loc[:cut, cols], b.loc[:cut, cols])
  # after the cut the junk must show up, otherwise this test couldn't catch anything
  for m in cols:
    assert not np.allclose(a.loc[a.index > cut, m], b.loc[b.index > cut, m]), m
