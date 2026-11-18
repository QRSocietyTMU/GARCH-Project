import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw without opening windows
import matplotlib.pyplot as plt
from src.evaluation import score
from src.regimes import regime_labels
from src.plots import plot_regimes, plot_forecasts, plot_var_backtest, plot_metric_ranks

MODEL_NAMES = ["hist_30", "hist_60", "hist_250", "ewma", "garch11", "gjr_garch"]


def test_plots_return_figures():
  rng = np.random.default_rng(5)
  idx = pd.bdate_range("2020-01-01", periods=300)
  fake = pd.DataFrame({"realized": rng.normal(0, 0.6, 300)}, index=idx)
  for m in MODEL_NAMES:
    fake[m] = 0.36 * rng.uniform(0.8, 1.2, 300)
  reg = regime_labels(pd.Series(rng.normal(0, 0.6, 600),
                                index=pd.bdate_range("2019-01-01", periods=600)))
  res = pd.concat({"EUR": score(fake), "AUD": score(fake)}, names=["currency", "model"])
  figs = [plot_regimes(fake["realized"], reg.reindex(idx), "t", show=False),
          plot_forecasts(fake, "t", regime=reg, show=False),
          plot_var_backtest(fake, "ewma", 0.01, "t", show=False),
          plot_metric_ranks(res, "QLIKE", show=False)]
  assert all(isinstance(f, plt.Figure) for f in figs)
  for f in figs:
    plt.close(f)
