import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def plot_eda(r, name, lags=40):
  """Histogram of returns against a normal curve, and the ACF of squared returns."""
  _, ax = plt.subplots(1, 2, figsize=(13, 4))

  ax[0].hist(r, bins=100, density=True, alpha=0.6, color="#2a78d6", label="Returns")
  x = np.linspace(r.min(), r.max(), 400)
  ax[0].plot(x, stats.norm.pdf(x, r.mean(), r.std()), color="#eb6834", lw=2, label="Normal")
  ax[0].set_title(f"{name}: histogram of log returns (%)")
  ax[0].legend()

  plot_acf(r ** 2, lags=lags, ax=ax[1], zero=False, auto_ylims=True)
  ax[1].set_title(f"{name}: ACF of squared returns")

  plt.tight_layout()
  plt.show()

def summary_stats(r):
  """Size, spread and shape of one pair's returns, plus the Jarque-Bera normality test."""
  jb_stat, jb_p = stats.jarque_bera(r)
  return pd.Series({
      "n": len(r),
      "mean": r.mean(),
      "std": r.std(),
      "ann_vol_%": r.std() * np.sqrt(252),
      "min": r.min(),
      "max": r.max(),
      "skew": stats.skew(r),
      "excess_kurt": stats.kurtosis(r),
      "JB_stat": jb_stat,
      "JB_p": jb_p,
  })


def dependence_tests(r, lags=(10, 20)):
  """Ljung-Box on returns and squared returns, and the ARCH LM test: stat and p_value."""
  rows = {}
  for L in lags:
    lb_r = acorr_ljungbox(r, lags=[L])
    lb_r2 = acorr_ljungbox(r ** 2, lags=[L])
    rows[f"LB returns Q({L})"] = (lb_r["lb_stat"].iloc[0], lb_r["lb_pvalue"].iloc[0])
    rows[f"LB squared Q({L})"] = (lb_r2["lb_stat"].iloc[0], lb_r2["lb_pvalue"].iloc[0])
  arch_lm = het_arch(r, nlags=10, result_object=True)
  rows["ARCH LM (10 lags)"] = (arch_lm.lm, arch_lm.lmpval)
  return pd.DataFrame(rows, index=["stat", "p_value"]).T
