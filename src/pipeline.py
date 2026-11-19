import time
import numpy as np
import pandas as pd
from src.benchmarks import hist_var, ewma
from src.garch import RollingGarch
from src.evaluation import score
from src.risk import backtest
from src.regimes import regime_labels, score_by_regime


def get_forecasts(returns, models, min_train=250, step=1):
  """For each day t, give every model ONLY the returns before t and store its
  variance forecast for day t next to what actually happened on day t.
  `returns` must already be clean: sorted by date, no NaNs."""

  # Validation Checks
  if not returns.index.is_monotonic_increasing:
    raise ValueError("returns must be sorted by date (index is not monotonic increasing)")
  if returns.isna().any():
    bad = returns.index[returns.isna()]
    raise ValueError(f"returns has {len(bad)} NaN value(s), first on {bad[0]}")
  if len(returns) <= min_train:
    raise ValueError(f"need more than min_train={min_train} returns, got {len(returns)}")

  out = pd.DataFrame({"realized": returns.iloc[min_train::step]}, index=returns.index[min_train::step])
  for name, fn in models.items():
    out[name] = [fn(returns.iloc[:i]) for i in range(min_train, len(returns), step)]
  return out


def build_models(refit_every=20):
  """Fresh model objects (GARCH keeps state, so every pair gets new ones)."""
  return {
    "hist_30": hist_var(30),
    "hist_60": hist_var(60),
    "hist_250": hist_var(250),
    "ewma": ewma(0.94),
    "garch11": RollingGarch(o=0, refit_every=refit_every),
    "gjr_garch": RollingGarch(o=1, refit_every=refit_every),
  }


MODEL_NAMES = list(build_models())


def next_day_vol(returns, refit_every=20):
  """Each model's forecast for the trading day AFTER the last date in `returns`,
  as annualised volatility in %."""
  return {m: np.sqrt(fn(returns) * 252) for m, fn in build_models(refit_every).items()}


def run_all(returns, refit_every=20, min_train=250):
  """Every model on every pair: forecasts, accuracy scores, VaR backtests and calm/stressed scores."""
  outs, fits, results, risk, by_regime = {}, {}, {}, {}, {}
  for c in returns.columns:
    t0 = time.time()
    models = build_models(refit_every)
    out = get_forecasts(returns[c], models, min_train=min_train)
    outs[c] = out
    fits[c] = {m: (models[m].n_fits, models[m].n_failed) for m in ["garch11", "gjr_garch"]}
    results[c] = score(out)
    risk[c] = backtest(out)
    by_regime[c] = score_by_regime(out, regime_labels(returns[c]))
    print(f"{c}/USD: {len(out)} forecast days, {time.time() - t0:.0f}s, "
          f"GARCH fits (total, failed): {fits[c]}")
  latest = pd.DataFrame({f"{c}/USD": next_day_vol(returns[c], refit_every) for c in returns.columns}).T
  return {
    "outs": outs,
    "results": pd.concat(results, names=["currency", "model"]),
    "risk": pd.concat(risk, names=["currency", "model", "alpha"]),
    "by_regime": pd.concat(by_regime, names=["currency", "regime", "model"]),
    "next_day_vol": latest,
    "garch_fits": fits,
  }
