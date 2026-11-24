import warnings
import numpy as np
import pandas as pd
from arch import arch_model


class RollingGarch:
  """Walk-forward GARCH(1,1) (o=0) or GJR-GARCH(1,1,1) (o=1).
  Re-estimates the parameters every `refit_every` days on the data before t, and on the
  days in between keeps those parameters and updates the variance with the new returns."""

  def __init__(self, o=0, refit_every=20, dist="normal"):
    self.o, self.refit_every, self.dist = o, refit_every, dist
    self.params, self.fit_len = None, None
    self.n_fits, self.failed, self.odd = 0, [], []  # failed/odd: last training date of those fits

  def _model(self, train):
    return arch_model(train, mean="Constant", vol="GARCH", p=1, o=self.o, q=1, dist=self.dist)

  def __call__(self, train):
    n = len(train)
    if self.params is None or n < self.fit_len or n - self.fit_len >= self.refit_every:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = self._model(train).fit(disp="off")
      self.n_fits += 1
      day = train.index[-1]
      if res.convergence_flag != 0:
        self.failed.append(day)
      if res.convergence_flag == 0 or self.params is None:  # on failure keep the last good parameters
        self.params = res.params
        if is_odd(self.params):
          self.odd.append(day)
      self.fit_len = n
    fc = self._model(train).fix(self.params).forecast(horizon=1, reindex=False)
    return float(fc.variance.iloc[-1, 0])


def persistence(params):
  return params["alpha[1]"] + params["beta[1]"] + 0.5 * params.get("gamma[1]", 0.0)


def is_odd(params):
  """A fit stuck at the edge of what's allowed: persistence ~1 (no stable long-run variance)
  or alpha ~0 (the variance stops reacting to new shocks)."""
  return persistence(params) >= 0.999 or params["alpha[1]"] < 1e-6


def param_table(res):
  """coef, std_err, t_stat and p_value for each parameter of one fitted model."""
  return pd.DataFrame({
    "coef": res.params,
    "std_err": res.std_err,
    "t_stat": res.tvalues,
    "p_value": res.pvalues,
  })


def summary_table(fits):
  """persistence, half-life, long-run vol, AIC and BIC, one row per model."""
  rows = {}
  for name, res in fits.items():
    pers = persistence(res.params)
    rows[name] = {
      "persistence": round(pers, 4),
      "half-life (days)": round(np.log(0.5) / np.log(pers), 1),
      "long-run vol (% a year)": round(np.sqrt(res.params["omega"] / (1 - pers) * 252), 2),
      "AIC": round(res.aic, 1),
      "BIC": round(res.bic, 1),
    }
  return pd.DataFrame(rows).T
