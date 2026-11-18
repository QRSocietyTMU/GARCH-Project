import pandas as pd
import numpy as np
from src.benchmarks import hist_var, ewma


def test_hist_var_values():
  x = pd.Series([1.0, 2, 3, 4, 5, 6])
  y = x.copy()
  y.iloc[:3] = 999                                      # old data outside the window
  assert np.isclose(hist_var(3)(x), 1.0)                # var of 4, 5, 6
  assert np.isclose(hist_var(3)(y), hist_var(3)(x))
  assert hist_var(5)(pd.Series([2.0] * 10)) == 0.0      # constant -> zero variance


def test_ewma_values():
  r = pd.Series([1.0, 2, 3])
  a = pd.Series([1.0, 1, 1, 1, 5])
  b = pd.Series([1.0, 1, 1, 1, 1])
  assert np.isclose(ewma(0.94)(r), 1.6492)              # 1 -> 1.18 -> 1.6492 by hand
  assert np.isclose(ewma(0.94)(pd.Series([2.0] * 50)), 4.0)
  assert ewma(0.94)(a) > ewma(0.94)(b)                  # a recent shock raises the forecast
