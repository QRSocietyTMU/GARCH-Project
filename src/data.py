import numpy as np
import pandas as pd
# Currency -> FRED series id
FX_SERIES = {
    "AUD": "DEXUSAL",
    "EUR": "DEXUSEU",
    "JPY": "DEXJPUS",
    "CHF": "DEXSZUS",
    "GBP": "DEXUSUK",
}
# FRED quotes these as foreign per USD; inverting makes every pair USD per 1 unit of currency (X/USD)
INVERTED = ["JPY", "CHF"]

START, END = "2003-01-02", "2026-11-06"
FETCH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"

def fetch_returns(fx_series=FX_SERIES, start=START, end=END):
  prices = pd.DataFrame({
      currency: pd.read_csv(FETCH_URL.format(sid=series_id, start=start, end=end), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
      for currency, series_id in fx_series.items()
  })
  prices[INVERTED] = 1 / prices[INVERTED]

  prices = prices.dropna().sort_index()
  return 100 * np.log(prices).diff().dropna()

