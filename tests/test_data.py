import pytest
from src.data import fetch_returns, FX_SERIES

@pytest.fixture(scope="module")
def returns():
  return fetch_returns()

def test_fetch_returns_dates_sorted_and_unique(returns):
  assert returns.index.is_monotonic_increasing
  assert returns.index.is_unique

def test_fetch_returns_has_all_pairs(returns):
  assert list(returns.columns) == list(FX_SERIES)

def test_fetch_returns_has_no_nans(returns):
  assert returns.notna().all().all()