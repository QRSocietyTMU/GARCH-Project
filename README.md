# GARCH-Project

Walk-forward volatility forecasting for five USD FX pairs (AUD, EUR, JPY, CHF, GBP), comparing historical-variance and EWMA benchmarks with GARCH(1,1) and GJR-GARCH. The notebook downloads its data from [FRED](https://fred.stlouisfed.org/) when it runs. See the **Data** section at the top of `main.ipynb` for sources and processing.

Requires Python 3.13+ and an internet connection.

## Project layout

```
main.ipynb        the analysis: runs everything and shows the tables and plots
src/
  data.py         download FX rates from FRED and turn them into daily returns
  eda.py          summary stats, autocorrelation tests, histogram/ACF plots
  benchmarks.py   historical-variance and EWMA forecasts
  garch.py        GARCH / GJR-GARCH: full-sample fit tables and the walk-forward model
  pipeline.py     walk-forward loop and run_all (every model on every pair)
  evaluation.py   MAE, RMSE, QLIKE
  risk.py         VaR/ES and the Kupiec / Christoffersen backtests
  regimes.py      calm/stressed labels and scores per regime
  plots.py        plotting helpers
tests/            one test file per module
old_src.ipynb     the original single-notebook version, kept for reference
```

Core logic lives in `src/`; `main.ipynb` only imports it and puts the story together.

## Setup

### Option 1: uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```bash
uv sync
```

This creates `.venv/` with the exact versions in `uv.lock`.

### Option 2: pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook

Open `main.ipynb` in VS Code and select the `.venv` interpreter as the kernel, or run it in Jupyter:

```bash
uv run --with jupyter jupyter lab          # uv
pip install jupyter && jupyter lab         # pip, inside the activated venv
```

A full run fits thousands of GARCH models and takes a few minutes.

## Running tests

Tests live in `tests/`, one file per module in `src/`. From the repo root:

```bash
uv run pytest                    # uv
pytest                           # pip, inside the activated venv
```

Useful flags: `-v` lists each test, `-x` stops at the first failure, and `tests/test_risk.py` runs a single file. Several tests download from FRED, so they need an internet connection.

The full suite takes about 2 minutes, mostly `test_run_all_final_checks`, which runs every model on every pair. Skip it while you're working with `uv run pytest -k "not run_all"`.

## Updating dependencies

Manage dependencies with uv, then regenerate `requirements.txt` so pip users get the same versions:

```bash
uv add <package>                 # or: uv remove <package>
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```
