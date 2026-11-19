import numpy as np
import matplotlib.pyplot as plt
from src.risk import var_es, var_hits

MODEL_COLORS = {"hist_30": "#2a78d6", "hist_60": "#eb6834", "hist_250": "#1baf7a",
                "ewma": "#eda100", "garch11": "#e87ba4", "gjr_garch": "#008300"}
STRESS_SHADE = "#c3c2b7"


def _style(ax):
  ax.grid(axis="y", color="#e6e5e0", lw=0.8)
  ax.spines[["top", "right"]].set_visible(False)


def _shade_stress(ax, regime):
  s = (regime == "stressed").reindex(regime.index).fillna(False).values
  ax.fill_between(regime.index, 0, 1, where=s, transform=ax.get_xaxis_transform(),
                  color=STRESS_SHADE, alpha=0.35, lw=0, label="stressed")


def plot_regimes(returns, regime, title, show=True):
  fig, ax = plt.subplots(figsize=(13, 3.5))
  _shade_stress(ax, regime)
  ax.plot(returns.index, returns, color="#2a78d6", lw=0.6, label="daily return (%)")
  ax.set_title(title)
  ax.set_ylabel("log return (%)")
  ax.legend(loc="upper left")
  _style(ax)
  fig.tight_layout()
  if show:
    plt.show()
  return fig


def plot_forecasts(out, title, models=None, regime=None, show=True):
  """Annualised volatility forecasts, sqrt(variance * 252), one line per model."""
  models = [m for m in (models or out.columns.drop("realized")) if m in out.columns]
  fig, ax = plt.subplots(figsize=(13, 4))
  if regime is not None:
    _shade_stress(ax, regime.reindex(out.index))
  for m in models:
    ax.plot(out.index, np.sqrt(out[m] * 252), color=MODEL_COLORS[m], lw=1.2, label=m)
  top = np.nanpercentile(np.sqrt(out[models] * 252), 99.5) * 1.3
  ax.set_ylim(0, top)
  ax.set_title(title)
  ax.set_ylabel("annualised vol (%)")
  ax.legend(loc="upper left", ncol=4, fontsize=8)
  _style(ax)
  fig.tight_layout()
  if show:
    plt.show()
  return fig


def plot_var_backtest(out, model, alpha, title, show=True):
  var, _ = var_es(out[model], alpha)
  hits = var_hits(out["realized"], var).astype(bool)
  fig, ax = plt.subplots(figsize=(13, 4))
  ax.plot(out.index, out["realized"], color="#c3c2b7", lw=0.5, label="daily return (%)")
  ax.plot(out.index, -var, color=MODEL_COLORS[model], lw=1.2,
          label=f"-VaR {int((1 - alpha) * 100)}% ({model})")
  ax.scatter(out.index[hits], out["realized"][hits], s=14, color="#eb6834", zorder=3,
             label=f"hits: {hits.sum()} ({hits.mean():.2%}, expected {alpha:.0%})")
  ax.set_title(title)
  ax.set_ylabel("%")
  ax.legend(loc="lower left", fontsize=8)
  _style(ax)
  fig.tight_layout()
  if show:
    plt.show()
  return fig


def plot_metric_ranks(results, metric, show=True):
  """Rank of each model within each pair (1 = best) so pairs with different
  volatility levels can share one picture. Cell text is the metric value."""
  models = results.index.get_level_values("model").unique()
  tab = results[metric].unstack("model")[models]
  ranks = tab.rank(axis=1)
  fig, ax = plt.subplots(figsize=(1.6 * tab.shape[1] + 2, 0.6 * tab.shape[0] + 1.5))
  ax.imshow(ranks.values, cmap="Blues_r", vmin=1, vmax=tab.shape[1] + 2, aspect="auto")
  for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
      rk = int(ranks.iloc[i, j])
      ax.text(j, i, f"{tab.iloc[i, j]:.4f}\n#{rk}", ha="center", va="center", fontsize=8,
              color="white" if rk <= 3 else "black")
  ax.set_xticks(range(tab.shape[1]), tab.columns)
  ax.set_yticks(range(tab.shape[0]), [f"{c}/USD" for c in tab.index])
  ax.set_title(f"{metric} by pair and model (#1 = best, darker = better)")
  fig.tight_layout()
  if show:
    plt.show()
  return fig
