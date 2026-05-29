from __future__ import annotations

from .overview import plot_backtest_overview, plot_colored_table, plot_long_short_comparison
from .returns import plot_cumulative_returns, plot_monthly_heatmap, plot_symbol_returns
from .risk import plot_daily_return_dist, plot_drawdown
from .tables import plot_drawdowns_table, plot_key_trades, plot_verdict
from .trades import plot_pairs_analysis

__all__ = [
    "plot_backtest_overview",
    "plot_colored_table",
    "plot_cumulative_returns",
    "plot_daily_return_dist",
    "plot_drawdown",
    "plot_drawdowns_table",
    "plot_key_trades",
    "plot_long_short_comparison",
    "plot_monthly_heatmap",
    "plot_pairs_analysis",
    "plot_symbol_returns",
    "plot_verdict",
]
