from __future__ import annotations

from .returns import plot_cumulative_returns, plot_monthly_heatmap, plot_symbol_returns, plot_yearly_returns
from .risk import plot_daily_return_dist, plot_drawdown, plot_rolling_metrics
from .tables import (
    plot_colored_table,
    plot_drawdowns_table,
    plot_key_trades,
    plot_segment_comparison,
    plot_stats_comparison,
    plot_verdict,
)
from .trades import plot_pairs_hold_dist, plot_pairs_pnl_dist

__all__ = [
    "plot_colored_table",
    "plot_cumulative_returns",
    "plot_daily_return_dist",
    "plot_drawdown",
    "plot_drawdowns_table",
    "plot_key_trades",
    "plot_monthly_heatmap",
    "plot_pairs_hold_dist",
    "plot_pairs_pnl_dist",
    "plot_rolling_metrics",
    "plot_segment_comparison",
    "plot_stats_comparison",
    "plot_symbol_returns",
    "plot_verdict",
    "plot_yearly_returns",
]
