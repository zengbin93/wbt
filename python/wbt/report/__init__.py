"""权重回测 HTML 报告生成模块。
"""

from __future__ import annotations

from ._generator import (
    LongShortComparisonChart,
    generate_backtest_report,
)
from ._plot_backtest import (
    get_performance_metrics_cards,
    plot_backtest_stats,
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_distribution,
    plot_drawdown_analysis,
    plot_long_short_comparison,
    plot_monthly_heatmap,
)
from .html_builder import HtmlReportBuilder

__all__ = [
    "HtmlReportBuilder",
    "LongShortComparisonChart",
    "generate_backtest_report",
    "get_performance_metrics_cards",
    "plot_backtest_stats",
    "plot_colored_table",
    "plot_cumulative_returns",
    "plot_daily_return_distribution",
    "plot_drawdown_analysis",
    "plot_long_short_comparison",
    "plot_monthly_heatmap",
]
