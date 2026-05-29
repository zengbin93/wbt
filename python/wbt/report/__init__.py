"""权重回测 HTML 报告生成模块。

绘图函数已统一到 :mod:`wbt.plotting`（以 BacktestResult 为标准输入）；
本模块只负责报告的组合与布局。
"""

from __future__ import annotations

from wbt.plotting import (
    plot_colored_table,
    plot_cumulative_returns,
    plot_long_short_comparison,
    plot_monthly_heatmap,
)

from ._generator import (
    generate_backtest_report,
    get_performance_metrics_cards,
)
from .html_builder import HtmlReportBuilder

__all__ = [
    "HtmlReportBuilder",
    "generate_backtest_report",
    "get_performance_metrics_cards",
    "plot_colored_table",
    "plot_cumulative_returns",
    "plot_long_short_comparison",
    "plot_monthly_heatmap",
]
