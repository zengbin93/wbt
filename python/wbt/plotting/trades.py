from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import COLOR_LONG, COLOR_SHORT, apply_default_layout, figure_to_html

if TYPE_CHECKING:
    from wbt.result import BacktestResult

_DIR_COLORS = {"多头": COLOR_LONG, "空头": COLOR_SHORT}


def plot_pairs_analysis(
    result: BacktestResult,
    title: str | None = "交易分析",
    to_html: bool = False,
) -> go.Figure | str:
    """两子图：盈亏比例分布（左）+ 持仓K线数分布（右），按方向分组。

    数据来自 BacktestResult.pairs_dist（已基于 Rust 聚合去重）。
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("盈亏比例分布", "持仓K线数分布"))

    pd_ = result.pairs_dist
    if not pd_.pnl_pct:
        apply_default_layout(fig, title=title, height=400)
        return figure_to_html(fig) if to_html else fig

    for direction, pnl in pd_.pnl_pct.items():
        color = _DIR_COLORS.get(direction, COLOR_LONG)
        fig.add_trace(
            go.Histogram(x=pnl, name=direction, marker_color=color, opacity=0.7, nbinsx=40),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=pd_.holds.get(direction),
                name=direction,
                marker_color=color,
                opacity=0.7,
                nbinsx=40,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    apply_default_layout(fig, title=title, height=400)
    fig.update_xaxes(title_text="盈亏比例 (%)", row=1, col=1)
    fig.update_xaxes(title_text="持仓K线数", row=1, col=2)
    return figure_to_html(fig) if to_html else fig
