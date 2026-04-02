from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import COLOR_LONG, COLOR_SHORT, apply_default_layout, figure_to_html


def plot_pairs_analysis(
    pairs: pd.DataFrame,
    title: str | None = "交易分析",
    to_html: bool = False,
) -> go.Figure | str:
    """Two subplots: P&L distribution by direction (left), holding period distribution (right).

    :param pairs: DataFrame from wb.pairs (has '盈亏比例', '持仓K线数', '交易方向' or '方向')
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("盈亏比例分布", "持仓K线数分布"))

    if pairs is None or pairs.empty:
        apply_default_layout(fig, title=title, height=400)
        return figure_to_html(fig) if to_html else fig

    # Detect direction column name
    dir_col = "交易方向" if "交易方向" in pairs.columns else ("方向" if "方向" in pairs.columns else None)
    pnl_col = "盈亏比例" if "盈亏比例" in pairs.columns else None
    hold_col = "持仓K线数" if "持仓K线数" in pairs.columns else None

    directions = pairs[dir_col].unique().tolist() if dir_col else [None]

    _dir_colors = {"多头": COLOR_LONG, "空头": COLOR_SHORT}

    for direction in directions:
        if direction is None:
            subset = pairs
            color = COLOR_LONG
            name = "全部"
        else:
            subset = pairs[pairs[dir_col] == direction]
            color = _dir_colors.get(str(direction), COLOR_LONG)
            name = str(direction)

        if pnl_col and pnl_col in subset.columns:
            fig.add_trace(
                go.Histogram(
                    x=subset[pnl_col] * 100,
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=40,
                ),
                row=1,
                col=1,
            )

        if hold_col and hold_col in subset.columns:
            fig.add_trace(
                go.Histogram(
                    x=subset[hold_col],
                    name=name,
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
