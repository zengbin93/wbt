from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ._common import COLOR_LONG, COLOR_SHORT, apply_default_layout, figure_to_html

if TYPE_CHECKING:
    from wbt.result import BacktestResult

_DIR_COLORS = {"多头": COLOR_LONG, "空头": COLOR_SHORT}


def plot_pairs_pnl_dist(
    result: BacktestResult,
    title: str | None = "盈亏比例分布",
    to_html: bool = False,
) -> go.Figure | str:
    """成交对盈亏比例分布直方图，按方向（多头/空头）分组。

    数据来自 BacktestResult.pairs_dist（已基于 Rust 聚合去重）。
    """
    fig = go.Figure()
    pd_ = result.pairs_dist
    if not pd_.pnl_pct:
        apply_default_layout(fig, title=title, height=400)
        return figure_to_html(fig) if to_html else fig

    for direction, pnl in pd_.pnl_pct.items():
        fig.add_trace(
            go.Histogram(
                x=pnl,
                name=direction,
                marker_color=_DIR_COLORS.get(direction, COLOR_LONG),
                opacity=0.7,
                nbinsx=40,
            )
        )

    apply_default_layout(fig, title=title, height=400)
    fig.update_xaxes(title_text="盈亏比例", tickformat=".1%")
    fig.update_yaxes(title_text="频次")
    return figure_to_html(fig) if to_html else fig


def plot_pairs_hold_dist(
    result: BacktestResult,
    title: str | None = "持仓K线数分布",
    to_html: bool = False,
) -> go.Figure | str:
    """成交对持仓K线数分布直方图，按方向（多头/空头）分组。

    数据来自 BacktestResult.pairs_dist（已基于 Rust 聚合去重）。
    """
    fig = go.Figure()
    pd_ = result.pairs_dist
    if not pd_.holds:
        apply_default_layout(fig, title=title, height=400)
        return figure_to_html(fig) if to_html else fig

    for direction, holds in pd_.holds.items():
        fig.add_trace(
            go.Histogram(
                x=holds,
                name=direction,
                marker_color=_DIR_COLORS.get(direction, COLOR_LONG),
                opacity=0.7,
                nbinsx=40,
            )
        )

    apply_default_layout(fig, title=title, height=400)
    fig.update_xaxes(title_text="持仓K线数")
    fig.update_yaxes(title_text="频次")
    return figure_to_html(fig) if to_html else fig
