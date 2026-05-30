from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import (
    COLOR_DRAWDOWN,
    COLOR_POSITIVE,
    COLOR_RETURN,
    COLOR_TOTAL,
    add_year_boundaries,
    apply_default_layout,
    figure_to_html,
)

if TYPE_CHECKING:
    from wbt.result import BacktestResult


def plot_drawdown(
    result: BacktestResult,
    key: str = "多空",
    title: str | None = "回撤分析",
    to_html: bool = False,
) -> go.Figure | str:
    """双轴图：回撤填充（左）+ 累计收益（右）。"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    curve = result.curves.get(key)
    if curve is None:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=curve.drawdown,
            fill="tozeroy",
            fillcolor=COLOR_DRAWDOWN,
            line={"color": "rgba(255,59,59,0.6)", "width": 1},
            name="回撤",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=curve.cum,
            mode="lines",
            line={"color": COLOR_TOTAL, "width": 1.5},
            name="累计收益",
        ),
        secondary_y=True,
    )

    add_year_boundaries(fig, result.year_starts)
    apply_default_layout(fig, title=title, height=400)
    fig.update_yaxes(title_text="回撤", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title_text="累计收益", tickformat=".1%", secondary_y=True)
    return figure_to_html(fig) if to_html else fig


def plot_daily_return_dist(
    result: BacktestResult,
    title: str | None = "日收益分布",
    to_html: bool = False,
) -> go.Figure | str:
    """日收益分布直方图，含均值与 ±2σ 竖线。"""
    rd = result.return_dist
    fig = go.Figure()

    if rd.values_pct.size == 0:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    fig.add_trace(
        go.Histogram(
            x=rd.values_pct,
            nbinsx=50,
            marker_color=COLOR_RETURN,
            opacity=0.7,
            name="日收益",
        )
    )

    mean_val = rd.mean_pct
    std_val = rd.std_pct
    for x_val, label, color in [
        (mean_val, f"均值 {mean_val:.3f}%", "orange"),
        (mean_val - 2 * std_val, f"-2σ {mean_val - 2 * std_val:.3f}%", "red"),
        (mean_val + 2 * std_val, f"+2σ {mean_val + 2 * std_val:.3f}%", "green"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )

    apply_default_layout(fig, title=title, height=400)
    fig.update_xaxes(title_text="日收益率 (%)")
    fig.update_yaxes(title_text="频次")
    return figure_to_html(fig) if to_html else fig


def plot_rolling_metrics(
    result: BacktestResult,
    title: str | None = None,
    to_html: bool = False,
) -> go.Figure | str:
    """滚动窗口指标双轴曲线：左轴年化收益/年化波动率(%)，右轴滚动夏普。

    数据来自 result.rolling（默认 252 日窗口），x 轴为窗口结束日。
    """
    rm = result.rolling
    if title is None:
        title = f"滚动指标（{rm.window}日）"
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if rm.edt.size == 0:
        apply_default_layout(fig, title=title, height=400)
        return figure_to_html(fig) if to_html else fig

    fig.add_trace(
        go.Scatter(x=rm.edt, y=rm.annual_return, mode="lines", name="滚动年化收益", line={"color": COLOR_POSITIVE}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=rm.edt, y=rm.annual_vol, mode="lines", name="滚动年化波动率", line={"color": "#7f8c8d"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=rm.edt, y=rm.sharpe, mode="lines", name="滚动夏普", line={"color": COLOR_TOTAL, "width": 1.5}),
        secondary_y=True,
    )

    add_year_boundaries(fig, result.year_starts)
    apply_default_layout(fig, title=title, height=400)
    fig.update_yaxes(title_text="年化收益 / 波动率", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title_text="夏普", secondary_y=True)
    return figure_to_html(fig) if to_html else fig
