from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ._common import (
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    CURVE_COLORS,
    add_year_boundaries,
    apply_default_layout,
    figure_to_html,
)

if TYPE_CHECKING:
    from wbt.result import BacktestResult

_FALLBACK_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]


def plot_cumulative_returns(
    result: BacktestResult,
    keys: list[str] | None = None,
    voladj: bool = False,
    title: str | None = None,
    to_html: bool = False,
) -> go.Figure | str:
    """绘制累计收益曲线。

    :param result: BacktestResult，直接消费 dates / curves / year_starts
    :param keys: 要绘制的曲线键，默认 ["多空"]；可选 多空/多头/空头/基准/超额
    :param voladj: 是否使用波动率归一后的曲线（result.curves_voladj），默认 False
    :param title: 图标题，None 时按 voladj 取默认值
    """
    if keys is None:
        keys = ["多空"]
    if title is None:
        title = "波动率归一累计收益" if voladj else "累计收益曲线"

    curves = result.curves_voladj if voladj else result.curves
    fig = go.Figure()
    for i, key in enumerate(keys):
        curve = curves.get(key)
        if curve is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=curve.cum,
                mode="lines",
                name=key,
                line={"color": CURVE_COLORS.get(key, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]), "width": 1.5},
            )
        )

    add_year_boundaries(fig, result.year_starts)
    apply_default_layout(fig, title=title, height=400)
    fig.update_yaxes(tickformat=".1%")
    return figure_to_html(fig) if to_html else fig


def plot_monthly_heatmap(
    result: BacktestResult,
    title: str | None = "月度收益热力图",
    to_html: bool = False,
) -> go.Figure | str:
    """绘制月度收益热力图（year × month）。"""
    m = result.monthly
    fig = go.Figure()

    if not m.years:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    fig.add_trace(
        go.Heatmap(
            x=[str(mo) for mo in m.months],
            y=[str(y) for y in m.years],
            z=m.z.tolist(),
            text=m.text.tolist(),
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"tickformat": ".1%"},
        )
    )

    apply_default_layout(fig, title=title, height=max(300, 60 * len(m.years) + 100))
    fig.update_xaxes(title_text="月份")
    fig.update_yaxes(title_text="年份")
    return figure_to_html(fig) if to_html else fig


def plot_symbol_returns(
    result: BacktestResult,
    title: str | None = "品种收益分布",
    to_html: bool = False,
) -> go.Figure | str:
    """品种累计收益水平条形图（已按收益升序）。"""
    sr = result.symbol_returns
    fig = go.Figure()
    if not sr.symbols:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    colors = [COLOR_NEGATIVE if v < 0 else COLOR_POSITIVE for v in sr.values]
    fig.add_trace(
        go.Bar(
            x=sr.values,
            y=sr.symbols,
            orientation="h",
            marker_color=colors,
        )
    )

    apply_default_layout(fig, title=title, height=max(300, 30 * len(sr.symbols) + 100))
    fig.update_xaxes(tickformat=".1%", title_text="累计收益率")
    return figure_to_html(fig) if to_html else fig
