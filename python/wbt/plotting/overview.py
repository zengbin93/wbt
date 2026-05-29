from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import (
    COLOR_DRAWDOWN,
    COLOR_RETURN,
    COLOR_TOTAL,
    CURVE_COLORS,
    add_year_boundaries,
    apply_default_layout,
    figure_to_html,
    fmt_cell,
)

if TYPE_CHECKING:
    from wbt.result import BacktestResult


def plot_backtest_overview(
    result: BacktestResult,
    title: str | None = "回测概览",
    to_html: bool = False,
) -> go.Figure | str:
    """2x2 概览：回撤+累计 / 日收益分布 / 月度热力图。"""
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=("回撤 & 累计收益", "日收益分布", "月度收益热力图"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    curve = result.curves.get("多空")
    if curve is None:
        apply_default_layout(fig, title=title, height=700)
        return figure_to_html(fig) if to_html else fig

    # 回撤 + 累计
    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=curve.drawdown,
            fill="tozeroy",
            fillcolor=COLOR_DRAWDOWN,
            line={"color": "rgba(255,59,59,0.6)", "width": 1},
            name="回撤",
        ),
        row=1,
        col=1,
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
        row=1,
        col=1,
        secondary_y=True,
    )

    # 日收益分布
    fig.add_trace(
        go.Histogram(
            x=result.return_dist.values_pct,
            nbinsx=40,
            marker_color=COLOR_RETURN,
            opacity=0.7,
            name="日收益",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 月度热力图
    m = result.monthly
    fig.add_trace(
        go.Heatmap(
            x=[str(mo) for mo in m.months],
            y=[str(y) for y in m.years],
            z=m.z.tolist(),
            text=m.text.tolist(),
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            showscale=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=700,
        title=title,
        title_font_size=14,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
    )
    return figure_to_html(fig) if to_html else fig


def plot_colored_table(
    result: BacktestResult,
    title: str | None = "绩效指标",
    to_html: bool = False,
) -> go.Figure | str:
    """将 result.stats 渲染为带颜色的表格。正值红、负值绿。"""
    fig = go.Figure()
    stats = result.stats
    if not stats:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    keys = list(stats.keys())
    values = [stats[k] for k in keys]

    def _cell_color(v: object) -> str:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if v > 0:
                return "rgba(231,76,60,0.12)"
            if v < 0:
                return "rgba(46,204,113,0.12)"
        return "white"

    fig.add_trace(
        go.Table(
            header={
                "values": ["指标", "数值"],
                "fill_color": "#3498db",
                "font_color": "white",
                "align": "center",
                "font_size": 13,
            },
            cells={
                "values": [keys, [fmt_cell(v) for v in values]],
                "fill_color": [["white"] * len(keys), [_cell_color(v) for v in values]],
                "align": ["left", "right"],
                "font_size": 12,
            },
        )
    )

    apply_default_layout(fig, title=title, height=max(400, 30 * len(keys) + 100))
    return figure_to_html(fig) if to_html else fig


def plot_long_short_comparison(
    result: BacktestResult,
    title: str | None = "多空对比",
    to_html: bool = False,
) -> go.Figure | str:
    """三行子图：原始累计曲线 / 波动率归一累计曲线（区分）+ 超额 / 指标对比表。"""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("累计收益曲线（原始）", "波动率归一后累计收益", "关键指标对比"),
        vertical_spacing=0.08,
        row_heights=[0.35, 0.35, 0.30],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
    )

    panel_keys = ["多空", "多头", "空头", "基准"]

    # 上图：原始累计
    for key in panel_keys:
        curve = result.curves.get(key)
        if curve is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=curve.cum,
                name=key,
                mode="lines",
                line={"color": CURVE_COLORS.get(key)},
                legendgroup=key,
            ),
            row=1,
            col=1,
        )
    add_year_boundaries(fig, result.year_starts, row=1, col=1)

    # 中图：波动率归一累计（名称带「归一」以区分原始）
    voladj = result.curves_voladj
    for key in panel_keys:
        curve = voladj.get(key)
        if curve is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=curve.cum,
                name=f"{key}(归一)",
                mode="lines",
                showlegend=False,
                line={"color": CURVE_COLORS.get(key)},
                legendgroup=key,
            ),
            row=2,
            col=1,
        )
    # 策略超额（归一口径）：curves['超额'] = 整体策略(多空) − 基准，非多头超额
    alpha_curve = voladj.get("超额")
    if alpha_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=result.dates,
                y=alpha_curve.cum,
                name="策略超额(归一)",
                mode="lines",
                line={"color": "#FF1493", "width": 3.0},
                legendgroup="超额",
            ),
            row=2,
            col=1,
        )
    add_year_boundaries(fig, result.year_starts, row=2, col=1)

    # 下表：多空 / 多头 / 空头 指标对比
    metric_keys = ["年化收益", "夏普比率", "卡玛比率", "最大回撤", "年化波动率", "日胜率"]
    sides = {"多空": result.stats, **result.stats_by_side}
    side_order = ["多空", "多头", "空头", "基准", "超额"]
    side_order = [s for s in side_order if s in sides]

    def _get(d: dict, k: str) -> str:
        return fmt_cell(d.get(k, "N/A"))

    header = ["指标", *side_order]
    columns: list[list[str]] = [metric_keys]
    for s in side_order:
        columns.append([_get(sides[s], k) for k in metric_keys])

    fig.add_trace(
        go.Table(
            header={
                "values": header,
                "fill_color": "#3498db",
                "font_color": "white",
                "align": "center",
                "font_size": 12,
            },
            cells={"values": columns, "align": ["left", *["right"] * len(side_order)], "font_size": 11},
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=1100,
        title=title,
        title_font_size=14,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="累计收益", tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title_text="累计收益", tickformat=".1%", row=2, col=1)
    return figure_to_html(fig) if to_html else fig
