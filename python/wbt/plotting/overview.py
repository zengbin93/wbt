from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import (
    COLOR_DRAWDOWN,
    COLOR_RETURN,
    COLOR_TOTAL,
    apply_default_layout,
    figure_to_html,
)


def plot_backtest_overview(
    daily_return: pd.DataFrame,
    col: str = "total",
    title: str | None = "回测概览",
    to_html: bool = False,
) -> go.Figure | str:
    """2x2 subplots overview: drawdown+cumulative, daily return histogram, monthly heatmap.

    Layout:
      - Top-left  (row=1, col=1): drawdown fill + cumulative return (secondary_y)
      - Top-right (row=1, col=2): daily return histogram
      - Bottom    (row=2, col=1..2 span): monthly heatmap

    :param daily_return: DataFrame from wb.daily_return
    :param col: column to analyse
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
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

    if daily_return.empty or col not in daily_return.columns:
        apply_default_layout(fig, title=title, height=700)
        return figure_to_html(fig) if to_html else fig

    df = daily_return.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    cumsum = df[col].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max

    # --- Top-left: drawdown + cumulative ---
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=drawdown,
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
            x=df["date"],
            y=cumsum,
            mode="lines",
            line={"color": COLOR_TOTAL, "width": 1.5},
            name="累计收益",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # --- Top-right: histogram ---
    series = df[col].dropna() * 100
    float(series.mean())
    float(series.std())

    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=40,
            marker_color=COLOR_RETURN,
            opacity=0.7,
            name="日收益",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # --- Bottom: monthly heatmap ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.groupby(["year", "month"])[col].sum().unstack(fill_value=0)
    years = pivot.index.tolist()
    months = [str(m) for m in pivot.columns.tolist()]
    z = pivot.values.tolist()
    text = [[f"{v * 100:.2f}%" for v in row] for row in z]

    fig.add_trace(
        go.Heatmap(
            x=months,
            y=[str(y) for y in years],
            z=z,
            text=text,
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
    stats: dict,
    title: str | None = "绩效指标",
    to_html: bool = False,
) -> go.Figure | str:
    """Display stats dict as a table with color-coded cells.

    Positive numeric values get a red tint (COLOR_POSITIVE), negative get green tint (COLOR_NEGATIVE).

    :param stats: dict from wb.stats
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    fig = go.Figure()

    if not stats:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    keys = list(stats.keys())
    values = [stats[k] for k in keys]

    def _fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _cell_color(v: object) -> str:
        if isinstance(v, (int, float)):
            if v > 0:
                return "rgba(231,76,60,0.12)"
            if v < 0:
                return "rgba(46,204,113,0.12)"
        return "white"

    fmt_values = [_fmt(v) for v in values]
    cell_colors = [_cell_color(v) for v in values]

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
                "values": [keys, fmt_values],
                "fill_color": [["white"] * len(keys), cell_colors],
                "align": ["left", "right"],
                "font_size": 12,
            },
        )
    )

    apply_default_layout(fig, title=title, height=max(400, 30 * len(keys) + 100))
    return figure_to_html(fig) if to_html else fig


def plot_long_short_comparison(
    daily_return: pd.DataFrame,
    stats: dict,
    long_stats: dict,
    short_stats: dict,
    title: str | None = "多空对比",
    to_html: bool = False,
) -> go.Figure | str:
    """Two panels: cumulative return curves (left), comparison table (right).

    :param daily_return: DataFrame from wb.daily_return (must have 'total' column)
    :param stats: overall stats dict from wb.stats
    :param long_stats: long stats dict from wb.long_stats
    :param short_stats: short stats dict from wb.short_stats
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("累计收益曲线", "关键指标对比"),
    )

    if not daily_return.empty:
        df = daily_return.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for col, color, name in [
            ("total", COLOR_TOTAL, "多空"),
        ]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[col].cumsum(),
                        mode="lines",
                        line={"color": color, "width": 1.5},
                        name=name,
                    ),
                    row=1,
                    col=1,
                )

    # Key metrics to compare
    _metric_keys = ["年化", "夏普", "最大回撤", "卡玛", "日胜率", "绝对收益"]

    def _get(d: dict, k: str) -> str:
        v = d.get(k, "N/A")
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    metric_names = [k for k in _metric_keys if k in stats or k in long_stats or k in short_stats]
    col_total = [_get(stats, k) for k in metric_names]
    col_long = [_get(long_stats, k) for k in metric_names]
    col_short = [_get(short_stats, k) for k in metric_names]

    fig.add_trace(
        go.Table(
            header={
                "values": ["指标", "多空", "多头", "空头"],
                "fill_color": "#3498db",
                "font_color": "white",
                "align": "center",
                "font_size": 12,
            },
            cells={
                "values": [metric_names, col_total, col_long, col_short],
                "align": ["left", "right", "right", "right"],
                "font_size": 11,
            },
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        title=title,
        title_font_size=14,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
    )
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    return figure_to_html(fig) if to_html else fig
