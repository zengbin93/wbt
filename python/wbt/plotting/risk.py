from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._common import (
    COLOR_DRAWDOWN,
    COLOR_RETURN,
    COLOR_TOTAL,
    add_year_boundaries,
    apply_default_layout,
    figure_to_html,
)


def plot_drawdown(
    daily_return: pd.DataFrame,
    col: str = "total",
    title: str | None = "回撤分析",
    to_html: bool = False,
) -> go.Figure | str:
    """Dual y-axis chart: drawdown fill area (left) + cumulative return line (right).

    :param daily_return: DataFrame from wb.daily_return
    :param col: column to analyse
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if daily_return.empty or col not in daily_return.columns:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    df = daily_return.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    cumsum = df[col].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max

    # Drawdown fill on primary y
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=drawdown,
            fill="tozeroy",
            fillcolor=COLOR_DRAWDOWN,
            line=dict(color="rgba(255,59,59,0.6)", width=1),
            name="回撤",
        ),
        secondary_y=False,
    )

    # Cumulative return on secondary y
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=cumsum,
            mode="lines",
            line=dict(color=COLOR_TOTAL, width=1.5),
            name="累计收益",
        ),
        secondary_y=True,
    )

    add_year_boundaries(fig, df["date"])
    apply_default_layout(fig, title=title, height=400)
    fig.update_yaxes(title_text="回撤", tickformat=".1%", secondary_y=False)
    fig.update_yaxes(title_text="累计收益", tickformat=".1%", secondary_y=True)
    return figure_to_html(fig) if to_html else fig


def plot_daily_return_dist(
    daily_return: pd.DataFrame,
    col: str = "total",
    title: str | None = "日收益分布",
    to_html: bool = False,
) -> go.Figure | str:
    """Histogram of daily returns with mean and ±2σ lines.

    :param daily_return: DataFrame from wb.daily_return
    :param col: column to plot
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    fig = go.Figure()

    if daily_return.empty or col not in daily_return.columns:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    series = daily_return[col].dropna() * 100  # convert to %

    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=50,
            marker_color=COLOR_RETURN,
            opacity=0.7,
            name="日收益",
        )
    )

    mean_val = float(series.mean())
    std_val = float(series.std())

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
