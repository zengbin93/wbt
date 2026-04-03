from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ._common import (
    COLOR_LONG,
    COLOR_SHORT,
    COLOR_TOTAL,
    add_year_boundaries,
    apply_default_layout,
    figure_to_html,
)

_COL_COLORS = [COLOR_TOTAL, COLOR_LONG, COLOR_SHORT, "#9b59b6", "#f39c12", "#1abc9c"]


def plot_cumulative_returns(
    daily_return: pd.DataFrame,
    cols: list[str] | None = None,
    title: str | None = "累计收益曲线",
    to_html: bool = False,
) -> go.Figure | str:
    """Plot cumulative return curves.

    :param daily_return: DataFrame from wb.daily_return (columns: date, symbol..., total)
    :param cols: columns to plot; defaults to ["total"]
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    df = daily_return.copy()
    if df.empty:
        fig = go.Figure()
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    if cols is None:
        cols = ["total"]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    fig = go.Figure()
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        color = _COL_COLORS[i % len(_COL_COLORS)]
        cumulative = df[col].cumsum()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=cumulative,
                mode="lines",
                name=col,
                line={"color": color, "width": 1.5},
            )
        )

    add_year_boundaries(fig, df["date"])
    apply_default_layout(fig, title=title, height=400)
    fig.update_yaxes(tickformat=".1%")
    return figure_to_html(fig) if to_html else fig


def plot_monthly_heatmap(
    daily_return: pd.DataFrame,
    col: str = "total",
    title: str | None = "月度收益热力图",
    to_html: bool = False,
) -> go.Figure | str:
    """Plot monthly return heatmap (year × month).

    :param daily_return: DataFrame from wb.daily_return
    :param col: column to aggregate
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    df = daily_return.copy()
    fig = go.Figure()

    if df.empty or col not in df.columns:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    df["date"] = pd.to_datetime(df["date"])
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
            colorbar={"tickformat": ".1%"},
        )
    )

    apply_default_layout(fig, title=title, height=max(300, 60 * len(years) + 100))
    fig.update_xaxes(title_text="月份")
    fig.update_yaxes(title_text="年份")
    return figure_to_html(fig) if to_html else fig


def plot_symbol_returns(
    dailys: pd.DataFrame,
    title: str | None = "品种收益分布",
    to_html: bool = False,
) -> go.Figure | str:
    """Horizontal bar chart of total return per symbol sorted ascending.

    :param dailys: DataFrame from wb.dailys (has 'symbol' and 'return' columns)
    :param title: chart title
    :param to_html: if True, return HTML string instead of Figure
    """
    fig = go.Figure()
    if dailys.empty or "symbol" not in dailys.columns or "return" not in dailys.columns:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    sr = dailys.groupby("symbol")["return"].sum().sort_values(ascending=True)
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in sr.values]

    fig.add_trace(
        go.Bar(
            x=sr.values,
            y=sr.index.tolist(),
            orientation="h",
            marker_color=colors,
        )
    )

    apply_default_layout(fig, title=title, height=max(300, 30 * len(sr) + 100))
    fig.update_xaxes(tickformat=".1%", title_text="累计收益率")
    return figure_to_html(fig) if to_html else fig
