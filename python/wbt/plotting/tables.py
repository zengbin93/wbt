from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ._common import COLOR_LONG, COLOR_SHORT, apply_default_layout, figure_to_html, fmt_cell

if TYPE_CHECKING:
    from wbt.result import BacktestResult


def plot_key_trades(
    result: BacktestResult,
    title: str | None = "关键交易（每年最赚/最亏）",
    to_html: bool = False,
) -> go.Figure | str:
    """每年最赚/最亏关键交易条形图。

    判断策略是否「赚了不该赚的钱、亏了不该亏的钱」：x 为盈亏比例，按年分组、
    best（红）/worst（绿）并列。数据来自 result.key_trades（已聚合去重）。
    """
    kt = result.key_trades
    fig = go.Figure()

    labels: list[str] = []
    pnls: list[float] = []
    colors: list[str] = []
    hovertexts: list[str] = []

    for year in sorted(set(kt.best) | set(kt.worst)):
        for trades, glyph, color in (
            (kt.best.get(year, []), "↑", COLOR_LONG),
            (kt.worst.get(year, []), "↓", COLOR_SHORT),
        ):
            for trade in trades:
                labels.append(f"{year} {trade.symbol}{glyph}")
                pnls.append(trade.pnl)
                colors.append(color)
                hovertexts.append(
                    f"{trade.symbol} {trade.direction}<br>{trade.open_dt} → {trade.close_dt}"
                    f"<br>盈亏 {trade.pnl:.2%}，持仓 {trade.hold_bars} 根，合并 {trade.count} 笔"
                )

    if not labels:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    fig.add_trace(
        go.Bar(
            x=pnls,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertext=hovertexts,
            hoverinfo="text",
        )
    )
    apply_default_layout(fig, title=title, height=max(300, 22 * len(labels) + 120))
    fig.update_xaxes(tickformat=".1%", title_text="盈亏比例")
    return figure_to_html(fig) if to_html else fig


def plot_drawdowns_table(
    result: BacktestResult,
    title: str | None = "回撤明细",
    to_html: bool = False,
) -> go.Figure | str:
    """top_drawdowns 明细表（审核页面）。"""
    rows = result.drawdowns
    fig = go.Figure()
    if not rows:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    headers = list(rows[0].keys())
    columns = [[fmt_cell(r.get(h, "")) for r in rows] for h in headers]
    fig.add_trace(
        go.Table(
            header={"values": headers, "fill_color": "#3498db", "font_color": "white", "align": "center"},
            cells={"values": columns, "align": "center", "font_size": 11},
        )
    )
    apply_default_layout(fig, title=title, height=max(300, 30 * len(rows) + 120))
    return figure_to_html(fig) if to_html else fig


def plot_verdict(
    result: BacktestResult,
    title: str | None = "策略判定",
    to_html: bool = False,
) -> go.Figure | str:
    """is_good_strategy 判定 + 年度指标表（审核页面）。"""
    v = result.verdict
    fig = go.Figure()

    yearly = v.get("yearly_metrics") or []
    if yearly and isinstance(yearly, list) and isinstance(yearly[0], dict):
        headers = list(yearly[0].keys())
        columns = [[fmt_cell(row.get(h, "")) for row in yearly] for h in headers]
        fig.add_trace(
            go.Table(
                header={"values": headers, "fill_color": "#3498db", "font_color": "white", "align": "center"},
                cells={"values": columns, "align": "center", "font_size": 11},
            )
        )

    is_good = bool(v.get("is_good", False))
    reason = str(v.get("reason", "") or "")
    verdict_text = f"<b>{'✅ 可用' if is_good else '❌ 不可用'}</b>"
    if reason:
        verdict_text += f"  ·  {reason}"
    fig.add_annotation(
        text=verdict_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.08,
        showarrow=False,
        font={"size": 14, "color": "#2ecc71" if is_good else "#e74c3c"},
    )

    apply_default_layout(fig, title=title, height=max(300, 28 * (len(yearly) + 2) + 120))
    return figure_to_html(fig) if to_html else fig
