from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ._common import COLOR_LONG, COLOR_SHORT, apply_default_layout, figure_to_html, fmt_cell

if TYPE_CHECKING:
    from wbt.result import BacktestResult

_COMPARE_METRICS = ["年化收益", "夏普比率", "卡玛比率", "最大回撤", "年化波动率", "日胜率"]
_COMPARE_SIDES = ["多空", "多头", "空头", "基准", "超额"]


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


def plot_stats_comparison(
    result: BacktestResult,
    title: str | None = "关键指标对比",
    to_html: bool = False,
) -> go.Figure | str:
    """多空 / 多头 / 空头 / 基准 / 超额 的关键指标横向对比表。

    消费 result.stats（多空）+ result.stats_by_side（其余侧），零数据转换。
    """
    fig = go.Figure()
    sides = {"多空": result.stats, **result.stats_by_side}
    side_order = [s for s in _COMPARE_SIDES if s in sides]
    if not side_order:
        apply_default_layout(fig, title=title)
        return figure_to_html(fig) if to_html else fig

    columns: list[list[str]] = [_COMPARE_METRICS]
    for s in side_order:
        columns.append([fmt_cell(sides[s].get(k, "N/A")) for k in _COMPARE_METRICS])

    fig.add_trace(
        go.Table(
            header={
                "values": ["指标", *side_order],
                "fill_color": "#3498db",
                "font_color": "white",
                "align": "center",
                "font_size": 12,
            },
            cells={
                "values": columns,
                "align": ["left", *["right"] * len(side_order)],
                "font_size": 11,
            },
        )
    )
    apply_default_layout(fig, title=title, height=max(300, 30 * len(_COMPARE_METRICS) + 120))
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
