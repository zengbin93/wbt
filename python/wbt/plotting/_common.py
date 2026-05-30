from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

# Color constants
COLOR_LONG = "#e74c3c"
COLOR_SHORT = "#2ecc71"
COLOR_TOTAL = "#3498db"
COLOR_DRAWDOWN = "rgba(255,59,59,0.15)"
COLOR_RETURN = "#1f77b4"
COLOR_POSITIVE = "#e74c3c"
COLOR_NEGATIVE = "#2ecc71"

# curves 各序列固定配色
CURVE_COLORS = {
    "多空": COLOR_TOTAL,
    "多头": COLOR_LONG,
    "空头": COLOR_SHORT,
    "基准": "#7f8c8d",
    "超额": "#9b59b6",
}


def figure_to_html(fig: go.Figure, include_plotlyjs: bool = True) -> str:
    """Convert a plotly Figure to an HTML string."""
    return fig.to_html(include_plotlyjs=include_plotlyjs)


def fmt_cell(v: object) -> str:
    """统一的表格单元格格式化：float 保留 4 位，其余 str()。"""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# 字段名 → 格式化类别的关键词（按优先级匹配，先整数/计数，再 BP/比率，最后百分比）
_COUNT_KEYS = ("天数", "次数", "K线数", "间隔", "数量", "days", "count")
_YEAR_KEYS = ("year", "年份")
_RATIO_KEYS = ("夏普", "卡玛", "盈亏比", "盈亏平衡", "赢面")
_PCT_KEYS = ("收益", "回撤", "波动", "胜率", "占比", "回报率", "覆盖", "abs_return", "alpha_return")


def fmt_value(key: str, v: object) -> str:
    """按字段名语义格式化单元格值。

    比率类→百分比（``.2%``），计数/天数→整数千分位，年份→无千分位整数，
    夏普/卡玛/盈亏比→``.2f``，单笔收益(BP)→``.2f``，None/NaN→「—」，bool→是/否。
    无法归类的数值回退 ``.4f``，非数值原样 ``str``。
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float) and math.isnan(v):
        return "—"
    if not isinstance(v, (int, float)):
        return str(v)

    k = str(key)
    if any(t in k for t in _YEAR_KEYS):
        return f"{v:.0f}"
    if any(t in k for t in _COUNT_KEYS):
        return f"{v:,.0f}"
    if "单笔收益" in k:  # BP 单位
        return f"{v:.2f}"
    if any(t in k for t in _RATIO_KEYS):
        return f"{v:.2f}"
    if any(t in k for t in _PCT_KEYS):
        return f"{v:.2%}"
    return f"{v:.4f}"


def add_year_boundaries(
    fig: go.Figure,
    year_starts: np.ndarray,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """在每年起始位置画竖直分隔线；``year_starts`` 直接来自 BacktestResult。"""
    if year_starts is None or len(year_starts) == 0:
        return
    for dt in year_starts:
        fig.add_vline(
            x=dt,
            line_dash="dash",
            line_color="rgba(100,100,100,0.4)",
            line_width=1,
            row=row,
            col=col,
        )


def apply_default_layout(fig: go.Figure, title: str | None = None, height: int = 500) -> None:
    """统一布局：透明画布（随容器/报告主题着色），plotly_white 提供浅色默认网格/字体。

    画布设为透明后，单图嵌入报告时由 HTML 面板背景透出；报告侧的主题切换脚本
    再通过 ``Plotly.relayout`` 改写字体/网格颜色以适配明暗主题。独立使用（白底页面）
    时透明等同白底，无副作用。
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        title_font_size=14,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # 工具栏背景透明，避免与深/浅主题面板形成不协调的灰盒（颜色由报告侧主题脚本再调）
        modebar={"bgcolor": "rgba(0,0,0,0)", "color": "rgba(130,140,160,0.55)", "activecolor": "#2f5fef"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )
