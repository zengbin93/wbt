from __future__ import annotations

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
    """Apply consistent plotly_white template styling to a figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        title_font_size=14,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )
