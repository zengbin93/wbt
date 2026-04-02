from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Color constants
COLOR_LONG = "#e74c3c"
COLOR_SHORT = "#2ecc71"
COLOR_TOTAL = "#3498db"
COLOR_DRAWDOWN = "rgba(255,59,59,0.15)"
COLOR_RETURN = "#1f77b4"
COLOR_POSITIVE = "#e74c3c"
COLOR_NEGATIVE = "#2ecc71"


def figure_to_html(fig: go.Figure, include_plotlyjs: bool = True) -> str:
    """Convert a plotly Figure to an HTML string."""
    return fig.to_html(include_plotlyjs=include_plotlyjs)


def add_year_boundaries(fig: go.Figure, dates: pd.Series) -> None:
    """Add vertical dashed lines at year start boundaries."""
    if dates is None or len(dates) == 0:
        return
    dates = pd.to_datetime(dates)
    years_seen: set[int] = set()
    for dt in dates:
        year = dt.year
        if year not in years_seen:
            years_seen.add(year)
            fig.add_vline(
                x=str(dt.date()),
                line_dash="dash",
                line_color="rgba(100,100,100,0.4)",
                line_width=1,
            )


def apply_default_layout(fig: go.Figure, title: str | None = None, height: int = 500) -> None:
    """Apply consistent plotly_white template styling to a figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        title_font_size=14,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=60),
    )
