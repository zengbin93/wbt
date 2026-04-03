from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

from wbt.plotting import (  # noqa: E402
    plot_backtest_overview,
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_long_short_comparison,
    plot_monthly_heatmap,
    plot_pairs_analysis,
    plot_symbol_returns,
)

# ---------------------------------------------------------------------------
# Fixtures are imported from conftest.py (sample_dfw and wb)
# ---------------------------------------------------------------------------


class TestPlotCumulativeReturns:
    def test_returns_figure(self, wb):
        fig = plot_cumulative_returns(wb.daily_return)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_multiple_cols(self, wb):
        fig = plot_cumulative_returns(wb.daily_return, cols=["SYM_A", "SYM_B", "total"])
        assert len(fig.data) >= 1

    def test_cumulative_returns_trace_count(self, wb):
        fig = plot_cumulative_returns(wb.daily_return)
        assert len(fig.data) >= 1  # At least one trace

    def test_to_html(self, wb):
        html = plot_cumulative_returns(wb.daily_return, to_html=True)
        assert isinstance(html, str)
        assert "<div" in html

    def test_empty_df(self):
        import pandas as pd

        fig = plot_cumulative_returns(pd.DataFrame(columns=["date", "total"]))
        assert isinstance(fig, go.Figure)


class TestPlotMonthlyHeatmap:
    def test_returns_figure(self, wb):
        fig = plot_monthly_heatmap(wb.daily_return)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_monthly_heatmap(wb.daily_return, to_html=True)
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_monthly_heatmap(pd.DataFrame(columns=["date", "total"]))
        assert isinstance(fig, go.Figure)


class TestPlotSymbolReturns:
    def test_returns_figure(self, wb):
        fig = plot_symbol_returns(wb.dailys)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_symbol_returns(wb.dailys, to_html=True)
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_symbol_returns(pd.DataFrame(columns=["symbol", "return"]))
        assert isinstance(fig, go.Figure)


class TestPlotDrawdown:
    def test_returns_figure(self, wb):
        fig = plot_drawdown(wb.daily_return)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)
        # Should have 2 traces: drawdown + cumulative
        assert len(fig.data) == 2

    def test_drawdown_two_traces(self, wb):
        fig = plot_drawdown(wb.daily_return)
        assert len(fig.data) == 2  # drawdown fill + cumulative line

    def test_to_html(self, wb):
        html = plot_drawdown(wb.daily_return, to_html=True)
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_drawdown(pd.DataFrame(columns=["date", "total"]))
        assert isinstance(fig, go.Figure)


class TestPlotDailyReturnDist:
    def test_returns_figure(self, wb):
        fig = plot_daily_return_dist(wb.daily_return)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_daily_return_dist(wb.daily_return, to_html=True)
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_daily_return_dist(pd.DataFrame(columns=["date", "total"]))
        assert isinstance(fig, go.Figure)


class TestPlotPairsAnalysis:
    def test_returns_figure(self, wb):
        fig = plot_pairs_analysis(wb.pairs)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_pairs_analysis(wb.pairs, to_html=True)
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_pairs_analysis(pd.DataFrame())
        assert isinstance(fig, go.Figure)


class TestPlotBacktestOverview:
    def test_returns_figure(self, wb):
        fig = plot_backtest_overview(wb.daily_return)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_backtest_overview(wb.daily_return, to_html=True)
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_backtest_overview(pd.DataFrame(columns=["date", "total"]))
        assert isinstance(fig, go.Figure)


class TestPlotColoredTable:
    def test_returns_figure(self, wb):
        fig = plot_colored_table(wb.stats)
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_colored_table(wb.stats, to_html=True)
        assert isinstance(html, str)

    def test_empty_dict(self):
        fig = plot_colored_table({})
        assert isinstance(fig, go.Figure)


class TestPlotLongShortComparison:
    def test_returns_figure(self, wb):
        fig = plot_long_short_comparison(
            wb.daily_return,
            wb.stats,
            wb.long_stats,
            wb.short_stats,
        )
        assert fig is not None
        assert hasattr(fig, "data")
        assert isinstance(fig, go.Figure)

    def test_to_html(self, wb):
        html = plot_long_short_comparison(
            wb.daily_return,
            wb.stats,
            wb.long_stats,
            wb.short_stats,
            to_html=True,
        )
        assert isinstance(html, str)

    def test_empty_df(self):
        import pandas as pd

        fig = plot_long_short_comparison(pd.DataFrame(), {}, {}, {})
        assert isinstance(fig, go.Figure)
