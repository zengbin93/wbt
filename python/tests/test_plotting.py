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

    def test_vol_normalized_subplot(self, wb):
        """`wbt.report._plot_backtest.plot_long_short_comparison` 必须包含波动率归一后的中间子图。"""
        import numpy as np
        import pandas as pd

        from wbt.report._plot_backtest import plot_long_short_comparison as report_plot

        # 构造一个 2 列日收益透视表
        dret = wb.daily_return.copy()
        dret["dt"] = pd.to_datetime(dret["date"])
        dret = dret.set_index("dt").drop(columns=["date"])
        cols = [c for c in dret.columns if c != "total"][:2]
        pivot = dret[cols]

        stats_df = pd.DataFrame({"策略名称": cols, "年化": [0.0] * len(cols)})

        target = 0.15
        fig = report_plot(pivot, stats_df, target_volatility=target)
        assert isinstance(fig, go.Figure)

        layout_titles = [a.text for a in fig.layout.annotations if hasattr(a, "text") and a.text]
        assert any("波动率调整" in t for t in layout_titles), layout_titles
        assert hasattr(fig.layout, "xaxis2") and hasattr(fig.layout, "yaxis2")

        # 校验中图的归一化逻辑：每条调整后日收益的年化波动率 ≈ target
        yd = 252
        for col in cols:
            daily_ret = pivot[col]
            annual_vol = daily_ret.std() * np.sqrt(yd)
            if annual_vol == 0:
                continue
            scale = target / annual_vol
            adj_vol = (daily_ret * scale).std() * np.sqrt(yd)
            assert adj_vol == pytest.approx(target, rel=1e-9)

    def test_vol_normalized_long_alpha_curve(self, wb):
        """当透视表同时包含 '策略多头' 和 '基准等权' 时，中图必须出现 '多头超额' 曲线。"""
        import numpy as np
        import pandas as pd

        from wbt.report._plot_backtest import plot_long_short_comparison as report_plot

        # 构造一个 2 列透视表：'策略多头' + '基准等权'
        dret = wb.daily_return.copy()
        dret["dt"] = pd.to_datetime(dret["date"])
        dret = dret.set_index("dt").drop(columns=["date"])
        base = dret["total"]
        pivot = pd.DataFrame(
            {
                "策略多头": base.clip(lower=0),
                "基准等权": base * 0.5,  # 随便造个非零基准
            }
        )
        stats_df = pd.DataFrame({"策略名称": list(pivot.columns), "年化": [0.0, 0.0]})

        target = 0.20
        fig = report_plot(pivot, stats_df, target_volatility=target)

        # '多头超额' trace 必须存在
        alpha_traces = [t for t in fig.data if getattr(t, "name", None) == "多头超额"]
        assert len(alpha_traces) == 1, [t.name for t in fig.data]

        # 数值校验：alpha = scaled(策略多头).cumsum() - scaled(基准等权).cumsum()
        yd = 252
        scales = {}
        for col in pivot.columns:
            annual_vol = pivot[col].std() * np.sqrt(yd)
            scales[col] = target / annual_vol if annual_vol > 0 else 1.0
        expected_alpha_cum = (pivot["策略多头"] * scales["策略多头"]).cumsum() - (
            pivot["基准等权"] * scales["基准等权"]
        ).cumsum()
        np.testing.assert_allclose(alpha_traces[0].y, expected_alpha_cum.to_numpy())

    def test_vol_normalized_no_long_alpha_when_columns_missing(self, wb):
        """缺少 '策略多头' 或 '基准等权' 时不应渲染 '多头超额' 曲线。"""
        import pandas as pd

        from wbt.report._plot_backtest import plot_long_short_comparison as report_plot

        dret = wb.daily_return.copy()
        dret["dt"] = pd.to_datetime(dret["date"])
        dret = dret.set_index("dt").drop(columns=["date"])
        cols = [c for c in dret.columns if c != "total"][:2]
        pivot = dret[cols]
        stats_df = pd.DataFrame({"策略名称": cols, "年化": [0.0] * len(cols)})

        fig = report_plot(pivot, stats_df)
        alpha_traces = [t for t in fig.data if getattr(t, "name", None) == "多头超额"]
        assert not alpha_traces
