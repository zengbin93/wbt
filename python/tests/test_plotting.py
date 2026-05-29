from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

from wbt import WeightBacktest  # noqa: E402
from wbt.plotting import (  # noqa: E402
    plot_backtest_overview,
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_drawdowns_table,
    plot_key_trades,
    plot_long_short_comparison,
    plot_monthly_heatmap,
    plot_pairs_analysis,
    plot_symbol_returns,
    plot_verdict,
)
from wbt.result import BacktestResult  # noqa: E402

# ---------------------------------------------------------------------------
# `wb` fixture 来自 conftest.py；绘图统一以 BacktestResult 为入参
# ---------------------------------------------------------------------------


@pytest.fixture
def result(wb: WeightBacktest) -> BacktestResult:
    return wb.to_result()


class TestPlotCumulativeReturns:
    def test_returns_figure(self, result):
        fig = plot_cumulative_returns(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_multiple_keys(self, result):
        fig = plot_cumulative_returns(result, keys=["多空", "多头", "空头"])
        assert len(fig.data) == 3

    def test_to_html(self, result):
        html = plot_cumulative_returns(result, to_html=True)
        assert isinstance(html, str)
        assert "<div" in html


class TestPlotMonthlyHeatmap:
    def test_returns_figure(self, result):
        fig = plot_monthly_heatmap(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_monthly_heatmap(result, to_html=True), str)


class TestPlotSymbolReturns:
    def test_returns_figure(self, result):
        fig = plot_symbol_returns(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_symbol_returns(result, to_html=True), str)


class TestPlotDrawdown:
    def test_two_traces(self, result):
        fig = plot_drawdown(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 回撤 + 累计收益

    def test_to_html(self, result):
        assert isinstance(plot_drawdown(result, to_html=True), str)


class TestPlotDailyReturnDist:
    def test_returns_figure(self, result):
        fig = plot_daily_return_dist(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_daily_return_dist(result, to_html=True), str)


class TestPlotPairsAnalysis:
    def test_returns_figure(self, result):
        fig = plot_pairs_analysis(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_pairs_analysis(result, to_html=True), str)


class TestPlotBacktestOverview:
    def test_returns_figure(self, result):
        fig = plot_backtest_overview(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_backtest_overview(result, to_html=True), str)


class TestPlotColoredTable:
    def test_returns_figure(self, result):
        fig = plot_colored_table(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_colored_table(result, to_html=True), str)


class TestPlotLongShortComparison:
    def test_returns_figure(self, result):
        fig = plot_long_short_comparison(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_to_html(self, result):
        assert isinstance(plot_long_short_comparison(result, to_html=True), str)

    def test_has_voladj_curves(self, result):
        """多空对比应包含原始与波动率归一两组曲线（区分）。"""
        fig = plot_long_short_comparison(result)
        names = [getattr(t, "name", "") or "" for t in fig.data]
        assert any("归一" in n for n in names), names


class TestPlotKeyTrades:
    def test_returns_figure(self, result):
        fig = plot_key_trades(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_key_trades(result, to_html=True), str)


class TestPlotDrawdownsTable:
    def test_returns_figure(self, result):
        fig = plot_drawdowns_table(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_drawdowns_table(result, to_html=True), str)


class TestPlotVerdict:
    def test_returns_figure(self, result):
        fig = plot_verdict(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_verdict(result, to_html=True), str)
