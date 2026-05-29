from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

from wbt import WeightBacktest  # noqa: E402
from wbt.plotting import (  # noqa: E402
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_drawdowns_table,
    plot_key_trades,
    plot_monthly_heatmap,
    plot_pairs_hold_dist,
    plot_pairs_pnl_dist,
    plot_stats_comparison,
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

    def test_voladj(self, result):
        """voladj=True 时消费 curves_voladj，仍能出曲线。"""
        fig = plot_cumulative_returns(result, keys=["多空", "超额"], voladj=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

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


class TestPlotPairsPnlDist:
    def test_returns_figure(self, result):
        fig = plot_pairs_pnl_dist(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_pairs_pnl_dist(result, to_html=True), str)

    def test_pnl_axis_is_percent_number_not_fraction(self, result):
        """pnl_pct 是百分数（1.0==1%），x 轴不能用 ".1%"（会再 ×100，放大 100 倍）。"""
        fig = plot_pairs_pnl_dist(result)
        assert fig.layout.xaxis.tickformat in (None, "")
        assert "%" in (fig.layout.xaxis.title.text or "")


class TestPlotPairsHoldDist:
    def test_returns_figure(self, result):
        fig = plot_pairs_hold_dist(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_pairs_hold_dist(result, to_html=True), str)


class TestPlotColoredTable:
    def test_returns_figure(self, result):
        fig = plot_colored_table(result)
        assert isinstance(fig, go.Figure)

    def test_to_html(self, result):
        assert isinstance(plot_colored_table(result, to_html=True), str)


class TestPlotStatsComparison:
    def test_returns_figure(self, result):
        fig = plot_stats_comparison(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_to_html(self, result):
        assert isinstance(plot_stats_comparison(result, to_html=True), str)


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
