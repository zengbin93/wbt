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
from wbt.plotting._common import fmt_value  # noqa: E402
from wbt.result import BacktestResult  # noqa: E402


class TestFmtValue:
    def test_ratio_to_percent(self):
        assert fmt_value("年化收益", 0.0058) == "0.58%"
        assert fmt_value("最大回撤", 0.3641) == "36.41%"
        assert fmt_value("日胜率", 0.4988) == "49.88%"

    def test_counts_are_integer_with_separator(self):
        assert fmt_value("交易次数", 681602.0) == "681,602"
        assert fmt_value("恢复天数", 2578.0) == "2,578"

    def test_year_has_no_separator(self):
        assert fmt_value("year", 2020) == "2020"

    def test_ratio_metrics_two_decimals(self):
        assert fmt_value("夏普比率", 0.0629) == "0.06"
        assert fmt_value("单笔盈亏比", 0.9937) == "0.99"

    def test_bp_two_decimals(self):
        assert fmt_value("单笔收益", 4.66) == "4.66"

    def test_missing_values(self):
        assert fmt_value("恢复天数", None) == "—"
        assert fmt_value("新高间隔", float("nan")) == "—"

    def test_bool_to_cn(self):
        assert fmt_value("完整年", True) == "是"
        assert fmt_value("达标", False) == "否"


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

    def test_win_rate_annotation(self, result):
        """月度/年度胜率应作为标注呈现（数据已在 result.monthly 算好）。"""
        fig = plot_monthly_heatmap(result)
        texts = " ".join(a.text or "" for a in fig.layout.annotations)
        assert "月度胜率" in texts and "年度胜率" in texts


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

    def test_benchmark_excess_not_na(self, result):
        """基准/超额 stats 键名不同（年化/夏普/卡玛），别名映射后不应出现「—」。"""
        fig = plot_stats_comparison(result)
        header = list(fig.data[0].header.values)  # ["指标","多空",...,"基准","超额"]
        cols = fig.data[0].cells.values
        for side in ("基准", "超额"):
            ci = header.index(side)
            assert "—" not in cols[ci], f"{side} 列出现缺失值: {cols[ci]}"


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

    def test_yearly_table_chinese_year_first(self, result):
        """年度指标表应为中文列名且「年份」在首列。"""
        fig = plot_verdict(result)
        # 若存在表格 trace，校验表头
        tables = [t for t in fig.data if t.type == "table"]
        if tables:
            header = list(tables[0].header.values)
            assert header[0] == "年份"
            assert "abs_return" not in header and "绝对收益" in header
