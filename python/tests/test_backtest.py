from __future__ import annotations

import pandas as pd
import pytest

from wbt import WeightBacktest

STATS_KEYS_29 = [
    "开始日期",
    "结束日期",
    "绝对收益",
    "年化",
    "夏普",
    "最大回撤",
    "卡玛",
    "日胜率",
    "日盈亏比",
    "日赢面",
    "年化波动率",
    "下行波动率",
    "非零覆盖",
    "盈亏平衡点",
    "新高间隔",
    "新高占比",
    "回撤风险",
    "回归年度回报率",
    "长度调整平均最大回撤",
    "交易胜率",
    "单笔收益",
    "持仓K线数",
    "多头占比",
    "空头占比",
    "与基准相关性",
    "与基准空头相关性",
    "波动比",
    "与基准波动相关性",
    "品种数量",
]

PERF_KEYS_17 = [
    "绝对收益",
    "年化",
    "夏普",
    "最大回撤",
    "卡玛",
    "日胜率",
    "日盈亏比",
    "日赢面",
    "年化波动率",
    "下行波动率",
    "非零覆盖",
    "盈亏平衡点",
    "新高间隔",
    "新高占比",
    "回撤风险",
    "回归年度回报率",
    "长度调整平均最大回撤",
]


class TestWeightBacktestInit:
    """验证 WeightBacktest 初始化参数正确保存。"""

    def test_creates_successfully(self, wb: WeightBacktest) -> None:
        assert wb.digits == 2
        assert wb.fee_rate == pytest.approx(0.0002)
        assert wb.weight_type == "ts"
        assert set(wb.symbols) == {"SYM_A", "SYM_B"}


class TestStats:
    """验证 stats 字典结构完整性和值合理性。"""

    def test_stats_keys(self, wb: WeightBacktest) -> None:
        stats = wb.stats
        assert isinstance(stats, dict)
        assert len(stats) == 29
        for key in STATS_KEYS_29:
            assert key in stats, f"missing key: {key}"

    def test_stats_date_format(self, wb: WeightBacktest) -> None:
        stats = wb.stats
        assert isinstance(stats["开始日期"], str)
        assert len(stats["开始日期"]) == 10

    def test_stats_values_consistency(self, wb: WeightBacktest) -> None:
        """验证 stats 内部数值的范围一致性。"""
        stats = wb.stats
        assert stats["品种数量"] == 2
        assert 0 <= stats["多头占比"] <= 1.0
        assert 0 <= stats["空头占比"] <= 1.0
        assert 0 <= stats["日胜率"] <= 1.0
        assert 0 <= stats["交易胜率"] <= 1.0
        assert stats["最大回撤"] >= 0
        assert stats["年化波动率"] >= 0


class TestSymbolDict:
    """验证符号字典提取。"""

    def test_symbol_dict(self, wb: WeightBacktest) -> None:
        sd = wb.symbol_dict
        assert isinstance(sd, list)
        assert len(sd) == 2


class TestDailyReturn:
    """验证 daily_return DataFrame 结构。"""

    def test_structure(self, wb: WeightBacktest) -> None:
        dr = wb.daily_return
        assert isinstance(dr, pd.DataFrame)
        assert "date" in dr.columns
        assert "total" in dr.columns
        assert len(dr) > 0


class TestDailys:
    """验证品种每日交易信息的列完整性和数值一致性。"""

    def test_columns(self, wb: WeightBacktest) -> None:
        df = wb.dailys
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "symbol",
            "date",
            "n1b",
            "edge",
            "return",
            "cost",
            "turnover",
            "long_edge",
            "short_edge",
            "long_cost",
            "short_cost",
            "long_turnover",
            "short_turnover",
            "long_return",
            "short_return",
        ]
        for col in expected_cols:
            assert col in df.columns, f"missing: {col}"

    def test_return_equals_edge_minus_cost(self, wb: WeightBacktest) -> None:
        """return 应等于 edge - cost。"""
        df = wb.dailys
        expected = df["edge"] - df["cost"]
        pd.testing.assert_series_equal(df["return"], expected, check_names=False, atol=1e-8)

    def test_long_short_edge_consistency(self, wb: WeightBacktest) -> None:
        """long_edge + short_edge 应等于 edge。"""
        df = wb.dailys
        expected = df["long_edge"] + df["short_edge"]
        pd.testing.assert_series_equal(df["edge"], expected, check_names=False, atol=1e-8)


class TestAlpha:
    """验证超额收益 DataFrame 的结构和数值。"""

    def test_structure(self, wb: WeightBacktest) -> None:
        df = wb.alpha
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "超额", "策略", "基准"]

    def test_alpha_equals_strategy_minus_benchmark(self, wb: WeightBacktest) -> None:
        """超额 应等于 策略 - 基准。"""
        df = wb.alpha
        expected = df["策略"] - df["基准"]
        pd.testing.assert_series_equal(df["超额"], expected, check_names=False, atol=1e-10)


class TestPairs:
    """验证交易对数据结构。"""

    def test_structure(self, wb: WeightBacktest) -> None:
        df = wb.pairs
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "symbol" in df.columns
            assert "交易方向" in df.columns


class TestAlphaAndBenchStats:
    """验证超额和基准的统计指标完整性。"""

    def test_alpha_stats(self, wb: WeightBacktest) -> None:
        stats = wb.alpha_stats
        assert isinstance(stats, dict)
        assert "开始日期" in stats
        assert "结束日期" in stats
        for key in PERF_KEYS_17:
            assert key in stats

    def test_bench_stats(self, wb: WeightBacktest) -> None:
        stats = wb.bench_stats
        assert isinstance(stats, dict)
        for key in PERF_KEYS_17:
            assert key in stats


class TestLongShortReturns:
    """验证多空分离收益的结构。"""

    def test_long_daily_return(self, wb: WeightBacktest) -> None:
        df = wb.long_daily_return
        assert isinstance(df, pd.DataFrame)
        assert "total" in df.columns

    def test_short_daily_return(self, wb: WeightBacktest) -> None:
        df = wb.short_daily_return
        assert isinstance(df, pd.DataFrame)
        assert "total" in df.columns

    def test_long_stats(self, wb: WeightBacktest) -> None:
        assert isinstance(wb.long_stats, dict)

    def test_short_stats(self, wb: WeightBacktest) -> None:
        assert isinstance(wb.short_stats, dict)


class TestSymbolMethods:
    """验证按品种查询相关方法。"""

    def test_get_top_symbols_profit(self, wb: WeightBacktest) -> None:
        result = wb.get_top_symbols(n=1, kind="profit")
        assert isinstance(result, list)

    def test_get_top_symbols_loss(self, wb: WeightBacktest) -> None:
        result = wb.get_top_symbols(n=1, kind="loss")
        assert isinstance(result, list)

    def test_get_top_symbols_n_exceeds(self, wb: WeightBacktest) -> None:
        result = wb.get_top_symbols(n=10, kind="profit")
        assert isinstance(result, list)

    def test_get_symbol_daily(self, wb: WeightBacktest) -> None:
        df = wb.get_symbol_daily("SYM_A")
        assert isinstance(df, pd.DataFrame)
        assert all(df["symbol"] == "SYM_A")

    def test_get_symbol_pairs(self, wb: WeightBacktest) -> None:
        df = wb.get_symbol_pairs("SYM_A")
        assert isinstance(df, pd.DataFrame)
