from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest

STATS_KEYS_25 = [
    "开始日期",
    "结束日期",
    "绝对收益",
    "年化收益",
    "夏普比率",
    "卡玛比率",
    "新高占比",
    "单笔盈亏比",
    "单笔收益",
    "日胜率",
    "周胜率",
    "月胜率",
    "季胜率",
    "年胜率",
    "最大回撤",
    "年化波动率",
    "下行波动率",
    "新高间隔",
    "交易次数",
    "年化交易次数",
    "持仓K线数",
    "交易胜率",
    "多头占比",
    "空头占比",
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
        assert len(stats) == 25
        for key in STATS_KEYS_25:
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
        # Verify new key names exist
        assert "年化收益" in stats
        assert "夏普比率" in stats
        assert "卡玛比率" in stats


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


def _round_half_away_from_zero(values: pd.Series, digits: int) -> pd.Series:
    scale = 10**digits
    return np.sign(values) * np.floor(np.abs(values) * scale + 0.5) / scale


def _bar_reference(data: pd.DataFrame, digits: int, fee_rate: float) -> pd.DataFrame:
    ref = data.sort_values(["symbol", "dt"], kind="stable").copy()
    ref["dt"] = pd.to_datetime(ref["dt"])
    ref["weight"] = _round_half_away_from_zero(ref["weight"], digits)
    grouped = ref.groupby("symbol", sort=False)
    ref["prev_weight"] = grouped["weight"].shift()
    ref["prev_price"] = grouped["price"].shift()
    ref = ref.loc[ref["prev_weight"].notna()].copy()
    ref["n1b"] = np.where(ref["prev_price"].eq(0), 0.0, ref["price"] / ref["prev_price"] - 1.0)
    ref["edge"] = ref["prev_weight"] * ref["n1b"]
    ref["turnover"] = (ref["weight"] - ref["prev_weight"]).abs()
    ref["cost"] = ref["turnover"] * fee_rate
    ref["return"] = ref["edge"] - ref["cost"]
    ref["long_weight"] = ref["weight"].clip(lower=0)
    ref["short_weight"] = ref["weight"].clip(upper=0)
    ref["prev_long_weight"] = ref["prev_weight"].clip(lower=0)
    ref["prev_short_weight"] = ref["prev_weight"].clip(upper=0)
    ref["long_edge"] = ref["prev_long_weight"] * ref["n1b"]
    ref["short_edge"] = ref["prev_short_weight"] * ref["n1b"]
    ref["long_turnover"] = (ref["long_weight"] - ref["prev_long_weight"]).abs()
    ref["short_turnover"] = (ref["short_weight"] - ref["prev_short_weight"]).abs()
    ref["long_cost"] = ref["long_turnover"] * fee_rate
    ref["short_cost"] = ref["short_turnover"] * fee_rate
    ref["long_return"] = ref["long_edge"] - ref["long_cost"]
    ref["short_return"] = ref["short_edge"] - ref["short_cost"]
    ref["date"] = ref["dt"].dt.normalize()
    columns = [
        "n1b", "edge", "return", "cost", "turnover", "long_edge", "short_edge",
        "long_cost", "short_cost", "long_turnover", "short_turnover", "long_return", "short_return",
    ]
    return ref.groupby(["symbol", "date"], as_index=False, sort=True)[columns].sum()


@pytest.mark.parametrize("weight_type, aggregate", [("ts", "mean"), ("cs", "sum")])
def test_bar_returns_match_sorted_half_away_reference(weight_type: str, aggregate: str) -> None:
    """Current-bar returns use Rust's normalized weights after stable symbol/time ordering."""
    digits = 2
    fee_rate = 0.001
    data = pd.DataFrame(
        [
            {"dt": "2024-01-03 09:00:00", "symbol": "B", "weight": -0.245, "price": 202.0},
            {"dt": "2024-01-02 10:00:00", "symbol": "A", "weight": -0.125, "price": 101.0},
            {"dt": "2024-01-02 09:00:00", "symbol": "B", "weight": 0.245, "price": 200.0},
            {"dt": "2024-01-03 09:00:00", "symbol": "A", "weight": 0.0, "price": 99.0},
            {"dt": "2024-01-02 09:00:00", "symbol": "A", "weight": 0.125, "price": 100.0},
            {"dt": "2024-01-02 10:00:00", "symbol": "B", "weight": 0.245, "price": 204.0},
        ]
    )
    expected = _bar_reference(data, digits, fee_rate)
    backtest = WeightBacktest(data, digits=digits, fee_rate=fee_rate, n_jobs=1, weight_type=weight_type)
    actual = backtest.dailys.copy()
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    expected = expected.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)

    assert list(actual[["symbol", "date"]].itertuples(index=False, name=None)) == list(
        expected[["symbol", "date"]].itertuples(index=False, name=None)
    )
    for column in expected.columns[2:]:
        np.testing.assert_allclose(actual[column], expected[column], atol=1e-12, err_msg=column)

    expected_total = expected.groupby("date", sort=True)["return"].agg(aggregate)
    actual_total = backtest.daily_return.set_index("date")["total"]
    actual_total.index = pd.to_datetime(actual_total.index)
    assert list(actual_total.index) == list(expected_total.index)
    np.testing.assert_allclose(actual_total, expected_total, atol=1e-12)


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
        stats = wb.long_stats
        assert isinstance(stats, dict)
        assert "年化收益" in stats
        assert "夏普比率" in stats
        assert "交易次数" in stats

    def test_short_stats(self, wb: WeightBacktest) -> None:
        stats = wb.short_stats
        assert isinstance(stats, dict)
        assert "年化收益" in stats
        assert "夏普比率" in stats


class TestSegmentStats:
    """验证分段统计功能。"""

    def test_segment_stats_default(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats()
        assert isinstance(stats, dict)
        assert "年化收益" in stats
        assert "交易次数" in stats

    def test_segment_stats_long(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats(kind="多头")
        assert isinstance(stats, dict)
        assert "年化收益" in stats

    def test_segment_stats_short(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats(kind="空头")
        assert isinstance(stats, dict)
        assert "年化收益" in stats


class TestLongAlphaStats:
    """验证波动率调整后的多头超额收益统计。"""

    def test_long_alpha_stats(self, wb: WeightBacktest) -> None:
        stats = wb.long_alpha_stats
        assert isinstance(stats, dict)
        assert "年化收益" in stats
        assert "夏普比率" in stats


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
