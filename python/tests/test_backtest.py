import numpy as np
import pandas as pd
import pytest
from wbt import WeightBacktest


@pytest.fixture
def sample_dfw():
    """2 symbols x 15 bars inline DataFrame"""
    rng = np.random.default_rng(42)
    n_days = 15
    # Use multi-day data with 4 bars per day per symbol to get per-symbol columns in daily_return
    rows = []
    for sym in ["SYM_A", "SYM_B"]:
        for d in range(n_days):
            for h in range(4):
                dt = f"2024-01-{d + 1:02d} {9 + h:02d}:30:00"
                w = round(rng.uniform(-0.5, 0.5), 2)
                p = 100.0 + rng.normal(0, 2)
                rows.append({"dt": dt, "symbol": sym, "weight": w, "price": round(p, 4)})
    return pd.DataFrame(rows)


@pytest.fixture
def wb(sample_dfw):
    return WeightBacktest(
        sample_dfw, digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252
    )


STATS_KEYS_29 = [
    "开始日期", "结束日期", "绝对收益", "年化", "夏普", "最大回撤", "卡玛",
    "日胜率", "日盈亏比", "日赢面", "年化波动率", "下行波动率", "非零覆盖",
    "盈亏平衡点", "新高间隔", "新高占比", "回撤风险", "回归年度回报率",
    "长度调整平均最大回撤", "交易胜率", "单笔收益", "持仓K线数",
    "多头占比", "空头占比", "与基准相关性", "与基准空头相关性",
    "波动比", "与基准波动相关性", "品种数量",
]

PERF_KEYS_17 = [
    "绝对收益", "年化", "夏普", "最大回撤", "卡玛", "日胜率", "日盈亏比",
    "日赢面", "年化波动率", "下行波动率", "非零覆盖", "盈亏平衡点",
    "新高间隔", "新高占比", "回撤风险", "回归年度回报率", "长度调整平均最大回撤",
]


class TestWeightBacktestInit:
    def test_creates_successfully(self, wb):
        assert wb.digits == 2
        assert wb.fee_rate == 0.0002
        assert wb.weight_type == "ts"
        assert set(wb.symbols) == {"SYM_A", "SYM_B"}


class TestStats:
    def test_stats_keys(self, wb):
        stats = wb.stats
        assert isinstance(stats, dict)
        assert len(stats) == 29
        for key in STATS_KEYS_29:
            assert key in stats, f"missing key: {key}"

    def test_stats_date_format(self, wb):
        stats = wb.stats
        assert isinstance(stats["开始日期"], str)
        assert len(stats["开始日期"]) == 10


class TestSymbolDict:
    def test_symbol_dict(self, wb):
        sd = wb.symbol_dict
        assert isinstance(sd, list)
        assert len(sd) == 2


class TestDailyReturn:
    def test_structure(self, wb):
        dr = wb.daily_return
        assert isinstance(dr, pd.DataFrame)
        assert "date" in dr.columns
        assert "total" in dr.columns
        assert len(dr) > 0


class TestDailys:
    def test_columns(self, wb):
        df = wb.dailys
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "symbol", "date", "n1b", "edge", "return", "cost", "turnover",
            "long_edge", "short_edge", "long_cost", "short_cost",
            "long_turnover", "short_turnover", "long_return", "short_return",
        ]
        for col in expected_cols:
            assert col in df.columns, f"missing: {col}"


class TestAlpha:
    def test_structure(self, wb):
        df = wb.alpha
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "超额", "策略", "基准"]


class TestPairs:
    def test_structure(self, wb):
        df = wb.pairs
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "symbol" in df.columns
            assert "交易方向" in df.columns


class TestAlphaAndBenchStats:
    def test_alpha_stats(self, wb):
        stats = wb.alpha_stats
        assert isinstance(stats, dict)
        assert "开始日期" in stats
        assert "结束日期" in stats
        for key in PERF_KEYS_17:
            assert key in stats

    def test_bench_stats(self, wb):
        stats = wb.bench_stats
        assert isinstance(stats, dict)
        for key in PERF_KEYS_17:
            assert key in stats


class TestLongShortReturns:
    def test_long_daily_return(self, wb):
        df = wb.long_daily_return
        assert isinstance(df, pd.DataFrame)
        assert "total" in df.columns

    def test_short_daily_return(self, wb):
        df = wb.short_daily_return
        assert isinstance(df, pd.DataFrame)
        assert "total" in df.columns

    def test_long_stats(self, wb):
        stats = wb.long_stats
        assert isinstance(stats, dict)

    def test_short_stats(self, wb):
        stats = wb.short_stats
        assert isinstance(stats, dict)


class TestSymbolMethods:
    def test_get_top_symbols_profit(self, wb):
        # daily_return from Rust only has date+total, so get_top_symbols
        # returns empty unless per-symbol columns are present
        result = wb.get_top_symbols(n=1, kind="profit")
        assert isinstance(result, list)

    def test_get_top_symbols_loss(self, wb):
        result = wb.get_top_symbols(n=1, kind="loss")
        assert isinstance(result, list)

    def test_get_top_symbols_n_exceeds(self, wb):
        result = wb.get_top_symbols(n=10, kind="profit")
        assert isinstance(result, list)

    def test_get_symbol_daily(self, wb):
        df = wb.get_symbol_daily("SYM_A")
        assert isinstance(df, pd.DataFrame)
        assert all(df["symbol"] == "SYM_A")

    def test_get_symbol_pairs(self, wb):
        df = wb.get_symbol_pairs("SYM_A")
        assert isinstance(df, pd.DataFrame)
