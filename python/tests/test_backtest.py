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


@pytest.fixture
def contract_dfw():
    """Deterministic data with known winners/losers and explicit open-close transitions."""
    rows = []
    dates = [f"2024-02-0{i} 09:30:00" for i in range(1, 5)]

    sym_a_weights = [0.0, 1.0, 1.0, 0.0]
    sym_a_prices = [100.0, 110.0, 120.0, 120.0]

    sym_b_weights = [0.0, 1.0, 1.0, 0.0]
    sym_b_prices = [100.0, 95.0, 90.0, 90.0]

    for dt, weight, price in zip(dates, sym_a_weights, sym_a_prices):
        rows.append({"dt": dt, "symbol": "SYM_A", "weight": weight, "price": price})

    for dt, weight, price in zip(dates, sym_b_weights, sym_b_prices):
        rows.append({"dt": dt, "symbol": "SYM_B", "weight": weight, "price": price})

    return pd.DataFrame(rows)


@pytest.fixture
def contract_wb(contract_dfw):
    return WeightBacktest(
        contract_dfw, digits=2, fee_rate=0.0, n_jobs=1, weight_type="ts", yearly_days=252
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

    def test_stats_values_consistency(self, wb):
        """Verify internal consistency of stats values"""
        stats = wb.stats
        # 品种数量 should match actual symbols
        assert stats["品种数量"] == 2
        # 多头占比 + 空头占比 should be <= 1 (some weights may be 0)
        assert 0 <= stats["多头占比"] <= 1.0
        assert 0 <= stats["空头占比"] <= 1.0
        # 日胜率 should be between 0 and 1
        assert 0 <= stats["日胜率"] <= 1.0
        # 交易胜率 between 0 and 1
        assert 0 <= stats["交易胜率"] <= 1.0
        # 最大回撤 >= 0
        assert stats["最大回撤"] >= 0
        # 年化波动率 >= 0
        assert stats["年化波动率"] >= 0


class TestSymbolDict:
    def test_symbol_dict(self, wb):
        sd = wb.symbol_dict
        assert isinstance(sd, list)
        assert len(sd) == 2


class TestDailyReturn:
    def test_structure(self, contract_wb):
        dr = contract_wb.daily_return
        assert isinstance(dr, pd.DataFrame)
        assert list(dr.columns) == ["date", "SYM_A", "SYM_B", "total"]
        assert len(dr) == 3

    def test_total_matches_mean_of_symbol_returns(self, contract_wb):
        dr = contract_wb.daily_return
        expected_total = dr[["SYM_A", "SYM_B"]].mean(axis=1)
        pd.testing.assert_series_equal(dr["total"], expected_total, check_names=False)


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

    def test_return_equals_edge_minus_cost(self, wb):
        """return should equal edge - cost for every row"""
        df = wb.dailys
        for _, row in df.iterrows():
            expected = row["edge"] - row["cost"]
            assert abs(row["return"] - expected) < 1e-8, (
                f"return={row['return']} != edge-cost={expected}"
            )

    def test_long_short_edge_equals_edge(self, wb):
        """edge should equal long_edge + short_edge for every row"""
        df = wb.dailys
        for _, row in df.iterrows():
            total_edge = row["long_edge"] + row["short_edge"]
            assert abs(row["edge"] - total_edge) < 1e-8, (
                f"edge={row['edge']} != long_edge+short_edge={total_edge}"
            )

    def test_long_short_cost_equals_cost(self, wb):
        """cost should equal long_cost + short_cost for every row"""
        df = wb.dailys
        for _, row in df.iterrows():
            total_cost = row["long_cost"] + row["short_cost"]
            assert abs(row["cost"] - total_cost) < 1e-8, (
                f"cost={row['cost']} != long_cost+short_cost={total_cost}"
            )

    def test_long_short_return_equals_return(self, wb):
        """return should equal long_return + short_return for every row"""
        df = wb.dailys
        for _, row in df.iterrows():
            total_return = row["long_return"] + row["short_return"]
            assert abs(row["return"] - total_return) < 1e-8, (
                f"return={row['return']} != long_return+short_return={total_return}"
            )


class TestAlpha:
    def test_structure(self, wb):
        df = wb.alpha
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["date", "超额", "策略", "基准"]

    def test_alpha_equals_strategy_minus_benchmark(self, wb):
        """超额 should equal 策略 - 基准 for every row"""
        df = wb.alpha
        for _, row in df.iterrows():
            diff = row["策略"] - row["基准"]
            assert abs(row["超额"] - diff) < 1e-10, (
                f"超额={row['超额']} != 策略-基准={diff}"
            )


class TestPairs:
    def test_structure(self, contract_wb):
        df = contract_wb.pairs
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symbol" in df.columns
        assert "交易方向" in df.columns

    def test_pairs_include_both_symbols(self, contract_wb):
        df = contract_wb.pairs
        assert set(df["symbol"]) == {"SYM_A", "SYM_B"}


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
    def test_get_top_symbols_profit(self, contract_wb):
        result = contract_wb.get_top_symbols(n=1, kind="profit")
        assert result == ["SYM_A"]

    def test_get_top_symbols_loss(self, contract_wb):
        result = contract_wb.get_top_symbols(n=1, kind="loss")
        assert result == ["SYM_B"]

    def test_get_top_symbols_n_exceeds(self, contract_wb):
        result = contract_wb.get_top_symbols(n=10, kind="profit")
        assert result == ["SYM_A", "SYM_B"]

    def test_get_symbol_daily(self, wb):
        df = wb.get_symbol_daily("SYM_A")
        assert isinstance(df, pd.DataFrame)
        assert all(df["symbol"] == "SYM_A")

    def test_get_symbol_pairs(self, contract_wb):
        df = contract_wb.get_symbol_pairs("SYM_A")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(df["symbol"] == "SYM_A")
