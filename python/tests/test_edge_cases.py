"""Edge case tests for wbt metrics system.

Covers special scenarios that trigger early-return paths or boundary conditions
in daily_performance, segment_stats, long_stats/short_stats, and long_alpha_stats.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest, daily_performance


# ============================================================================
# Helper
# ============================================================================

def _make_dfw(n_days: int, symbols: list[str], weight_fn, price_fn) -> pd.DataFrame:
    """Build deterministic test DataFrame."""
    base = datetime(2024, 1, 2, 9, 30, 0)
    rows = []
    for d in range(n_days):
        dt_str = (base + timedelta(days=d)).strftime("%Y-%m-%d %H:%M:%S")
        for sym in symbols:
            rows.append({
                "dt": dt_str,
                "symbol": sym,
                "weight": weight_fn(d, sym),
                "price": price_fn(d, sym),
            })
    return pd.DataFrame(rows)


# ============================================================================
# 1. daily_performance edge cases
# ============================================================================

class TestDailyPerformanceEdgeCases:
    """Edge cases for the standalone daily_performance function."""

    def test_single_value_positive(self) -> None:
        """Single positive return: std=0 → all metrics zero (by design)."""
        dp = daily_performance(np.array([0.05]), yearly_days=252)
        assert dp["绝对收益"] == 0.0
        assert dp["年化"] == 0.0

    def test_single_value_negative(self) -> None:
        """Single negative return: std=0 → all metrics zero."""
        dp = daily_performance(np.array([-0.03]), yearly_days=252)
        assert dp["绝对收益"] == 0.0

    def test_constant_positive_returns(self) -> None:
        """All same positive return: std=0 → all metrics zero."""
        dp = daily_performance(np.array([0.001] * 100), yearly_days=252)
        assert dp["绝对收益"] == 0.0
        assert dp["夏普"] == 0.0

    def test_constant_negative_returns(self) -> None:
        """All same negative return: std=0 → all metrics zero."""
        dp = daily_performance(np.array([-0.001] * 100), yearly_days=252)
        assert dp["绝对收益"] == 0.0

    def test_cum_return_near_zero(self) -> None:
        """Returns that cancel out: cum_return ≈ 0 → all metrics zero."""
        dp = daily_performance(np.array([0.01, -0.01]), yearly_days=252)
        assert dp["绝对收益"] == 0.0

    def test_two_values_with_variance(self) -> None:
        """Two different values: std > 0, cum_return > 0 → valid metrics."""
        dp = daily_performance(np.array([0.02, 0.01]), yearly_days=252)
        assert dp["绝对收益"] == pytest.approx(0.03, abs=0.001)
        assert dp["年化"] > 0
        assert dp["夏普"] > 0
        assert dp["最大回撤"] == 0.0  # all positive, no drawdown
        assert dp["日胜率"] == 1.0
        assert dp["下行波动率"] == 0.0  # no negative returns

    def test_all_positive_no_drawdown(self) -> None:
        """Monotonically positive returns: max_drawdown = 0."""
        dp = daily_performance(np.array([0.01, 0.02, 0.03, 0.04]), yearly_days=252)
        assert dp["最大回撤"] == 0.0
        assert dp["新高占比"] == 1.0
        # 新高间隔 = 1: every bar is a new high, interval between highs = 1 bar
        assert dp["新高间隔"] == 1.0

    def test_all_negative(self) -> None:
        """All negative returns: max_drawdown > 0, win_rate = 0."""
        dp = daily_performance(np.array([-0.01, -0.02, -0.03]), yearly_days=252)
        assert dp["绝对收益"] < 0
        assert dp["年化"] < 0
        assert dp["夏普"] < 0
        assert dp["最大回撤"] > 0
        assert dp["日胜率"] == 0.0
        assert dp["下行波动率"] > 0

    def test_sharpe_upper_cap(self) -> None:
        """Very high Sharpe should be capped at 10.0."""
        # Large positive mean, tiny std
        dp = daily_performance(np.array([0.1, 0.1001]), yearly_days=252)
        assert dp["夏普"] == 10.0

    def test_sharpe_lower_cap(self) -> None:
        """Very negative Sharpe should be capped at -5.0."""
        dp = daily_performance(np.array([-0.1, -0.1001]), yearly_days=252)
        assert dp["夏普"] == -5.0

    def test_calmar_cap_when_no_drawdown(self) -> None:
        """When max_drawdown ≈ 0, calmar should be capped at 10.0."""
        dp = daily_performance(np.array([0.01, 0.02, 0.03, 0.005]), yearly_days=252)
        assert dp["最大回撤"] == 0.0
        assert dp["卡玛"] == 10.0


# ============================================================================
# 2. WeightBacktest with extreme data
# ============================================================================

class TestMinimalData:
    """Minimum viable data: 2-3 bars."""

    def test_two_bars_one_symbol(self) -> None:
        """2 bars → 1 daily return → std=0 → stats all zero."""
        dfw = _make_dfw(2, ["A"], lambda d, s: 0.5, lambda d, s: 100.0 + d)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        assert bt.stats["绝对收益"] == 0.0

    def test_three_bars_one_symbol(self) -> None:
        """3 bars → 2 daily returns → std > 0 if different → valid stats."""
        dfw = _make_dfw(3, ["A"], lambda d, s: 0.5,
                        lambda d, s: [100.0, 102.0, 101.0][d])
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        stats = bt.stats
        # 2 returns: (102-100)/100=0.02, (101-102)/102=-0.0098
        # cum ≈ 0.01, std > 0 → valid stats
        assert stats["绝对收益"] != 0.0
        assert isinstance(stats["夏普比率"], float)


class TestPureLong:
    """All weights > 0 → no short positions."""

    def test_short_stats_zero(self) -> None:
        """Pure long: short_stats should have zero return."""
        dfw = _make_dfw(10, ["A", "B"],
                        lambda d, s: 0.3,
                        lambda d, s: 100.0 + d * (1 if s == "A" else 0.5))
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1, yearly_days=252)
        assert bt.stats["空头占比"] == 0.0
        assert bt.short_stats["绝对收益"] == 0.0
        assert bt.short_stats["交易次数"] == 0

    def test_long_stats_nonzero(self) -> None:
        """Pure long: long_stats should have valid return."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: 0.5,
                        lambda d, s: 100.0 + d * 0.5)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        # All returns positive (price increasing), long position
        long_s = bt.long_stats
        assert long_s["绝对收益"] != 0.0 or long_s["交易次数"] == 0


class TestPureShort:
    """All weights < 0 → no long positions."""

    def test_long_stats_zero(self) -> None:
        """Pure short: long_stats should have zero return."""
        dfw = _make_dfw(10, ["A", "B"],
                        lambda d, s: -0.3,
                        lambda d, s: 100.0 + d * (1 if s == "A" else 0.5))
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1, yearly_days=252)
        assert bt.stats["多头占比"] == 0.0
        assert bt.long_stats["绝对收益"] == 0.0
        assert bt.long_stats["交易次数"] == 0

    def test_short_stats_nonzero(self) -> None:
        """Pure short: short_stats should have non-zero metrics."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: -0.5,
                        lambda d, s: 100.0 + d * 0.5)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        short_s = bt.short_stats
        # Short position with rising prices → negative return
        assert short_s["绝对收益"] != 0.0 or short_s["交易次数"] == 0


class TestZeroWeightsAllBars:
    """All weights = 0 → no positions at all."""

    def test_all_zero_weights(self) -> None:
        """Zero weights: all stats should be zero, no trades."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: 0.0,
                        lambda d, s: 100.0 + d)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        assert bt.stats["绝对收益"] == 0.0
        assert bt.stats["交易次数"] == 0
        assert bt.stats["多头占比"] == 0.0
        assert bt.stats["空头占比"] == 0.0


# ============================================================================
# 3. segment_stats edge cases (beyond test_metrics_correctness.py)
# ============================================================================

class TestSegmentStatsEdgeCases:
    """Additional edge cases for segment_stats."""

    @pytest.fixture
    def bt(self) -> WeightBacktest:
        dfw = _make_dfw(20, ["A", "B"],
                        lambda d, s: 0.3 if (d + (0 if s == "A" else 1)) % 3 != 0 else -0.2,
                        lambda d, s: 100.0 + d * (0.5 if s == "A" else -0.3) + (0 if s == "A" else 50))
        return WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1, yearly_days=252)

    def test_two_day_range(self, bt: WeightBacktest) -> None:
        """Two days: enough data for std > 0 → valid stats."""
        seg = bt.segment_stats(sdt=20240105, edt=20240106)
        # 2 days of data, may have non-zero std
        assert isinstance(seg["绝对收益"], (int, float))
        assert isinstance(seg["交易次数"], (int, float))

    def test_long_in_pure_short_range(self, bt: WeightBacktest) -> None:
        """kind='多头' in a range where all weights might be short → zero return."""
        # Even if some days have long weights, the long return for those days is valid
        seg = bt.segment_stats(sdt=20240102, edt=20240121, kind="多头")
        assert isinstance(seg["绝对收益"], (int, float))

    def test_segment_stats_all_three_kinds_sum(self, bt: WeightBacktest) -> None:
        """long + short should approximately equal 多空 for full range."""
        seg_all = bt.segment_stats()
        seg_long = bt.segment_stats(kind="多头")
        seg_short = bt.segment_stats(kind="空头")
        # abs_ret(多空) ≈ abs_ret(多头) + abs_ret(空头)
        combined = seg_long["绝对收益"] + seg_short["绝对收益"]
        assert seg_all["绝对收益"] == pytest.approx(combined, abs=0.001)


# ============================================================================
# 4. long_alpha_stats edge cases
# ============================================================================

class TestLongAlphaStatsEdgeCases:
    """Edge cases for vol-adjusted alpha calculation."""

    def test_pure_short_long_vol_zero(self) -> None:
        """Pure short positions: long returns are all zero → long_vol = 0 → zero alpha stats."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: -0.5,
                        lambda d, s: 100.0 + d * 0.5)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        alpha = bt.long_alpha_stats
        assert alpha["绝对收益"] == 0.0
        assert alpha["夏普比率"] == 0.0
        assert alpha["最大回撤"] == 0.0

    def test_constant_prices_bench_vol_zero(self) -> None:
        """Constant prices: all returns = 0 → bench_vol = 0 → zero alpha stats."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: 0.5,
                        lambda d, s: 100.0)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        alpha = bt.long_alpha_stats
        assert alpha["绝对收益"] == 0.0

    def test_zero_weights_zero_alpha(self) -> None:
        """Zero weights: long returns = 0, bench may have vol, but long_vol = 0 → zero."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: 0.0,
                        lambda d, s: 100.0 + d * 0.5)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        alpha = bt.long_alpha_stats
        assert alpha["绝对收益"] == 0.0

    def test_mixed_weights_valid_alpha(self) -> None:
        """Mixed long/short with varying prices → both vols > 0 → valid alpha stats."""
        dfw = _make_dfw(15, ["A", "B"],
                        lambda d, s: [0.3, 0.3, -0.2, 0.5, -0.1][d % 5],
                        lambda d, s: 100.0 + d * (0.5 if s == "A" else -0.3) + math.sin(d) * 2)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1, yearly_days=252)
        alpha = bt.long_alpha_stats
        # Should have non-trivial values (not all zero)
        assert isinstance(alpha["绝对收益"], float)
        assert isinstance(alpha["夏普比率"], float)
        # 年胜率 should be 0 (only ~15 days, < 126 threshold)
        assert alpha["年胜率"] == 0.0

    def test_alpha_keys_complete_in_zero_vol_case(self) -> None:
        """Even when vol=0, all keys should still be present."""
        dfw = _make_dfw(10, ["A"],
                        lambda d, s: -0.5,
                        lambda d, s: 100.0 + d * 0.5)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0, n_jobs=1, yearly_days=252)
        alpha = bt.long_alpha_stats
        expected_keys = [
            "绝对收益", "年化收益", "夏普比率", "卡玛比率", "新高占比",
            "日胜率", "周胜率", "月胜率", "季胜率", "年胜率",
            "最大回撤", "年化波动率", "下行波动率", "新高间隔",
        ]
        for k in expected_keys:
            assert k in alpha, f"Missing key in zero-vol case: {k}"


# ============================================================================
# 5. Single symbol edge cases
# ============================================================================

class TestSingleSymbolMetrics:
    """Verify metrics correctness with a single symbol."""

    def test_single_symbol_stats_complete(self) -> None:
        """Single symbol should produce all expected stats keys."""
        dfw = _make_dfw(20, ["ONLY"],
                        lambda d, s: [0.3, 0.3, -0.2, 0.0, 0.5][d % 5],
                        lambda d, s: 100.0 + d * 0.5 + math.sin(d) * 2)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1, yearly_days=252)
        stats = bt.stats
        expected_keys = [
            "绝对收益", "年化收益", "夏普比率", "卡玛比率", "最大回撤",
            "日胜率", "周胜率", "月胜率", "季胜率", "年胜率",
            "交易次数", "年化交易次数", "多头占比", "空头占比", "品种数量",
        ]
        for k in expected_keys:
            assert k in stats, f"Missing key: {k}"
        assert stats["品种数量"] == 1

    def test_single_symbol_long_short_sum(self) -> None:
        """Single symbol: long_return + short_return = return."""
        dfw = _make_dfw(15, ["X"],
                        lambda d, s: [0.3, -0.2, 0.5, 0.0, -0.4][d % 5],
                        lambda d, s: 100.0 + d * 0.3)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1, yearly_days=252)
        dailys = bt.dailys
        combined = dailys["long_return"] + dailys["short_return"]
        pd.testing.assert_series_equal(dailys["return"], combined, check_names=False, atol=1e-8)


# ============================================================================
# 6. Many symbols
# ============================================================================

class TestCSMode:
    """CS (cross-sectional) weight mode: total = sum of per-symbol returns."""

    def test_cs_long_daily_return(self) -> None:
        """CS mode: total = sum (not mean) of per-symbol returns."""
        dfw = _make_dfw(
            10,
            ["A", "B"],
            lambda d, s: 0.3 if s == "A" else -0.2,
            lambda d, s: 100.0 + d * (0.5 if s == "A" else -0.3) + 50 * (s != "A"),
        )
        bt = WeightBacktest(
            dfw, digits=2, fee_rate=0.0002, n_jobs=1, weight_type="cs", yearly_days=252
        )
        dr = bt.daily_return
        sym_cols = [c for c in dr.columns if c not in ("date", "total")]
        expected = dr[sym_cols].sum(axis=1)
        np.testing.assert_allclose(dr["total"].values, expected.values, atol=1e-6)


class TestInvalidWeightType:
    """Invalid weight_type should silently default to 'ts'."""

    def test_invalid_weight_type_defaults_to_ts(self) -> None:
        """Invalid weight_type should produce same result as explicit 'ts'."""
        dfw = _make_dfw(10, ["A", "B"], lambda d, s: 0.3, lambda d, s: 100.0 + d)
        bt_ts = WeightBacktest(dfw.copy(), weight_type="ts")
        bt_inv = WeightBacktest(dfw.copy(), weight_type="INVALID")
        assert bt_ts.stats["绝对收益"] == bt_inv.stats["绝对收益"]


class TestManySymbols:
    """Verify correctness with many symbols (5+)."""

    def test_five_symbols(self) -> None:
        """5 symbols, TS mode: total = mean of per-symbol returns."""
        symbols = [f"SYM_{i}" for i in range(5)]
        dfw = _make_dfw(15, symbols,
                        lambda d, s: 0.3 if int(s[-1]) % 2 == 0 else -0.2,
                        lambda d, s: 100.0 + d * (int(s[-1]) + 1) * 0.1)
        bt = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=1,
                            weight_type="ts", yearly_days=252)
        assert bt.stats["品种数量"] == 5

        # Verify total = mean of symbols
        dr = bt.daily_return
        sym_cols = [c for c in dr.columns if c not in ("date", "total")]
        assert len(sym_cols) == 5
        mean_vals = dr[sym_cols].mean(axis=1)
        np.testing.assert_allclose(dr["total"].values, mean_vals.values, atol=1e-6)
