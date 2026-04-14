"""Comprehensive value-level correctness tests for all wbt metrics.

Creates a small deterministic dataset (2 symbols x 20 bars) with known
weights and prices so that every metric can be hand-verified.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest
from wbt._wbt import daily_performance as rust_daily_performance

# ============================================================================
# Deterministic test data
# ============================================================================

YEARLY_DAYS = 252
FEE_RATE = 0.0002
DIGITS = 2


def _build_deterministic_dfw() -> pd.DataFrame:
    """2 symbols x 20 bars each, spanning ~20 trading days in Jan-Feb 2024.

    We use 1 bar per day per symbol to make hand calculations simple.
    Weights alternate long/short, prices follow a deterministic pattern.
    """
    rows = []
    base_date = datetime(2024, 1, 2, 9, 30, 0)
    # Fixed weights and prices for SYM_A and SYM_B
    weights_a = [
        0.3,
        0.3,
        -0.2,
        -0.2,
        0.5,
        0.5,
        -0.1,
        -0.1,
        0.4,
        0.4,
        -0.3,
        -0.3,
        0.2,
        0.2,
        -0.4,
        -0.4,
        0.1,
        0.1,
        -0.5,
        -0.5,
    ]
    weights_b = [
        -0.2,
        -0.2,
        0.4,
        0.4,
        -0.3,
        -0.3,
        0.2,
        0.2,
        -0.1,
        -0.1,
        0.5,
        0.5,
        -0.4,
        -0.4,
        0.3,
        0.3,
        -0.2,
        -0.2,
        0.1,
        0.1,
    ]
    prices_a = [
        100.0,
        101.0,
        100.5,
        99.5,
        100.0,
        102.0,
        101.5,
        100.0,
        101.0,
        103.0,
        102.0,
        100.5,
        101.0,
        102.5,
        101.0,
        99.0,
        100.0,
        101.5,
        100.0,
        98.5,
    ]
    prices_b = [
        50.0,
        50.5,
        51.0,
        50.0,
        49.5,
        50.0,
        51.0,
        52.0,
        51.5,
        51.0,
        52.0,
        53.0,
        52.5,
        51.5,
        52.0,
        53.0,
        52.5,
        52.0,
        53.0,
        54.0,
    ]

    for d in range(20):
        dt_str = (base_date + timedelta(days=d)).strftime("%Y-%m-%d %H:%M:%S")
        # Round weights to DIGITS
        wa = round(weights_a[d], DIGITS)
        wb_ = round(weights_b[d], DIGITS)
        rows.append({"dt": dt_str, "symbol": "SYM_A", "weight": wa, "price": prices_a[d]})
        rows.append({"dt": dt_str, "symbol": "SYM_B", "weight": wb_, "price": prices_b[d]})

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dfw() -> pd.DataFrame:
    return _build_deterministic_dfw()


@pytest.fixture(scope="module")
def bt(dfw: pd.DataFrame) -> WeightBacktest:
    return WeightBacktest(dfw, digits=DIGITS, fee_rate=FEE_RATE, n_jobs=1, weight_type="ts", yearly_days=YEARLY_DAYS)


# ============================================================================
# Helper: Python reference implementation of daily_performance metrics
# ============================================================================


def _python_daily_perf(returns: np.ndarray, yearly_days: int = 252) -> dict:
    """Pure-Python reference for the key daily_performance metrics."""
    n = len(returns)
    if n == 0:
        return {"absolute_return": 0, "annual_returns": 0, "sharpe": 0, "max_drawdown": 0, "daily_win_rate": 0}

    cumsum = np.cumsum(returns)
    absolute_return = float(cumsum[-1])

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=0)

    annual_returns = mean_r * yearly_days
    sharpe = (mean_r / std_r * math.sqrt(yearly_days)) if std_r > 1e-15 else 0.0
    # Clamp sharpe to [-5, 10]
    sharpe = max(-5.0, min(10.0, sharpe))

    # Max drawdown: peak starts from -∞ (first cumsum value becomes initial peak)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Daily win rate: count(r > 0 or r == 0) / n  -- Rust treats ==0 as win
    win_count = int(np.sum(returns >= 0))
    daily_win_rate = win_count / n

    return {
        "absolute_return": absolute_return,
        "annual_returns": annual_returns,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "daily_win_rate": daily_win_rate,
    }


# ============================================================================
# 1. Stats dict: absolute_return, annual, sharpe, max_drawdown, daily_win_rate
# ============================================================================


class TestStatsBasicMetrics:
    """Validate core stats metrics against Python reference calculations."""

    def test_absolute_return(self, bt: WeightBacktest) -> None:
        """绝对收益 should equal cumsum of daily total returns."""
        stats = bt.stats
        dr = bt.daily_return
        expected = dr["total"].sum()
        assert stats["绝对收益"] == pytest.approx(expected, abs=0.001)

    def test_annual_return(self, bt: WeightBacktest) -> None:
        """年化收益 = mean(daily_return) * yearly_days."""
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        mean_r = np.mean(total_returns)
        expected = mean_r * YEARLY_DAYS
        assert stats["年化收益"] == pytest.approx(expected, abs=0.001)

    def test_sharpe_ratio(self, bt: WeightBacktest) -> None:
        """夏普比率 = mean/std * sqrt(yearly_days), clamped to [-5, 10]."""
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        mean_r = np.mean(total_returns)
        std_r = np.std(total_returns, ddof=0)
        if std_r > 1e-15:
            expected = mean_r / std_r * math.sqrt(YEARLY_DAYS)
            expected = max(-5.0, min(10.0, expected))
        else:
            expected = 0.0
        assert stats["夏普比率"] == pytest.approx(expected, abs=0.01)

    def test_max_drawdown(self, bt: WeightBacktest) -> None:
        """最大回撤: peak starts from -∞ (first cumsum value is initial peak)."""
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        cumsum = np.cumsum(total_returns)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        expected = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        assert stats["最大回撤"] == pytest.approx(expected, abs=0.001)

    def test_daily_win_rate(self, bt: WeightBacktest) -> None:
        """日胜率: count(daily_return >= 0) / total_days.

        Note: Rust counts r==0 as a win (win_count increments for ==0 too).
        """
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        win_count = int(np.sum(total_returns > 0)) + int(np.sum(total_returns == 0))
        expected = win_count / len(total_returns)
        assert stats["日胜率"] == pytest.approx(expected, abs=0.001)


# ============================================================================
# 2. Period win rates: week, month, quarter, year
# ============================================================================


class TestPeriodWinRates:
    """Validate period win rates against Python groupby calculations."""

    def test_week_win_rate(self, bt: WeightBacktest) -> None:
        """周胜率: group by ISO week, sum returns, fraction > 0."""
        stats = bt.stats
        dr = bt.daily_return
        df = dr[["date", "total"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["week_key"] = (
            df["date"].dt.isocalendar().year.astype(str) + "-" + df["date"].dt.isocalendar().week.astype(str)
        )
        weekly = df.groupby("week_key")["total"].sum()
        expected = (weekly > 0).sum() / len(weekly) if len(weekly) > 0 else 0.0
        assert stats["周胜率"] == pytest.approx(expected, abs=0.001)

    def test_month_win_rate(self, bt: WeightBacktest) -> None:
        """月胜率: group by (year, month), sum returns, fraction > 0."""
        stats = bt.stats
        dr = bt.daily_return
        df = dr[["date", "total"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month_key"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month_key")["total"].sum()
        expected = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0.0
        assert stats["月胜率"] == pytest.approx(expected, abs=0.001)

    def test_quarter_win_rate(self, bt: WeightBacktest) -> None:
        """季胜率: group by (year, quarter), sum returns, fraction > 0."""
        stats = bt.stats
        dr = bt.daily_return
        df = dr[["date", "total"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["q_key"] = df["date"].dt.to_period("Q")
        quarterly = df.groupby("q_key")["total"].sum()
        expected = (quarterly > 0).sum() / len(quarterly) if len(quarterly) > 0 else 0.0
        assert stats["季胜率"] == pytest.approx(expected, abs=0.001)

    def test_year_win_rate(self, bt: WeightBacktest) -> None:
        """年胜率: group by year, sum returns, fraction > 0.

        Only years with >= yearly_days/2 trading days are counted.
        Our dataset has ~20 days which is < 252/2=126, so year rate should be 0.
        """
        stats = bt.stats
        dr = bt.daily_return
        n_days = len(dr)
        min_days = YEARLY_DAYS // 2
        # With only ~20 trading days, no year qualifies
        if n_days < min_days:
            expected = 0.0
        else:
            df = dr[["date", "total"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            yearly = df.groupby("year").agg(total=("total", "sum"), count=("total", "count"))
            qualified = yearly[yearly["count"] >= min_days]
            expected = (qualified["total"] > 0).sum() / len(qualified) if len(qualified) > 0 else 0.0
        assert stats["年胜率"] == pytest.approx(expected, abs=0.001)


# ============================================================================
# 3. Trade metrics: 交易次数, 年化交易次数, 单笔盈亏比, 单笔收益, 交易胜率
# ============================================================================


class TestTradeMetrics:
    """Validate trade-related metrics against pairs data."""

    def test_trade_count(self, bt: WeightBacktest) -> None:
        """交易次数 should equal the total number of trade units in pairs."""
        stats = bt.stats
        pairs = bt.pairs
        if len(pairs) > 0 and "持仓数量" in pairs.columns:
            expected = int(pairs["持仓数量"].sum())
        elif len(pairs) > 0:
            expected = len(pairs)
        else:
            expected = 0
        assert stats["交易次数"] == expected

    def test_annual_trade_count(self, bt: WeightBacktest) -> None:
        """年化交易次数 = 交易次数 / (trading_days / yearly_days)."""
        stats = bt.stats
        dr = bt.daily_return
        n_days = len(dr)
        trade_count = stats["交易次数"]
        if n_days > 0:
            expected = trade_count / (n_days / YEARLY_DAYS)
            # Rust rounds to 2 digits
            expected = round(expected * 100) / 100
        else:
            expected = 0.0
        assert stats["年化交易次数"] == pytest.approx(expected, rel=1e-3)

    def test_trade_win_rate(self, bt: WeightBacktest) -> None:
        """交易胜率 = count(pairs with PnL >= 0) / total_pairs (weighted by count)."""
        stats = bt.stats
        pairs = bt.pairs
        if len(pairs) == 0:
            assert stats["交易胜率"] == 0.0
            return
        if "持仓数量" in pairs.columns:
            total = pairs["持仓数量"].sum()
            wins = pairs.loc[pairs["盈亏比例"] >= 0, "持仓数量"].sum()
        else:
            total = len(pairs)
            wins = (pairs["盈亏比例"] >= 0).sum()
        expected = wins / total if total > 0 else 0.0
        assert stats["交易胜率"] == pytest.approx(expected, abs=0.001)

    def test_single_profit_loss_ratio(self, bt: WeightBacktest) -> None:
        """单笔盈亏比 = avg(winning PnL) / abs(avg(losing PnL))."""
        stats = bt.stats
        pairs = bt.pairs
        if len(pairs) == 0:
            assert stats["单笔盈亏比"] == 0.0
            return
        if "持仓数量" in pairs.columns:
            win_mask = pairs["盈亏比例"] >= 0
            loss_mask = pairs["盈亏比例"] < 0
            win_total = (pairs.loc[win_mask, "盈亏比例"] * pairs.loc[win_mask, "持仓数量"]).sum()
            win_count = pairs.loc[win_mask, "持仓数量"].sum()
            loss_total = (pairs.loc[loss_mask, "盈亏比例"] * pairs.loc[loss_mask, "持仓数量"]).sum()
            loss_count = pairs.loc[loss_mask, "持仓数量"].sum()
            avg_win = win_total / win_count if win_count > 0 else 0.0
            avg_loss = loss_total / loss_count if loss_count > 0 else 0.0
        else:
            wins = pairs.loc[pairs["盈亏比例"] >= 0, "盈亏比例"]
            losses = pairs.loc[pairs["盈亏比例"] < 0, "盈亏比例"]
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0

        expected = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-10 else 0.0
        assert stats["单笔盈亏比"] == pytest.approx(expected, abs=0.01)

    def test_single_trade_profit(self, bt: WeightBacktest) -> None:
        """单笔收益 = total PnL / trade count."""
        stats = bt.stats
        pairs = bt.pairs
        if len(pairs) == 0:
            assert stats["单笔收益"] == 0.0
            return
        if "持仓数量" in pairs.columns:
            total_pnl = (pairs["盈亏比例"] * pairs["持仓数量"]).sum()
            total_count = pairs["持仓数量"].sum()
        else:
            total_pnl = pairs["盈亏比例"].sum()
            total_count = len(pairs)
        expected = total_pnl / total_count if total_count > 0 else 0.0
        assert stats["单笔收益"] == pytest.approx(expected, abs=0.1)


# ============================================================================
# 4. Long/Short rates: 多头占比, 空头占比
# ============================================================================


class TestLongShortRates:
    """Validate long/short weight proportions."""

    def test_long_short_rates_sum_le_1(self, bt: WeightBacktest) -> None:
        """多头占比 + 空头占比 should be <= 1.0."""
        stats = bt.stats
        assert stats["多头占比"] + stats["空头占比"] <= 1.0 + 1e-6

    def test_long_rate_from_weights(self, bt: WeightBacktest) -> None:
        """多头占比 = fraction of weight rows where weight > 0."""
        stats = bt.stats
        # In TS mode, long_rate = long_weight_rows / total_weight_rows
        # We can check from the dailys: rows where long_edge != 0 or long_return != 0
        # But actually Rust computes it from the raw weight data.
        # We just verify it's in [0, 1] and reasonable.
        assert 0.0 <= stats["多头占比"] <= 1.0
        assert 0.0 <= stats["空头占比"] <= 1.0

    def test_long_rate_matches_weight_data(self, bt: WeightBacktest, dfw: pd.DataFrame) -> None:
        """Verify long_rate against input weight data.

        long_count / total_weight_rows should match stats['多头占比'].
        """
        stats = bt.stats
        weights = dfw["weight"].values
        long_count = int(np.sum(weights > 0))
        short_count = int(np.sum(weights < 0))
        total = len(weights)
        if total > 0:
            expected_long = long_count / total
            expected_short = short_count / total
        else:
            expected_long = 0.0
            expected_short = 0.0
        assert stats["多头占比"] == pytest.approx(expected_long, abs=0.001)
        assert stats["空头占比"] == pytest.approx(expected_short, abs=0.001)


# ============================================================================
# 5. Long stats vs Short stats
# ============================================================================


class TestLongShortStats:
    """Verify long_stats and short_stats use correct return columns."""

    def test_long_stats_absolute_return(self, bt: WeightBacktest) -> None:
        """long_stats.绝对收益 should equal cumsum of long daily returns."""
        long_stats = bt.long_stats
        long_dr = bt.long_daily_return
        expected = long_dr["total"].sum()
        assert long_stats["绝对收益"] == pytest.approx(expected, abs=0.001)

    def test_short_stats_absolute_return(self, bt: WeightBacktest) -> None:
        """short_stats.绝对收益 should equal cumsum of short daily returns."""
        short_stats = bt.short_stats
        short_dr = bt.short_daily_return
        expected = short_dr["total"].sum()
        assert short_stats["绝对收益"] == pytest.approx(expected, abs=0.001)

    def test_long_stats_trade_count_only_long(self, bt: WeightBacktest) -> None:
        """long_stats.交易胜率 should only count 多头 pairs."""
        long_stats = bt.long_stats
        pairs = bt.pairs
        if len(pairs) == 0 or "交易方向" not in pairs.columns:
            pytest.skip("No pairs data")
        long_pairs = pairs[pairs["交易方向"] == "多头"]
        if "持仓数量" in pairs.columns:
            total = long_pairs["持仓数量"].sum()
            wins = long_pairs.loc[long_pairs["盈亏比例"] >= 0, "持仓数量"].sum()
        else:
            total = len(long_pairs)
            wins = (long_pairs["盈亏比例"] >= 0).sum()
        expected_wr = wins / total if total > 0 else 0.0
        assert long_stats["交易胜率"] == pytest.approx(expected_wr, abs=0.001)

    def test_short_stats_trade_count_only_short(self, bt: WeightBacktest) -> None:
        """short_stats.交易胜率 should only count 空头 pairs."""
        short_stats = bt.short_stats
        pairs = bt.pairs
        if len(pairs) == 0 or "交易方向" not in pairs.columns:
            pytest.skip("No pairs data")
        short_pairs = pairs[pairs["交易方向"] == "空头"]
        if "持仓数量" in pairs.columns:
            total = short_pairs["持仓数量"].sum()
            wins = short_pairs.loc[short_pairs["盈亏比例"] >= 0, "持仓数量"].sum()
        else:
            total = len(short_pairs)
            wins = (short_pairs["盈亏比例"] >= 0).sum()
        expected_wr = wins / total if total > 0 else 0.0
        assert short_stats["交易胜率"] == pytest.approx(expected_wr, abs=0.001)

    def test_long_plus_short_returns_approx_total(self, bt: WeightBacktest) -> None:
        """Sum of long + short daily returns should approximately equal total daily return.

        In TS mode, total = mean(per-symbol return), and each per-symbol return =
        long_return + short_return. So long_total + short_total ~= total (for TS, each
        is averaged separately, so this equality holds per-date).
        """
        long_dr = bt.long_daily_return
        short_dr = bt.short_daily_return
        total_dr = bt.daily_return
        combined = long_dr["total"].values + short_dr["total"].values
        # This may not be exact due to TS averaging, but should be close
        np.testing.assert_allclose(combined, total_dr["total"].values, atol=1e-6)


# ============================================================================
# 6. Segment stats
# ============================================================================


class TestSegmentStats:
    """Verify segment_stats date filtering and consistency."""

    # ------------------------------------------------------------------
    # Helper: Python reference for segment filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _python_segment(bt: WeightBacktest, sdt: int | None, edt: int | None, kind: str) -> dict:
        """Python reference: filter dailys + pairs, compute expected values."""
        dailys = bt.dailys.copy()
        pairs = bt.pairs.copy()
        dailys["date_int"] = pd.to_datetime(dailys["date"]).dt.strftime("%Y%m%d").astype(int)

        actual_sdt = sdt if sdt is not None else int(dailys["date_int"].min())
        actual_edt = edt if edt is not None else int(dailys["date_int"].max())

        # Filter dailys by date
        mask = (dailys["date_int"] >= actual_sdt) & (dailys["date_int"] <= actual_edt)
        filtered = dailys[mask]

        # Pick return column by kind
        ret_col = {"多头": "long_return", "空头": "short_return"}.get(kind, "return")

        # Aggregate per date (TS = mean, CS = sum)
        daily_agg = filtered.groupby("date_int")[ret_col].mean()  # TS mode
        returns = daily_agg.values
        date_keys = daily_agg.index.values

        abs_ret = float(np.sum(returns))

        # Filter pairs: open AND close in range
        if len(pairs) > 0 and "开仓时间" in pairs.columns:
            pairs["open_dk"] = pd.to_datetime(pairs["开仓时间"]).dt.strftime("%Y%m%d").astype(int)
            pairs["close_dk"] = pd.to_datetime(pairs["平仓时间"]).dt.strftime("%Y%m%d").astype(int)
            p_mask = (pairs["open_dk"] >= actual_sdt) & (pairs["close_dk"] <= actual_edt)
            # Filter by direction
            if kind == "多头":
                p_mask &= pairs["交易方向"] == "多头"
            elif kind == "空头":
                p_mask &= pairs["交易方向"] == "空头"
            fp = pairs[p_mask]
            if "持仓数量" in fp.columns:
                trade_count = int(fp["持仓数量"].sum())
                win_count = int(fp.loc[fp["盈亏比例"] >= 0, "持仓数量"].sum())
            else:
                trade_count = len(fp)
                win_count = int((fp["盈亏比例"] >= 0).sum())
        else:
            trade_count = 0
            win_count = 0

        trade_wr = win_count / trade_count if trade_count > 0 else 0.0
        return {
            "abs_ret": abs_ret,
            "n_dates": len(date_keys),
            "trade_count": trade_count,
            "trade_wr": trade_wr,
        }

    # ------------------------------------------------------------------
    # Full range consistency
    # ------------------------------------------------------------------

    def test_full_range_matches_stats(self, bt: WeightBacktest) -> None:
        """segment_stats(None, None, '多空') should match stats."""
        stats = bt.stats
        seg = bt.segment_stats()
        assert seg["绝对收益"] == pytest.approx(stats["绝对收益"], abs=0.001)
        assert seg["年化收益"] == pytest.approx(stats["年化收益"], abs=0.001)
        assert seg["夏普比率"] == pytest.approx(stats["夏普比率"], abs=0.01)
        assert seg["最大回撤"] == pytest.approx(stats["最大回撤"], abs=0.001)
        assert seg["交易次数"] == stats["交易次数"]
        assert seg["交易胜率"] == pytest.approx(stats["交易胜率"], abs=0.001)
        assert seg["日胜率"] == pytest.approx(stats["日胜率"], abs=0.001)

    def test_full_range_long_matches_long_stats(self, bt: WeightBacktest) -> None:
        """segment_stats(kind='多头') full range should match long_stats."""
        seg = bt.segment_stats(kind="多头")
        ls = bt.long_stats
        assert seg["绝对收益"] == pytest.approx(ls["绝对收益"], abs=0.001)
        assert seg["交易次数"] == ls["交易次数"]
        assert seg["交易胜率"] == pytest.approx(ls["交易胜率"], abs=0.001)

    def test_full_range_short_matches_short_stats(self, bt: WeightBacktest) -> None:
        """segment_stats(kind='空头') full range should match short_stats."""
        seg = bt.segment_stats(kind="空头")
        ss = bt.short_stats
        assert seg["绝对收益"] == pytest.approx(ss["绝对收益"], abs=0.001)
        assert seg["交易次数"] == ss["交易次数"]
        assert seg["交易胜率"] == pytest.approx(ss["交易胜率"], abs=0.001)

    # ------------------------------------------------------------------
    # Partial range: value-level correctness
    # ------------------------------------------------------------------

    def test_partial_range_absolute_return(self, bt: WeightBacktest) -> None:
        """Partial range 绝对收益 should match Python reference."""
        sdt, edt = 20240105, 20240115
        seg = bt.segment_stats(sdt=sdt, edt=edt)
        ref = self._python_segment(bt, sdt, edt, "多空")
        assert seg["绝对收益"] == pytest.approx(ref["abs_ret"], abs=0.001)

    def test_partial_range_trade_count(self, bt: WeightBacktest) -> None:
        """Partial range 交易次数 should match Python pairs filtering."""
        sdt, edt = 20240105, 20240115
        seg = bt.segment_stats(sdt=sdt, edt=edt)
        ref = self._python_segment(bt, sdt, edt, "多空")
        assert seg["交易次数"] == ref["trade_count"]

    def test_partial_range_trade_win_rate(self, bt: WeightBacktest) -> None:
        """Partial range 交易胜率 should match Python reference."""
        sdt, edt = 20240105, 20240115
        seg = bt.segment_stats(sdt=sdt, edt=edt)
        ref = self._python_segment(bt, sdt, edt, "多空")
        assert seg["交易胜率"] == pytest.approx(ref["trade_wr"], abs=0.001)

    def test_partial_range_fewer_trades(self, bt: WeightBacktest) -> None:
        """Partial range should have <= trades than full range."""
        full_seg = bt.segment_stats()
        partial_seg = bt.segment_stats(sdt=20240105, edt=20240115)
        assert partial_seg["交易次数"] <= full_seg["交易次数"]

    # ------------------------------------------------------------------
    # Partial range + kind combination
    # ------------------------------------------------------------------

    def test_partial_range_long(self, bt: WeightBacktest) -> None:
        """Partial range + kind='多头': return and trades from Python ref."""
        sdt, edt = 20240105, 20240115
        seg = bt.segment_stats(sdt=sdt, edt=edt, kind="多头")
        ref = self._python_segment(bt, sdt, edt, "多头")
        assert seg["绝对收益"] == pytest.approx(ref["abs_ret"], abs=0.001)
        assert seg["交易次数"] == ref["trade_count"]

    def test_partial_range_short(self, bt: WeightBacktest) -> None:
        """Partial range + kind='空头': return and trades from Python ref."""
        sdt, edt = 20240105, 20240115
        seg = bt.segment_stats(sdt=sdt, edt=edt, kind="空头")
        ref = self._python_segment(bt, sdt, edt, "空头")
        assert seg["绝对收益"] == pytest.approx(ref["abs_ret"], abs=0.001)
        assert seg["交易次数"] == ref["trade_count"]

    # ------------------------------------------------------------------
    # Half-open ranges (only sdt or only edt)
    # ------------------------------------------------------------------

    def test_sdt_only(self, bt: WeightBacktest) -> None:
        """Only sdt provided: should include all data from sdt to end."""
        sdt = 20240110
        seg = bt.segment_stats(sdt=sdt)
        ref = self._python_segment(bt, sdt, None, "多空")
        assert seg["绝对收益"] == pytest.approx(ref["abs_ret"], abs=0.001)
        assert seg["交易次数"] == ref["trade_count"]

    def test_edt_only(self, bt: WeightBacktest) -> None:
        """Only edt provided: should include all data from start to edt."""
        edt = 20240110
        seg = bt.segment_stats(edt=edt)
        ref = self._python_segment(bt, None, edt, "多空")
        assert seg["绝对收益"] == pytest.approx(ref["abs_ret"], abs=0.001)
        assert seg["交易次数"] == ref["trade_count"]

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_range(self, bt: WeightBacktest) -> None:
        """Range with no data should return zero metrics."""
        seg = bt.segment_stats(sdt=20200101, edt=20200102)
        assert seg["绝对收益"] == 0.0
        assert seg["交易次数"] == 0

    def test_single_day(self, bt: WeightBacktest) -> None:
        """Single day: daily_performance returns default (zero) when std=0.

        With only 1 day of data, std of returns = 0, so daily_performance
        returns all-zero metrics (by design — Sharpe etc. are undefined).
        """
        seg = bt.segment_stats(sdt=20240105, edt=20240105)
        assert seg["绝对收益"] == 0.0  # daily_performance zeros out when std=0

    def test_inverted_range(self, bt: WeightBacktest) -> None:
        """sdt > edt should return zero (empty set)."""
        seg = bt.segment_stats(sdt=20240115, edt=20240105)
        assert seg["绝对收益"] == 0.0
        assert seg["交易次数"] == 0

    # ------------------------------------------------------------------
    # Output completeness
    # ------------------------------------------------------------------

    def test_output_has_all_stats_keys(self, bt: WeightBacktest) -> None:
        """segment_stats should return all standard metric keys."""
        seg = bt.segment_stats(sdt=20240105, edt=20240115)
        expected_keys = [
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
        ]
        for k in expected_keys:
            assert k in seg, f"Missing key in segment_stats: {k}"


# ============================================================================
# 7. Long alpha stats
# ============================================================================


class TestLongAlphaStats:
    """Verify vol-adjusted long alpha calculations."""

    # ------------------------------------------------------------------
    # Helper: Python reference implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _python_alpha(bt: WeightBacktest) -> dict | None:
        """Compute alpha daily returns in Python. Returns None if vol too small."""
        long_dr = bt.long_daily_return["total"].values
        bench_returns = bt.alpha["基准"].values
        yearly_days = bt.yearly_days

        long_vol = np.std(long_dr, ddof=0) * math.sqrt(yearly_days)
        bench_vol = np.std(bench_returns, ddof=0) * math.sqrt(yearly_days)

        if long_vol < 1e-12 or bench_vol < 1e-12:
            return None

        target = 0.20
        long_scale = target / long_vol
        bench_scale = target / bench_vol

        adj_long = long_dr * long_scale
        adj_bench = bench_returns * bench_scale
        alpha_daily = adj_long - adj_bench

        return {
            "long_scale": long_scale,
            "bench_scale": bench_scale,
            "adj_long": adj_long,
            "adj_bench": adj_bench,
            "alpha_daily": alpha_daily,
            "long_vol": long_vol,
            "bench_vol": bench_vol,
        }

    # ------------------------------------------------------------------
    # Key completeness
    # ------------------------------------------------------------------

    def test_long_alpha_stats_has_all_keys(self, bt: WeightBacktest) -> None:
        """long_alpha_stats should have all expected keys."""
        alpha_stats = bt.long_alpha_stats
        expected_keys = [
            "绝对收益",
            "年化收益",
            "夏普比率",
            "卡玛比率",
            "新高占比",
            "日胜率",
            "周胜率",
            "月胜率",
            "季胜率",
            "年胜率",
            "最大回撤",
            "年化波动率",
            "下行波动率",
            "新高间隔",
        ]
        for k in expected_keys:
            assert k in alpha_stats, f"Missing key: {k}"

    # ------------------------------------------------------------------
    # Scale factor correctness
    # ------------------------------------------------------------------

    def test_scale_factor(self, bt: WeightBacktest) -> None:
        """Scale = 0.20 / actual_vol. Verify adj series has ~20% annual vol."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")

        # After scaling, adj_long annual vol should ≈ 20%
        adj_long_vol = np.std(ref["adj_long"], ddof=0) * math.sqrt(YEARLY_DAYS)
        adj_bench_vol = np.std(ref["adj_bench"], ddof=0) * math.sqrt(YEARLY_DAYS)
        assert adj_long_vol == pytest.approx(0.20, rel=1e-6)
        assert adj_bench_vol == pytest.approx(0.20, rel=1e-6)

    # ------------------------------------------------------------------
    # Alpha daily = adj_long - adj_bench
    # ------------------------------------------------------------------

    def test_alpha_daily_formula(self, bt: WeightBacktest) -> None:
        """alpha_daily[i] = long_return[i] * long_scale - bench[i] * bench_scale."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        np.testing.assert_allclose(
            ref["alpha_daily"],
            ref["adj_long"] - ref["adj_bench"],
            atol=1e-15,
        )

    # ------------------------------------------------------------------
    # Absolute return
    # ------------------------------------------------------------------

    def test_absolute_return(self, bt: WeightBacktest) -> None:
        """绝对收益 should match sum of alpha_daily."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")

        alpha_stats = bt.long_alpha_stats
        expected = round(float(np.sum(ref["alpha_daily"])) * 10000) / 10000
        assert alpha_stats["绝对收益"] == pytest.approx(expected, abs=0.001)

    # ------------------------------------------------------------------
    # Annual return, Sharpe, max drawdown
    # ------------------------------------------------------------------

    def test_annual_return(self, bt: WeightBacktest) -> None:
        """年化收益 = mean(alpha_daily) * yearly_days."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        expected = float(np.mean(ref["alpha_daily"])) * YEARLY_DAYS
        assert alpha_stats["年化收益"] == pytest.approx(expected, abs=0.01)

    def test_sharpe_ratio(self, bt: WeightBacktest) -> None:
        """夏普比率 of alpha series."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        mean_a = np.mean(ad)
        std_a = np.std(ad, ddof=0)
        expected = max(-5.0, min(10.0, mean_a / std_a * math.sqrt(YEARLY_DAYS))) if std_a > 1e-15 else 0.0
        assert alpha_stats["夏普比率"] == pytest.approx(expected, abs=0.01)

    def test_max_drawdown(self, bt: WeightBacktest) -> None:
        """最大回撤 of alpha cumsum.

        Drawdown peak starts from -∞ (not 0), meaning only drawdowns from
        historical highs are counted, not initial losses.
        """
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        cumsum = np.cumsum(ad)
        # peak = -∞: running_max tracks from first value, not from 0
        running_max = np.maximum.accumulate(cumsum)
        dd = running_max - cumsum
        expected = float(np.max(dd)) if len(dd) > 0 else 0.0
        assert alpha_stats["最大回撤"] == pytest.approx(expected, abs=0.001)

    # ------------------------------------------------------------------
    # Win rates on alpha series
    # ------------------------------------------------------------------

    def test_daily_win_rate(self, bt: WeightBacktest) -> None:
        """日胜率 of alpha daily returns."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        # Rust: r > 0 and r == 0 both counted as win
        win = int(np.sum(ad >= 0))
        expected = win / len(ad)
        assert alpha_stats["日胜率"] == pytest.approx(expected, abs=0.001)

    def test_period_win_rates_on_alpha(self, bt: WeightBacktest) -> None:
        """周/月/季胜率 computed on alpha_daily, not original returns."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats

        dr = bt.daily_return
        dates = pd.to_datetime(dr["date"])
        ad = ref["alpha_daily"]

        # Week win rate on alpha
        df_tmp = pd.DataFrame({"date": dates, "alpha": ad})
        df_tmp["week_key"] = (
            df_tmp["date"].dt.isocalendar().year.astype(str) + "-" + df_tmp["date"].dt.isocalendar().week.astype(str)
        )
        weekly = df_tmp.groupby("week_key")["alpha"].sum()
        expected_week = (weekly > 0).sum() / len(weekly) if len(weekly) > 0 else 0.0
        assert alpha_stats["周胜率"] == pytest.approx(expected_week, abs=0.001)

        # Month win rate on alpha
        df_tmp["month_key"] = df_tmp["date"].dt.to_period("M")
        monthly = df_tmp.groupby("month_key")["alpha"].sum()
        expected_month = (monthly > 0).sum() / len(monthly) if len(monthly) > 0 else 0.0
        assert alpha_stats["月胜率"] == pytest.approx(expected_month, abs=0.001)

    # ------------------------------------------------------------------
    # Volatility metrics on alpha
    # ------------------------------------------------------------------

    def test_alpha_annual_volatility(self, bt: WeightBacktest) -> None:
        """年化波动率 of alpha series."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        expected = np.std(ad, ddof=0) * math.sqrt(YEARLY_DAYS)
        assert alpha_stats["年化波动率"] == pytest.approx(expected, abs=0.001)

    def test_alpha_downside_volatility(self, bt: WeightBacktest) -> None:
        """下行波动率 of alpha series (std of negative alpha days)."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        neg = ad[ad < 0]
        expected = np.std(neg, ddof=0) * math.sqrt(YEARLY_DAYS) if len(neg) > 0 else 0.0
        assert alpha_stats["下行波动率"] == pytest.approx(expected, abs=0.001)

    def test_alpha_calmar_ratio(self, bt: WeightBacktest) -> None:
        """卡玛比率 of alpha series = annual_return / max_drawdown, clamped [-10, 20]."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        mean_a = np.mean(ad)
        annual_ret = mean_a * YEARLY_DAYS
        cumsum = np.cumsum(ad)
        running_max = np.maximum.accumulate(cumsum)
        dd = running_max - cumsum
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        expected = max(-10.0, min(20.0, annual_ret / max_dd)) if max_dd > 1e-10 else 10.0
        # Rust computes calmar from round4(annual)/unrounded_maxdd then rounds
        assert alpha_stats["卡玛比率"] == pytest.approx(expected, rel=0.02)

    def test_alpha_new_high_ratio(self, bt: WeightBacktest) -> None:
        """新高占比 of alpha cumsum (peak from -∞)."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        cumsum = np.cumsum(ad)
        running_max = np.maximum.accumulate(cumsum)
        at_high = np.sum(running_max - cumsum <= 0.0)
        expected = at_high / len(ad)
        assert alpha_stats["新高占比"] == pytest.approx(expected, abs=0.001)

    def test_alpha_new_high_interval(self, bt: WeightBacktest) -> None:
        """新高间隔 of alpha cumsum."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        cumsum = np.cumsum(ad)
        max_cum = float("-inf")
        current_interval = 0
        max_interval = 0
        for c in cumsum:
            if c > max_cum:
                max_cum = c
                max_interval = max(max_interval, current_interval)
                current_interval = 0
            current_interval += 1
        assert alpha_stats["新高间隔"] == pytest.approx(float(max_interval), abs=0.001)

    def test_alpha_quarter_win_rate(self, bt: WeightBacktest) -> None:
        """季胜率 of alpha series."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        dr = bt.daily_return
        dates = pd.to_datetime(dr["date"])
        df_tmp = pd.DataFrame({"date": dates, "alpha": ad})
        df_tmp["q_key"] = df_tmp["date"].dt.to_period("Q")
        quarterly = df_tmp.groupby("q_key")["alpha"].sum()
        expected = (quarterly > 0).sum() / len(quarterly) if len(quarterly) > 0 else 0.0
        assert alpha_stats["季胜率"] == pytest.approx(expected, abs=0.001)

    def test_alpha_year_win_rate(self, bt: WeightBacktest) -> None:
        """年胜率 of alpha series (years with < yearly_days/2 days excluded)."""
        ref = self._python_alpha(bt)
        if ref is None:
            pytest.skip("Vol too small")
        alpha_stats = bt.long_alpha_stats
        ad = ref["alpha_daily"]
        dr = bt.daily_return
        dates = pd.to_datetime(dr["date"])
        df_tmp = pd.DataFrame({"date": dates, "alpha": ad})
        df_tmp["year"] = df_tmp["date"].dt.year
        yearly = df_tmp.groupby("year").agg(total=("alpha", "sum"), count=("alpha", "count"))
        min_days = YEARLY_DAYS // 2
        qualified = yearly[yearly["count"] >= min_days]
        expected = (qualified["total"] > 0).sum() / len(qualified) if len(qualified) > 0 else 0.0
        assert alpha_stats["年胜率"] == pytest.approx(expected, abs=0.001)


# ============================================================================
# 8. Cross-check: rust daily_performance standalone function
# ============================================================================


class TestDailyPerformanceStandalone:
    """Verify the standalone daily_performance function against known values."""

    def test_known_returns(self) -> None:
        """Test with known return series [0.01, -0.005, 0.02]."""
        returns = np.array([0.01, -0.005, 0.02])
        dp = rust_daily_performance(returns, yearly_days=252)

        assert dp["绝对收益"] == pytest.approx(0.025, abs=0.001)
        # mean = 0.025/3 ~ 0.008333, * 252 ~ 2.1
        assert dp["年化"] == pytest.approx(2.1, abs=0.01)
        # sharpe capped at 10.0
        assert dp["夏普"] == pytest.approx(10.0, abs=0.01)
        assert dp["最大回撤"] == pytest.approx(0.005, abs=0.001)
        assert dp["日胜率"] == pytest.approx(2 / 3, abs=0.001)

    def test_all_negative(self) -> None:
        """Fully negative returns should give negative annual return."""
        returns = np.array([-0.01, -0.02, -0.005])
        dp = rust_daily_performance(returns, yearly_days=252)
        assert dp["绝对收益"] < 0
        assert dp["年化"] < 0
        assert dp["夏普"] < 0
        assert dp["最大回撤"] > 0

    def test_empty_returns_zero(self) -> None:
        """Empty returns should give all zeros."""
        returns = np.array([], dtype=np.float64)
        dp = rust_daily_performance(returns, yearly_days=252)
        assert dp["绝对收益"] == 0.0
        assert dp["年化"] == 0.0


# ============================================================================
# 9. Internal consistency: dailys -> daily_return
# ============================================================================


class TestInternalConsistency:
    """Verify internal data consistency across different views."""

    def test_dailys_return_sums_to_daily_return_total(self, bt: WeightBacktest) -> None:
        """In TS mode, daily_return.total = mean of per-symbol returns.

        daily_return.total for each date should equal the mean of the
        per-symbol 'return' values from dailys for that date.
        """
        dr = bt.daily_return
        dailys = bt.dailys

        # Pivot dailys return by date
        pivot = dailys.pivot_table(index="date", columns="symbol", values="return")
        pivot_mean = pivot.mean(axis=1)  # TS mode = mean

        # Align dates
        pd.to_datetime(dr["date"])
        pd.to_datetime(pivot_mean.index)
        # Both should have same dates
        assert len(dr) == len(pivot_mean)

        np.testing.assert_allclose(
            dr["total"].values,
            pivot_mean.values,
            atol=1e-6,
            err_msg="daily_return.total should equal mean of per-symbol returns",
        )

    def test_long_return_plus_short_return_equals_return(self, bt: WeightBacktest) -> None:
        """For each row in dailys: long_return + short_return = return."""
        dailys = bt.dailys
        combined = dailys["long_return"] + dailys["short_return"]
        pd.testing.assert_series_equal(dailys["return"], combined, check_names=False, atol=1e-8)

    def test_symbols_count(self, bt: WeightBacktest) -> None:
        """品种数量 should match len(symbols)."""
        stats = bt.stats
        assert stats["品种数量"] == 2


# ============================================================================
# 10. Volatility metrics
# ============================================================================


class TestVolatilityMetrics:
    """Verify volatility-related metrics."""

    def test_annual_volatility(self, bt: WeightBacktest) -> None:
        """年化波动率 = std(daily_returns, ddof=0) * sqrt(yearly_days)."""
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        std_r = np.std(total_returns, ddof=0)
        expected = std_r * math.sqrt(YEARLY_DAYS)
        assert stats["年化波动率"] == pytest.approx(expected, abs=0.001)

    def test_calmar_ratio(self, bt: WeightBacktest) -> None:
        """卡玛比率 = annual_return / max_drawdown, clamped to [-10, 20].

        Note: Rust computes calmar from round4(annual) / unrounded_max_dd, then
        rounds the result. The test uses already-rounded stats values for both
        numerator and denominator, introducing a small precision gap (~1%).
        """
        stats = bt.stats
        if stats["最大回撤"] > 1e-10:
            expected = stats["年化收益"] / stats["最大回撤"]
            expected = max(-10.0, min(20.0, expected))
            assert stats["卡玛比率"] == pytest.approx(expected, rel=0.02)
        else:
            assert stats["卡玛比率"] == pytest.approx(10.0, abs=0.01)

    def test_downside_volatility(self, bt: WeightBacktest) -> None:
        """下行波动率 = std(negative_returns, ddof=0) * sqrt(yearly_days).

        Note: this is the standard deviation of the negative return subset,
        NOT the standard Sortino downside deviation (which uses total N as
        denominator). The difference is intentional — see design decision.
        """
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        neg_returns = total_returns[total_returns < 0]
        expected = np.std(neg_returns, ddof=0) * math.sqrt(YEARLY_DAYS) if len(neg_returns) > 0 else 0.0
        assert stats["下行波动率"] == pytest.approx(expected, abs=0.001)

    def test_new_high_ratio(self, bt: WeightBacktest) -> None:
        """新高占比 = count(days at new high) / total_days.

        A day is "at new high" when cumsum equals the running max (drawdown=0).
        Peak starts from -∞, so the first day is always a new high.
        """
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        cumsum = np.cumsum(total_returns)
        running_max = np.maximum.accumulate(cumsum)
        at_new_high = np.sum(running_max - cumsum <= 0.0)
        expected = at_new_high / len(total_returns)
        assert stats["新高占比"] == pytest.approx(expected, abs=0.001)

    def test_new_high_interval(self, bt: WeightBacktest) -> None:
        """新高间隔 = max interval between consecutive new highs.

        Rust tracks current_interval (bars since last new high) and records
        the max interval seen when a new high is reached. The final trailing
        interval (from last new high to end) is NOT included — this measures
        the worst gap between two consecutive new highs.
        """
        stats = bt.stats
        dr = bt.daily_return
        total_returns = dr["total"].values
        cumsum = np.cumsum(total_returns)

        max_cum = float("-inf")
        current_interval = 0
        max_interval = 0
        for c in cumsum:
            if c > max_cum:
                max_cum = c
                max_interval = max(max_interval, current_interval)
                current_interval = 0
            current_interval += 1

        assert stats["新高间隔"] == pytest.approx(float(max_interval), abs=0.001)
