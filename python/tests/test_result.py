from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest
from wbt.result import (
    BacktestResult,
    Curve,
    KeyTrades,
    MonthlyHeatmap,
    PairsDist,
    ReturnDist,
    SymbolReturns,
)

CURVE_KEYS = {"多空", "多头", "空头", "基准", "超额"}


@pytest.fixture
def result(wb: WeightBacktest) -> BacktestResult:
    return wb.to_result()


# ---------------------------------------------------------------------------
# B1 入口与类型
# ---------------------------------------------------------------------------
def test_to_result_returns_dto(result: BacktestResult) -> None:
    assert isinstance(result, BacktestResult)
    assert isinstance(result.start_date, str) and isinstance(result.end_date, str)
    assert result.symbol_count >= 1
    assert result.weight_type in {"ts", "cs"}
    assert result.yearly_days == 252


# ---------------------------------------------------------------------------
# B2 字段形状一致
# ---------------------------------------------------------------------------
def test_curves_keys_and_length(result: BacktestResult) -> None:
    assert set(result.curves.keys()) == CURVE_KEYS
    n = len(result.dates)
    for key, c in result.curves.items():
        assert isinstance(c, Curve)
        assert len(c.daily) == n, key
        assert len(c.cum) == n, key
        assert len(c.drawdown) == n, key


def test_dates_sorted_unique(result: BacktestResult) -> None:
    dates = result.dates
    assert np.issubdtype(dates.dtype, np.datetime64)
    assert len(np.unique(dates)) == len(dates)
    assert np.all(dates[:-1] <= dates[1:])


def test_year_starts_subset_and_first(result: BacktestResult) -> None:
    dates = pd.to_datetime(result.dates)
    ys = pd.to_datetime(result.year_starts)
    assert set(ys).issubset(set(dates))
    # 每个 year_start 是该年最早的日期
    for d in ys:
        same_year = dates[dates.year == d.year]
        assert d == same_year.min()
    assert set(ys.year) == set(dates.year.unique())


# ---------------------------------------------------------------------------
# B3 单位与口径
# ---------------------------------------------------------------------------
def test_cum_and_drawdown_definition(result: BacktestResult) -> None:
    for key, c in result.curves.items():
        np.testing.assert_allclose(c.cum, np.cumsum(c.daily), atol=1e-12, err_msg=key)
        expected_dd = c.cum - np.maximum.accumulate(c.cum)
        np.testing.assert_allclose(c.drawdown, expected_dd, atol=1e-12, err_msg=key)
        assert np.all(c.drawdown <= 1e-12), key


def test_return_dist_units(result: BacktestResult) -> None:
    rd = result.return_dist
    assert isinstance(rd, ReturnDist)
    total_daily = result.curves["多空"].daily
    expected = total_daily[~np.isnan(total_daily)] * 100
    np.testing.assert_allclose(np.sort(rd.values_pct), np.sort(expected), atol=1e-9)
    np.testing.assert_allclose(rd.mean_pct, float(np.mean(expected)), atol=1e-9)


def test_monthly_shape_and_winrates(result: BacktestResult) -> None:
    m = result.monthly
    assert isinstance(m, MonthlyHeatmap)
    assert m.months == list(range(1, 13))
    assert m.z.shape == (len(m.years), 12)
    assert m.text.shape == m.z.shape
    assert 0.0 <= m.month_win_rate <= 1.0
    assert 0.0 <= m.year_win_rate <= 1.0


# ---------------------------------------------------------------------------
# 分布 / 品种
# ---------------------------------------------------------------------------
def test_symbol_returns_sorted(result: BacktestResult) -> None:
    sr = result.symbol_returns
    assert isinstance(sr, SymbolReturns)
    assert len(sr.symbols) == len(sr.values)
    assert np.all(sr.values[:-1] <= sr.values[1:]), "按收益升序"


def test_pairs_dist_grouped(result: BacktestResult) -> None:
    pd_ = result.pairs_dist
    assert isinstance(pd_, PairsDist)
    assert set(pd_.pnl_pct.keys()) == set(pd_.holds.keys())
    for k, arr in pd_.pnl_pct.items():
        assert len(arr) == len(pd_.holds[k])


# ---------------------------------------------------------------------------
# C 按需计算（cached_property）
# ---------------------------------------------------------------------------
def test_curves_voladj_cached_and_keys(result: BacktestResult) -> None:
    va1 = result.curves_voladj
    va2 = result.curves_voladj
    assert va1 is va2, "cached_property 应缓存同一对象"
    assert set(va1.keys()) == CURVE_KEYS
    n = len(result.dates)
    for c in va1.values():
        assert len(c.cum) == n


def test_drawdowns_records(result: BacktestResult) -> None:
    dd1 = result.drawdowns
    assert result.drawdowns is dd1
    assert isinstance(dd1, list)
    if dd1:
        assert isinstance(dd1[0], dict)


def test_key_trades_structure(result: BacktestResult) -> None:
    kt = result.key_trades
    assert isinstance(kt, KeyTrades)
    assert isinstance(kt.best, dict)
    assert isinstance(kt.worst, dict)
    for year, rows in {**kt.best, **kt.worst}.items():
        assert isinstance(year, int)
        assert len(rows) <= 3
        for r in rows:
            assert hasattr(r, "symbol") and hasattr(r, "pnl") and hasattr(r, "count")


def test_verdict_has_is_good(result: BacktestResult) -> None:
    v = result.verdict
    assert "is_good" in v


def test_yearly_returns_aligned(result: BacktestResult) -> None:
    yr = result.yearly_returns
    n = len(yr.years)
    assert yr.abs_returns.shape == yr.alpha_returns.shape == (n,)
    assert yr.years == sorted(yr.years)


def test_rolling_series_aligned(result: BacktestResult) -> None:
    rm = result.rolling
    assert rm.window == 252
    n = rm.edt.size
    assert rm.sharpe.size == rm.annual_return.size == rm.annual_vol.size == n


def test_segment_comparison_keys(result: BacktestResult) -> None:
    sc = result.segment_comparison
    assert "全样本" in sc
    # 数据足够长时应有近 1 年
    assert sc["全样本"] is result.stats or "年化收益" in sc["全样本"]


# ---------------------------------------------------------------------------
# D 序列化
# ---------------------------------------------------------------------------
def test_to_dict_json_safe(result: BacktestResult) -> None:
    d_light = result.to_dict()
    json.dumps(d_light)  # 不应抛错
    assert "curves" in d_light and "stats" in d_light

    d_full = result.to_dict(full=True)
    s = json.dumps(d_full)  # 全字段也必须 JSON 安全
    assert "key_trades" in d_full
    assert "verdict" in d_full
    assert "drawdowns" in d_full
    assert "curves_voladj" in d_full
    assert isinstance(s, str)


def test_stats_by_side_keys(result: BacktestResult) -> None:
    assert {"多头", "空头", "基准", "超额"}.issubset(set(result.stats_by_side.keys()))


# ---------------------------------------------------------------------------
# K 数值回归
# ---------------------------------------------------------------------------
def test_curves_voladj_hits_target_vol(result: BacktestResult) -> None:
    """波动率归一后各曲线年化波动率 ≈ target_vol（非退化序列）。

    「超额」例外：它是 norm(多头) − norm(基准) 之差，波动率取决于两者相关性，
    不再等于 target_vol，故单独排除（其口径由 test_curves_voladj_excess_is_diff 校验）。
    """
    target = 0.20
    sqrt_yd = float(np.sqrt(result.yearly_days))
    for key, c in result.curves_voladj.items():
        if key == "超额" or c.daily.size <= 1:
            continue
        annual_vol = float(np.std(c.daily, ddof=1)) * sqrt_yd
        if annual_vol == 0:
            continue
        assert annual_vol == pytest.approx(target, rel=1e-6), key


def test_curves_voladj_excess_is_diff(result: BacktestResult) -> None:
    """归一超额 == 归一多头 − 归一基准（先各自归一化、再相减）。"""
    va = result.curves_voladj
    expected = va["多头"].daily - va["基准"].daily
    np.testing.assert_allclose(va["超额"].daily, expected, rtol=0, atol=1e-12)


def test_monthly_z_rows_match_yearly_totals(result: BacktestResult) -> None:
    """月度热力图每年（行）求和 == 当年日收益求和。"""
    total = result.curves["多空"].daily
    years = pd.DatetimeIndex(result.dates).year
    for i, y in enumerate(result.monthly.years):
        expected = float(total[years == y].sum())
        np.testing.assert_allclose(result.monthly.z[i].sum(), expected, atol=1e-9)


def test_cum_matches_daily_return_total(result: BacktestResult, wb: WeightBacktest) -> None:
    """多空累计末值 == daily_return['total'] 的 cumsum 末值（迁移前后口径一致）。"""
    expected = float(wb.daily_return["total"].sum())
    assert float(result.curves["多空"].cum[-1]) == pytest.approx(expected, rel=1e-9)
