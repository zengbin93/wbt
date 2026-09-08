"""F06/F07: full-sample normalization must agree across stats, curves and verdicts."""

import numpy as np
import pytest

from wbt import WeightBacktest, daily_performance
from wbt._wbt import _normalize_returns, _vol_adjusted_alpha


@pytest.mark.parametrize("yearly_days", [12, 252, 365])
@pytest.mark.parametrize("target_vol", [0.1, 0.2, 0.45])
@pytest.mark.parametrize("weight_type", ["ts", "cs"])
def test_alpha_curves_match_stats_and_verdicts(sample_dfw, yearly_days, target_vol, weight_type):
    wb = WeightBacktest(sample_dfw, yearly_days=yearly_days, weight_type=weight_type, n_jobs=1)
    result = wb.to_result(target_vol=target_vol)
    long = result.curves["多头"].daily
    bench = result.curves["基准"].daily
    expected = target_vol / np.sqrt(yearly_days) * (long / long.std(ddof=0) - bench / bench.std(ddof=0))
    np.testing.assert_allclose(result.curves_voladj["多头超额"].daily, expected, atol=1e-14)
    assert result.verdict == wb.is_good_strategy(mode="history", target_vol=target_vol)
    assert result.verdict_recent == wb.is_good_strategy(mode="recent", target_vol=target_vol)
    assert result.verdict_recent["recent_alpha_return"] == pytest.approx(expected.sum(), abs=1e-12)
    assert result.yearly_returns.alpha_returns[0] == pytest.approx(expected.sum(), abs=1e-12)
    # Legacy stats property keeps its documented fixed 20% target.
    dp = daily_performance(expected * (0.2 / target_vol), yearly_days=yearly_days)
    for stat, metric in (
        ("绝对收益", "绝对收益"),
        ("年化收益", "年化"),
        ("最大回撤", "最大回撤"),
        ("夏普比率", "夏普"),
    ):
        assert wb.long_alpha_stats[stat] == pytest.approx(dp[metric], abs=1e-4)


@pytest.mark.parametrize("bad", [[], [1.0], [0.0, 0.0], [1e-15, -1e-15], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_private_array_adapters_preserve_degeneracy(bad):
    assert _normalize_returns(bad, 252, 0.2) is None
    assert _vol_adjusted_alpha(bad, [0.01, -0.02], 252, 0.2) is None
    assert _vol_adjusted_alpha([0.01, -0.02], bad, 252, 0.2) is None


@pytest.mark.parametrize("constant_price", [False, True])
def test_degenerate_alpha_has_explicit_presentation(sample_dfw, constant_price):
    if constant_price:
        sample_dfw["price"] = 100.0
    else:
        sample_dfw["weight"] = 0.0
    wb = WeightBacktest(sample_dfw, n_jobs=1)
    result = wb.to_result(target_vol=0.4)
    assert all(v == 0 for v in wb.long_alpha_stats.values())
    for verdict in (result.verdict, result.verdict_recent):
        assert verdict["alpha_degenerate"] is True
        assert verdict["is_good"] is False
    assert result.verdict["history_alpha_max_drawdown"] is None
    assert result.verdict_recent["recent_alpha_return"] is None
    assert np.isnan(result.curves_voladj["多头超额"].daily).all()
    assert all(m["alpha_return"] is None and m["alpha_max_drawdown"] is None for m in result.verdict["yearly_metrics"])
    assert np.isnan(result.yearly_returns.alpha_returns).all()
    exported = result.to_dict(full=True)["curves_voladj"]["多头超额"]
    assert all(v is None for v in exported["daily"])


def test_identical_nondegenerate_inputs_are_valid_zero_alpha():
    assert _vol_adjusted_alpha([0.01, -0.02], [0.01, -0.02], 252, 0.2) == [0.0, 0.0]
