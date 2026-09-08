"""F02: hand-calculated expectations for cancelling and constant returns."""

import math

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest, daily_performance, rolling_daily_performance, top_drawdowns

KEYS = [
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

# Rows are fixed mathematical expectations, not another implementation of the kernel.
CASES = [
    pytest.param([0.1, -0.1], [0, 0, 0, 0.1, 0, 0.5, 1, 0, 1.5875, 0, 1, 1, 1, 0.5, 0.063, -25.2, 0.0008], id="cancel"),
    pytest.param([-0.1, 0.1], [0, 0, 0, 0, 0, 0.5, 1, 0, 1.5875, 0, 1, 1, 0, 1, 0, 25.2, 0], id="cancel_no_drawdown"),
    pytest.param(
        [0.01] * 3, [0.03, 2.52, 0, 0, 10, 1, 5, 5, 0, 0, 1, 0.3333, 0, 1, 0, 2.52, 0], id="constant_positive"
    ),
    pytest.param(
        [-0.01] * 3,
        [-0.03, -2.52, 0, 0.02, -10, 0, 0, -1, 0, 0, 1, 1, 2, 0.3333, 0, -2.52, 0.0016],
        id="constant_negative",
    ),
    pytest.param([0.0] * 3, [0] * 15 + [None, 0], id="zero"),
    pytest.param([], [0] * 15 + [None, 0], id="empty"),
    pytest.param([0.01], [0.01, 2.52, 0, 0, 10, 1, 5, 5, 0, 0, 1, 1, 0, 1, 0, None, 0], id="single_positive"),
    pytest.param([-0.01], [-0.01, -2.52, 0, 0, -10, 0, 0, -1, 0, 0, 1, 1, 0, 1, 0, None, 0], id="single_negative"),
]


@pytest.mark.parametrize("returns,expected", CASES)
def test_degenerate_metrics_match_known_values(returns, expected):
    result = daily_performance(np.array(returns, dtype=float), yearly_days=252)
    assert list(result) == KEYS
    for key, value in zip(KEYS, expected, strict=True):
        if value is None:
            assert result[key] is None
        else:
            assert math.isfinite(result[key]), key
            assert result[key] == pytest.approx(value, abs=1e-4), key


@pytest.mark.parametrize("returns,expected", CASES)
def test_rolling_metrics_preserve_degenerate_windows(returns, expected):
    df = pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=len(returns)), "ret": returns})
    result = rolling_daily_performance(df, "ret", window=10, min_periods=0, yearly_days=252)
    if not returns:
        assert result.empty
        return
    last = result.iloc[-1]
    for key, value in zip(KEYS, expected, strict=True):
        if value is None:
            assert pd.isna(last[key])
        else:
            assert last[key] == pytest.approx(value, abs=1e-4), key


@pytest.mark.parametrize(
    "returns,absolute,drawdown",
    [([0.1, -0.1], 0, 0.1), ([0.01] * 3, 0.03, 0), ([-0.01] * 3, -0.03, 0.02), ([0.0] * 3, 0, 0)],
)
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_backtest_stats_curves_and_drawdowns_agree(returns, absolute, drawdown, side):
    # Prices produce the requested strategy returns for either a long or short position.
    prices = [100.0]
    for value in returns:
        prices.append(prices[-1] * (1 + side * value))
    df = pd.DataFrame(
        {"dt": pd.date_range("2024-01-01", periods=len(prices)), "symbol": "A", "weight": side, "price": prices}
    )
    bt = WeightBacktest(df, fee_rate=0, yearly_days=252)
    curve = bt.to_result().curves["多空"]
    np.testing.assert_allclose(curve.daily, returns, atol=1e-12)
    assert curve.cum[-1] == pytest.approx(absolute, abs=1e-12)
    assert -curve.drawdown.min() == pytest.approx(drawdown, abs=1e-12)
    performance = daily_performance(curve.daily)
    assert performance["回撤风险"] == pytest.approx(0.063 if len(returns) == 2 else 0, abs=1e-4)
    direction = "多头" if side == 1 else "空头"
    for stats in [bt.stats, bt.long_stats if side == 1 else bt.short_stats, bt.segment_stats(kind=direction)]:
        assert stats["绝对收益"] == pytest.approx(absolute, abs=1e-4)
        assert stats["最大回撤"] == pytest.approx(drawdown, abs=1e-4)
        assert stats["年化波动率"] == pytest.approx(1.5875 if len(returns) == 2 else 0, abs=1e-4)
    windows = top_drawdowns(pd.Series(returns, index=pd.date_range("2024-01-01", periods=len(returns))))
    if drawdown:
        assert -windows["净值回撤"].min() == pytest.approx(drawdown)
    else:
        assert windows.empty


def test_tiny_nonzero_volatility_does_not_divide_by_rounded_zero():
    # Unrounded volatility is 1e-6 * sqrt(252); displayed volatility rounds to zero.
    result = daily_performance(np.array([1e-6, -1e-6]))
    assert result["年化波动率"] == 0
    assert result["回撤风险"] == pytest.approx(0.063)
    assert result["非零覆盖"] == 1
