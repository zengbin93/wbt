from __future__ import annotations

import numpy as np
import pytest

from wbt import daily_performance

EXPECTED_KEYS = [
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


class TestDailyPerformance:
    """验证 daily_performance 函数的计算正确性。"""

    def test_known_values(self) -> None:
        """手算验证: returns = [0.01, -0.005, 0.02]"""
        returns = np.array([0.01, -0.005, 0.02])
        result = daily_performance(returns, yearly_days=252)

        assert set(EXPECTED_KEYS) == set(result.keys())
        assert result["绝对收益"] == pytest.approx(0.025)
        assert result["年化"] == pytest.approx(2.1)
        assert result["夏普"] == pytest.approx(10.0)  # capped from 12.876
        assert result["最大回撤"] == pytest.approx(0.005)
        assert result["卡玛"] == pytest.approx(20.0)  # capped from 420
        assert result["日胜率"] == pytest.approx(0.6667, abs=1e-4)
        assert result["日盈亏比"] == pytest.approx(3.0)
        assert result["年化波动率"] == pytest.approx(0.1631, abs=1e-4)
        assert result["非零覆盖"] == pytest.approx(1.0)
        assert result["盈亏平衡点"] == pytest.approx(0.6667, abs=1e-4)

    def test_all_zero(self) -> None:
        """全零收益应返回零值统计。"""
        returns = np.zeros(100)
        result = daily_performance(returns)
        assert result["绝对收益"] == pytest.approx(0.0)
        assert result["年化"] == pytest.approx(0.0)
        assert result["夏普"] == pytest.approx(0.0)
        assert result["最大回撤"] == pytest.approx(0.0)

    def test_empty_returns_default(self) -> None:
        """空数组应返回默认零值。"""
        returns = np.array([], dtype=np.float64)
        result = daily_performance(returns)
        assert result["绝对收益"] == pytest.approx(0.0)

    def test_yearly_days_proportional(self) -> None:
        """年化收益率应与 yearly_days 成正比，比例约为 365/252。"""
        returns = np.array([0.002, -0.001] * 50)
        r252 = daily_performance(returns, yearly_days=252)
        r365 = daily_performance(returns, yearly_days=365)
        assert r252["年化"] != 0.0
        ratio = r365["年化"] / r252["年化"]
        assert ratio == pytest.approx(365 / 252, rel=0.01)

    def test_negative_returns_known(self) -> None:
        """负收益场景: [-0.01, -0.02, 0.005]，最大回撤 = 0.02。"""
        returns = np.array([-0.01, -0.02, 0.005])
        result = daily_performance(returns, yearly_days=252)
        assert result["绝对收益"] == pytest.approx(-0.025)
        assert result["日胜率"] == pytest.approx(0.3333, abs=1e-4)  # 1 win out of 3
        assert result["最大回撤"] == pytest.approx(0.02)
        assert result["年化"] < 0
