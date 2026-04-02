import numpy as np
from wbt import daily_performance


EXPECTED_KEYS = [
    "绝对收益", "年化", "夏普", "最大回撤", "卡玛", "日胜率", "日盈亏比",
    "日赢面", "年化波动率", "下行波动率", "非零覆盖", "盈亏平衡点",
    "新高间隔", "新高占比", "回撤风险", "回归年度回报率", "长度调整平均最大回撤",
]


class TestDailyPerformance:
    def test_known_values(self):
        """Hand-calculated: returns = [0.01, -0.005, 0.02]"""
        returns = np.array([0.01, -0.005, 0.02])
        result = daily_performance(returns, yearly_days=252)
        assert set(EXPECTED_KEYS) == set(result.keys())
        assert result["绝对收益"] == 0.025
        assert result["年化"] == 2.1
        assert result["夏普"] == 10.0  # capped from 12.876
        assert result["最大回撤"] == 0.005
        assert result["卡玛"] == 20.0  # capped from 420
        assert result["日胜率"] == 0.6667
        assert result["日盈亏比"] == 3.0
        assert result["年化波动率"] == 0.1631
        assert result["非零覆盖"] == 1.0
        assert result["盈亏平衡点"] == 0.6667

    def test_all_zero(self):
        returns = np.zeros(100)
        result = daily_performance(returns)
        assert result["绝对收益"] == 0.0
        assert result["年化"] == 0.0
        assert result["夏普"] == 0.0
        assert result["最大回撤"] == 0.0

    def test_empty_returns_default(self):
        returns = np.array([], dtype=np.float64)
        result = daily_performance(returns)
        assert set(EXPECTED_KEYS) == set(result.keys())
        assert result["绝对收益"] == 0.0
        assert result["年化"] == 0.0
        assert result["夏普"] == 0.0
        assert result["最大回撤"] == 0.0
        assert result["回归年度回报率"] is None

    def test_yearly_days_proportional(self):
        """annual_returns = mean * yearly_days, so ratio should be 365/252"""
        returns = np.array([0.002, -0.001] * 50)
        r252 = daily_performance(returns, yearly_days=252)
        r365 = daily_performance(returns, yearly_days=365)
        if r252["年化"] != 0:
            ratio = r365["年化"] / r252["年化"]
            assert abs(ratio - 365 / 252) < 0.01

    def test_negative_returns_known(self):
        """returns = [-0.01, -0.02, 0.005]
        max_drawdown = 0.02 (underwater peak at -0.01, valley at -0.03)
        """
        returns = np.array([-0.01, -0.02, 0.005])
        result = daily_performance(returns, yearly_days=252)
        assert result["绝对收益"] == -0.025
        assert result["日胜率"] == 0.3333  # 1 win out of 3
        assert result["最大回撤"] == 0.02   # underwater: [0, -0.02, -0.015]
        assert result["年化"] < 0
