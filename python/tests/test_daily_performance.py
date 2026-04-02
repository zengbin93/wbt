import numpy as np
from wbt import daily_performance


EXPECTED_KEYS = [
    "绝对收益", "年化", "夏普", "最大回撤", "卡玛", "日胜率", "日盈亏比",
    "日赢面", "年化波动率", "下行波动率", "非零覆盖", "盈亏平衡点",
    "新高间隔", "新高占比", "回撤风险", "回归年度回报率", "长度调整平均最大回撤",
]


class TestDailyPerformance:
    def test_normal_returns(self):
        returns = np.random.default_rng(42).normal(0.001, 0.02, 252)
        result = daily_performance(returns)
        assert isinstance(result, dict)
        assert set(EXPECTED_KEYS) == set(result.keys())

    def test_all_zero(self):
        returns = np.zeros(100)
        result = daily_performance(returns)
        assert result["绝对收益"] == 0.0
        assert result["年化"] == 0.0

    def test_empty_returns_default(self):
        returns = np.array([], dtype=np.float64)
        result = daily_performance(returns)
        assert result["绝对收益"] == 0.0

    def test_yearly_days_affects_result(self):
        returns = np.random.default_rng(42).normal(0.001, 0.02, 100)
        r252 = daily_performance(returns, yearly_days=252)
        r365 = daily_performance(returns, yearly_days=365)
        assert r252["年化"] != r365["年化"]

    def test_positive_returns(self):
        # Use varying positive returns to avoid zero-std edge case
        returns = np.linspace(0.001, 0.003, 252)
        result = daily_performance(returns, yearly_days=252)
        assert result["绝对收益"] > 0
        assert result["年化"] > 0
        assert result["最大回撤"] == 0.0
        assert result["日胜率"] == 1.0
