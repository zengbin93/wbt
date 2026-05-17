from __future__ import annotations

import wbt


def test_public_exports() -> None:
    """验证包公开导出的核心对象可用。"""
    assert wbt.WeightBacktest is not None
    assert wbt.daily_performance is not None


def test_migrated_czsc_exports() -> None:
    """5 个从 czsc 迁移过来的 API 都能从顶层导入。"""
    assert callable(wbt.cal_yearly_days)
    assert callable(wbt.rolling_daily_performance)
    assert callable(wbt.weights_simple_ensemble)
    assert callable(wbt.cal_trade_price)
    assert callable(wbt.log_strategy_info)
