from __future__ import annotations

import wbt


def test_public_exports() -> None:
    """验证包公开导出的核心对象可用。"""
    assert wbt.WeightBacktest is not None
    assert wbt.daily_performance is not None
