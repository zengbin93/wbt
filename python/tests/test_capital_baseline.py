import numpy as np
import pandas as pd
import pytest

from wbt import daily_performance, top_drawdowns
from wbt.result import _build_curve


@pytest.mark.parametrize(
    "daily, expected",
    [
        ([-0.10, 0.02, 0.01], [-0.10, -0.08, -0.07]),
        ([-0.125, 0.125, 0.25], [-0.125, 0.0, 0.0]),
        ([0.125, 0.25, 0.125], [0.0, 0.0, 0.0]),
    ],
)
def test_capital_baseline_across_metrics_curves_and_details(daily, expected):
    values = np.array(daily)
    curve = _build_curve(values)
    np.testing.assert_allclose(curve.drawdown, expected)
    stats = daily_performance(values)
    assert stats["最大回撤"] == pytest.approx(-min(expected))
    details = top_drawdowns(pd.Series(values, index=pd.date_range("2024-01-01", periods=3)))
    if min(expected) < 0:
        assert details.iloc[0]["净值回撤"] == pytest.approx(min(expected))
        assert len(details) == 1
    else:
        assert details.empty
    if sum(daily) < 0:
        assert stats["绝对收益"] == pytest.approx(-0.07)
        assert stats["卡玛"] < 0
        assert stats["新高占比"] == 0
        assert stats["新高间隔"] == 3
