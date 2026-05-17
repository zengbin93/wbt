from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dts = pd.date_range("2022-01-01", periods=n, freq="D")
    rets = rng.normal(0.0005, 0.01, size=n)
    return pd.DataFrame({"dt": dts, "ret": rets})


def test_dt_column_input() -> None:
    from wbt.utils.rolling_daily_performance import rolling_daily_performance

    df = _sample_df()
    out = rolling_daily_performance(df, "ret", window=252, min_periods=100)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 300
    for col in ("sdt", "edt", "年化", "夏普", "最大回撤"):
        assert col in out.columns


def test_dt_index_input_matches_dt_column_input() -> None:
    from wbt.utils.rolling_daily_performance import rolling_daily_performance

    df = _sample_df()
    out_col = rolling_daily_performance(df, "ret", window=252, min_periods=100)
    out_idx = rolling_daily_performance(df.set_index("dt"), "ret", window=252, min_periods=100)
    assert len(out_idx) == 300
    # 两种输入路径应该产生完全一致的结果
    pd.testing.assert_frame_equal(out_idx, out_col)


def test_explicit_yearly_days_affects_output() -> None:
    from wbt.utils.rolling_daily_performance import rolling_daily_performance

    df = _sample_df()
    out_252 = rolling_daily_performance(df, "ret", window=252, min_periods=100, yearly_days=252)
    out_365 = rolling_daily_performance(df, "ret", window=252, min_periods=100, yearly_days=365)
    assert len(out_252) == 300
    # yearly_days 影响年化指标计算；两次结果在年化列上应该不同
    assert not out_252["年化"].equals(out_365["年化"])
