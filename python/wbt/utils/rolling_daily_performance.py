"""Minimal Python adapter for the Rust ``_wbt.rolling_daily_performance`` symbol.

All business logic (sorting, NaN handling, yearly_days inference, rolling loop)
lives in Rust. This module only adapts pandas DataFrame ↔ Arrow IPC bytes.
"""

from __future__ import annotations

import pandas as pd

from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes
from wbt._wbt import rolling_daily_performance as _rolling_rs


def rolling_daily_performance(
    df: pd.DataFrame,
    ret_col: str,
    window: int = 252,
    min_periods: int = 100,
    yearly_days: int | None = None,
) -> pd.DataFrame:
    """计算滚动日收益的各项指标（业务逻辑在 Rust 内）。

    :param df: 日收益数据，columns 含 ['dt', ret_col]，或 index 为 datetime
    :param ret_col: 收益列名
    :param window: 滚动窗口（自然天数）
    :param min_periods: 预热跳过的行数（前 min_periods 个 edt 不出结果，从第 min_periods+1 行开始计算）
    :param yearly_days: 年度交易日数；None 时由 Rust 调 cal_yearly_days 推断
    """
    if isinstance(df.index, pd.DatetimeIndex):
        work = pd.DataFrame({"dt": df.index, ret_col: df[ret_col].values})
    else:
        work = df[["dt", ret_col]].copy()
        work["dt"] = pd.to_datetime(work["dt"])

    data = pandas_to_arrow_bytes(work)
    out_bytes = _rolling_rs(data, ret_col, window, min_periods, yearly_days)
    return arrow_bytes_to_pd_df(out_bytes)
