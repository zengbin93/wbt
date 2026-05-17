"""Minimal Python adapter for the Rust ``_wbt.cal_yearly_days`` symbol.

All business rules (span check, 252 fallback, warning) live in Rust.
This module only converts Python date-like iterables to unix-ms i64 lists.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from wbt._wbt import cal_yearly_days as _cal_yearly_days_rs


def cal_yearly_days(dts: Iterable) -> int:
    """计算年度交易日数量。

    业务规则（跨度判定、252 兜底、warning）由 Rust 层完成；
    本函数仅负责把 Python 端的日期序列转成 Rust 接收的 unix-ms 列表。
    """
    ts_ms = pd.to_datetime(pd.Series(list(dts))).astype("datetime64[ms]").astype("int64").tolist()
    return int(_cal_yearly_days_rs(ts_ms))
