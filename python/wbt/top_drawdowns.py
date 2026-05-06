"""Public Python wrapper around the Rust ``_wbt.top_drawdowns`` symbol.

Accepts a daily-return ``pandas.Series`` (date-indexed) and returns a
``pandas.DataFrame`` with the rs-czsc-compatible top-N drawdown schema:

    回撤开始 / 回撤结束 / 回撤修复 / 净值回撤 / 回撤天数 / 恢复天数 / 新高间隔
"""

from __future__ import annotations

import pandas as pd

from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes
from wbt._wbt import top_drawdowns as _top_drawdowns


def top_drawdowns(returns: pd.Series, top: int = 10) -> pd.DataFrame:
    """Identify the top-N drawdown windows in a daily-return series.

    :param returns: pandas Series indexed by date, holding daily
        period-over-period returns (not cumulative). Index dtype must
        be convertible to datetime.
    :param top: maximum number of drawdowns to return; the actual row
        count may be smaller if the underwater curve flattens out
        before ``top`` drawdowns are extracted.
    :return: DataFrame with columns
        ``回撤开始 / 回撤结束 / 回撤修复 / 净值回撤 / 回撤天数 / 恢复天数 / 新高间隔``.
    """
    df_in = pd.DataFrame({"date": returns.index, "returns": returns.values})
    data = pandas_to_arrow_bytes(df_in)
    return arrow_bytes_to_pd_df(_top_drawdowns(data, top))
