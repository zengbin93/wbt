"""按 symbol 分组生成 TWAP / VWAP 等下根 K 线交易价表。

🟡 Experimental：windows 默认值可能调整。
"""

from __future__ import annotations

import pandas as pd


def cal_trade_price(df: pd.DataFrame, digits: int | None = None, **kwargs) -> pd.DataFrame:
    """计算给定品种基础周期 K 线数据的交易价格表。

    :param df: 基础周期 K 线，必须包含 symbol/dt/open/close/vol 列
    :param digits: 保留小数位数；None 时按各品种 close 列推断
    :param kwargs:
        - windows: TWAP/VWAP 窗口列表，默认 (5, 10, 15, 20, 30, 60)
        - copy: 是否复制输入，默认 True
    """
    assert "symbol" in df.columns, "数据中必须包含 symbol 列"
    for col in ("dt", "open", "close", "vol"):
        assert col in df.columns, f"数据中必须包含 {col} 列"

    if kwargs.get("copy", True):
        df = df.copy()

    symbols = df["symbol"].unique().tolist()
    windows = kwargs.get("windows", (5, 10, 15, 20, 30, 60))

    dfs: list[pd.DataFrame] = []
    for symbol in symbols:
        sub = df[df["symbol"] == symbol].sort_values("dt").reset_index(drop=True)

        sym_digits = digits
        if sym_digits is None:
            sym_digits = sub["close"].astype(str).str.split(".").str[1].str.len().max()
            if pd.isna(sym_digits):
                sym_digits = 0
            sym_digits = int(sym_digits)

        sub["TP_CLOSE"] = sub["close"]
        sub["TP_NEXT_OPEN"] = sub["open"].shift(-1)
        sub["TP_NEXT_CLOSE"] = sub["close"].shift(-1)
        price_cols = ["TP_CLOSE", "TP_NEXT_OPEN", "TP_NEXT_CLOSE"]

        sub["_vcp"] = sub["vol"] * sub["close"]
        for t in windows:
            sub[f"TP_TWAP{t}"] = sub["close"].rolling(t).mean().shift(-t)
            vol_sum = sub["vol"].rolling(t).sum()
            vcp_sum = sub["_vcp"].rolling(t).sum()
            sub[f"TP_VWAP{t}"] = (vcp_sum / vol_sum).shift(-t)
            price_cols.extend([f"TP_TWAP{t}", f"TP_VWAP{t}"])
        sub.drop(columns=["_vcp"], inplace=True)

        for pc in price_cols:
            sub[pc] = sub[pc].fillna(sub["close"])
        sub[price_cols] = sub[price_cols].round(sym_digits)
        dfs.append(sub)

    return pd.concat(dfs, ignore_index=True)
