from __future__ import annotations

import numpy as np
import pandas as pd


def _bars(symbol: str = "A", n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "dt": pd.date_range("2024-01-01", periods=n, freq="min"),
            "open": np.arange(n, dtype=float) + 100,
            "close": np.arange(n, dtype=float) + 101,
            "vol": np.arange(n, dtype=float) + 1,
        }
    )


def test_columns_present_for_default_windows() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    out = cal_trade_price(_bars())
    expected = {"TP_CLOSE", "TP_NEXT_OPEN", "TP_NEXT_CLOSE"}
    for w in (5, 10, 15, 20, 30, 60):
        expected.add(f"TP_TWAP{w}")
        expected.add(f"TP_VWAP{w}")
    assert expected.issubset(set(out.columns))


def test_multi_symbol_concat_order_preserved() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    bars = pd.concat([_bars("A"), _bars("B")], ignore_index=True)
    out = cal_trade_price(bars)
    assert set(out["symbol"].unique()) == {"A", "B"}


def test_custom_windows() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    out = cal_trade_price(_bars(), windows=(3, 7))
    assert "TP_TWAP3" in out.columns
    assert "TP_VWAP7" in out.columns
    assert "TP_TWAP5" not in out.columns


def test_tp_close_equals_close() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    bars = _bars()
    out = cal_trade_price(bars)
    assert (out["TP_CLOSE"] == bars["close"]).all()


def test_twap_numeric_and_tail_fill() -> None:
    """验证 TWAP 计算数值正确，且尾部 NaN 用 close 填充。"""
    from wbt.utils.cal_trade_price import cal_trade_price

    bars = _bars(n=10)  # close = [101, 102, ..., 110]
    out = cal_trade_price(bars, windows=(3,))

    # 第 0 行的 TP_TWAP3 = mean(close[2..4]) = mean([103, 104, 105]) = 103.0
    assert out["TP_TWAP3"].iloc[0] == 103.0

    # 尾部 3 行 TWAP 计算结果为 NaN，应被 fillna 替换为对应 close
    for idx in (7, 8, 9):
        assert out["TP_TWAP3"].iloc[idx] == out["close"].iloc[idx]
