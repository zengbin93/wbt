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
