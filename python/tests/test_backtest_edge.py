from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from wbt import WeightBacktest


def make_dfw(
    n_bars: int = 10,
    symbols: list[str] | None = None,
    weight_fn: Callable[[int, str], float] | None = None,
) -> pd.DataFrame:
    """构造测试用持仓权重 DataFrame。

    Args:
        n_bars: K线数量
        symbols: 品种列表，默认为 ["A"]
        weight_fn: 权重生成函数 (i, symbol) -> float，默认固定 0.5

    Returns:
        包含 dt, symbol, weight, price 列的 DataFrame
    """
    if symbols is None:
        symbols = ["A"]

    def _default_weight(i: int, s: str) -> float:
        return 0.5

    if weight_fn is None:
        weight_fn = _default_weight

    base = pd.Timestamp("2024-01-01 09:30:00")
    dates = [(base + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n_bars)]
    rows: list[dict] = []
    for sym in symbols:
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "dt": dt,
                    "symbol": sym,
                    "weight": float(weight_fn(i, sym)),
                    "price": 100.0 + i * 0.5,
                }
            )
    return pd.DataFrame(rows)


class TestMissingColumns:
    """验证缺少必要列时的错误处理。"""

    def test_missing_weight(self) -> None:
        df = pd.DataFrame(
            {
                "dt": ["2024-01-01 09:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
                "symbol": ["A"] * 3,
                "price": [100.0, 101.0, 102.0],
            }
        )
        with pytest.raises(KeyError):
            WeightBacktest(df)

    def test_missing_price(self) -> None:
        df = pd.DataFrame(
            {
                "dt": ["2024-01-01 09:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
                "symbol": ["A"] * 3,
                "weight": [0.5, 0.5, 0.5],
            }
        )
        with pytest.raises(KeyError):
            WeightBacktest(df)


class TestSingleSymbol:
    """验证单品种场景。"""

    def test_single_symbol_works(self) -> None:
        dfw = make_dfw(n_bars=20, symbols=["ONLY"])
        wb = WeightBacktest(dfw, digits=2)
        assert wb.stats is not None
        assert len(wb.symbol_dict) == 1


class TestZeroWeights:
    """验证全零权重场景。"""

    def test_all_zero_weights(self) -> None:
        dfw = make_dfw(n_bars=20, symbols=["A"], weight_fn=lambda i, s: 0.0)
        wb = WeightBacktest(dfw, digits=2)
        assert wb.stats["绝对收益"] == 0.0


class TestWeightTypes:
    """验证时序/截面策略权重类型差异。"""

    def test_ts_vs_cs_differ(self) -> None:
        dfw_ts = make_dfw(
            n_bars=20,
            symbols=["A", "B"],
            weight_fn=lambda i, s: 0.3 if s == "A" else -0.2,
        )
        dfw_cs = dfw_ts.copy()
        wb_ts = WeightBacktest(dfw_ts, weight_type="ts")
        wb_cs = WeightBacktest(dfw_cs, weight_type="cs")
        assert wb_ts.stats["绝对收益"] != wb_cs.stats["绝对收益"]


class TestNullValues:
    """验证空值数据的错误处理。"""

    def test_null_raises(self) -> None:
        df = pd.DataFrame(
            {
                "dt": ["2024-01-01 09:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
                "symbol": ["A"] * 3,
                "weight": [0.5, None, 0.5],
                "price": [100.0, 101.0, 102.0],
            }
        )
        with pytest.raises(ValueError, match="空值"):
            WeightBacktest(df)
