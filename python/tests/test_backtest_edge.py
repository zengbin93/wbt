import pandas as pd
import pytest
from wbt import WeightBacktest


def make_dfw(n_bars=10, symbols=None, weight_fn=None):
    """Helper to construct test DataFrames"""
    if symbols is None:
        symbols = ["A"]
    if weight_fn is None:
        weight_fn = lambda i, s: 0.5

    # Use string dates to avoid Arrow microsecond serialization issues
    base = pd.Timestamp("2024-01-01 09:30:00")
    dates = [(base + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n_bars)]
    rows = []
    for sym in symbols:
        for i, dt in enumerate(dates):
            rows.append({
                "dt": dt,
                "symbol": sym,
                "weight": float(weight_fn(i, sym)),
                "price": 100.0 + i * 0.5,
            })
    return pd.DataFrame(rows)


class TestMissingColumns:
    def test_missing_weight(self):
        df = pd.DataFrame({
            "dt": ["2024-01-01 09:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            "symbol": ["A"] * 3,
            "price": [100.0, 101.0, 102.0],
        })
        with pytest.raises(KeyError):
            WeightBacktest(df)

    def test_missing_price(self):
        df = pd.DataFrame({
            "dt": ["2024-01-01 09:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            "symbol": ["A"] * 3,
            "weight": [0.5, 0.5, 0.5],
        })
        with pytest.raises(KeyError):
            WeightBacktest(df)


class TestSingleSymbol:
    def test_single_symbol_works(self):
        dfw = make_dfw(n_bars=20, symbols=["ONLY"])
        wb = WeightBacktest(dfw, digits=2)
        assert wb.stats is not None
        assert len(wb.symbol_dict) == 1


class TestZeroWeights:
    def test_all_zero_weights(self):
        dfw = make_dfw(n_bars=20, symbols=["A"], weight_fn=lambda i, s: 0.0)
        wb = WeightBacktest(dfw, digits=2)
        stats = wb.stats
        assert stats["绝对收益"] == 0.0


class TestWeightTypes:
    def test_ts_vs_cs_differ(self):
        dfw_ts = make_dfw(
            n_bars=20, symbols=["A", "B"],
            weight_fn=lambda i, s: 0.3 if s == "A" else -0.2,
        )
        dfw_cs = dfw_ts.copy()
        wb_ts = WeightBacktest(dfw_ts, weight_type="ts")
        wb_cs = WeightBacktest(dfw_cs, weight_type="cs")
        assert wb_ts.stats["绝对收益"] != wb_cs.stats["绝对收益"]


class TestNullValues:
    def test_null_raises(self):
        df = pd.DataFrame({
            "dt": ["2024-01-01 09:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            "symbol": ["A"] * 3,
            "weight": [0.5, None, 0.5],
            "price": [100.0, 101.0, 102.0],
        })
        with pytest.raises(ValueError, match="空值"):
            WeightBacktest(df)
