"""F03: reject invalid modes and preserve TS/CS aggregation across input paths."""

import numpy as np
import polars as pl
import pytest

from wbt import WeightBacktest, backtest
from wbt._df_convert import arrow_bytes_to_pd_df, polars_to_arrow_bytes
from wbt._wbt import PyWeightBacktest

PATHS = [
    "pandas",
    "polars",
    "lazy",
    "csv",
    "parquet",
    "feather",
    "arrow",
    "arrow_bytes",
    "ffi_csv",
    "ffi_parquet",
    "ffi_feather",
    "ffi_arrow",
    "backtest",
]


def input_frame(short=False):
    return pl.DataFrame(
        {
            "dt": [f"2024-01-0{day} 09:00:00" for day in range(1, 5)] * 2,
            "symbol": ["A"] * 4 + ["B"] * 4,
            "weight": [0.5] * 4 + ([-0.25] if short else [0.5]) * 4,
            "price": [100.0, 110.0, 99.0, 108.9, 200.0, 240.0, 216.0, 237.6],
        }
    )


def run_input(df, path, tmp_path, weight_type):
    kwargs = {} if weight_type is None else {"weight_type": weight_type}
    if path == "arrow_bytes":
        return PyWeightBacktest.from_arrow(polars_to_arrow_bytes(df), fee_rate=0, n_jobs=1, **kwargs)
    if path == "backtest":
        return backtest(df, fee_rate=0, n_jobs=1, **kwargs)
    if path == "pandas":
        data = df.to_pandas()
    elif path == "polars":
        data = df
    elif path == "lazy":
        data = df.lazy()
    else:
        extension = path.removeprefix("ffi_")
        data = tmp_path / f"input.{extension}"
        if extension == "csv":
            df.write_csv(data)
        elif extension == "parquet":
            df.write_parquet(data)
        else:
            df.write_ipc(data)
        if path.startswith("ffi_"):
            return PyWeightBacktest.from_file(str(data), fee_rate=0, n_jobs=1, **kwargs)
    return WeightBacktest(data, fee_rate=0, n_jobs=1, **kwargs)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("weight_type", ["INVALID", "TS", "CS", "", " ts", "cs ", "Ts"])
def test_invalid_weight_type_is_rejected(path, weight_type, tmp_path):
    with pytest.raises(ValueError, match="weight_type") as exc:
        run_input(input_frame(), path, tmp_path, weight_type)
    message = str(exc.value)
    assert f'"{weight_type}"' in message
    assert "'ts'" in message and "'cs'" in message


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("weight_type", [None, "ts", "cs"], ids=["default", "ts", "cs"])
@pytest.mark.parametrize("short", [False, True], ids=["all_long", "long_short"])
def test_valid_weight_type_returns_and_curves(path, weight_type, short, tmp_path):
    bt = run_input(input_frame(short), path, tmp_path, weight_type)
    # Hand-calculated: previous bar weight * price return, with no fees.
    a = np.array([0.05, -0.05, 0.05])
    b = np.array([-0.05, 0.025, -0.025] if short else [0.10, -0.05, 0.05])
    divisor = 1 if weight_type == "cs" else 2
    total = (a + b) / divisor
    inner = bt._inner if isinstance(bt, WeightBacktest) else bt
    dr = arrow_bytes_to_pd_df(inner.daily_return())
    np.testing.assert_allclose(dr["A"], a, atol=1e-12)
    np.testing.assert_allclose(dr["B"], b, atol=1e-12)
    np.testing.assert_allclose(dr["total"], total, atol=1e-12)
    assert inner.stats()["绝对收益"] == pytest.approx(round(float(total.sum()), 4))
    if isinstance(bt, WeightBacktest):
        result = bt.to_result()
        assert result.weight_type == (weight_type or "ts")
        curves = result.curves
        np.testing.assert_allclose(curves["多空"].daily, total, atol=1e-12)
        np.testing.assert_allclose(curves["多头"].daily, a / divisor if short else total, atol=1e-12)
        np.testing.assert_allclose(curves["空头"].daily, b / divisor if short else np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(curves["多空"].daily, curves["多头"].daily + curves["空头"].daily, atol=1e-12)
