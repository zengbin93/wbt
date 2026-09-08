"""F01: every input path must share the Rust input contract."""

import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from wbt import WeightBacktest
from wbt._df_convert import arrow_bytes_to_pd_df, polars_to_arrow_bytes
from wbt._wbt import PyWeightBacktest

PATHS = ["pandas", "polars", "lazy", "csv", "parquet", "arrow", "arrow_bytes"]


def input_frame():
    return pl.DataFrame(
        {
            "dt": [f"2024-01-0{day} 09:00:00" for day in range(1, 6)] * 2,
            "symbol": ["A"] * 5 + ["B"] * 5,
            "weight": [1, 0, -1, 1, 0] * 2,
            "price": [100, 102, 99, 103, 101, 200, 199, 203, 201, 205],
        }
    )


def run_input(df, path, tmp_path):
    if path == "arrow_bytes":
        return PyWeightBacktest.from_arrow(polars_to_arrow_bytes(df), n_jobs=1)
    if path == "pandas":
        data = df.to_pandas()
    elif path == "polars":
        data = df
    elif path == "lazy":
        data = df.lazy()
    else:
        data = tmp_path / f"input.{path}"
        if path == "csv":
            df.write_csv(data)
        elif path == "parquet":
            df.write_parquet(data)
        else:
            df.write_ipc(data)
    return WeightBacktest(data, n_jobs=1)._inner


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("dtype", [pl.Int64, pl.Int32, pl.UInt64, pl.Float32, pl.Float64])
def test_numeric_inputs_match_float_reference(path, dtype, tmp_path):
    df = input_frame()
    if dtype == pl.UInt64:
        df = df.with_columns(pl.col("weight").abs())
    reference = run_input(df.with_columns(pl.col("weight", "price").cast(pl.Float64)), "polars", tmp_path)
    actual = run_input(df.with_columns(pl.col("weight", "price").cast(dtype)), path, tmp_path)
    assert actual.stats() == reference.stats()
    assert actual.symbol_dict() == ["A", "B"]
    for method in ["daily_return", "dailys", "pairs"]:
        assert_frame_equal(
            arrow_bytes_to_pd_df(getattr(actual, method)()),
            arrow_bytes_to_pd_df(getattr(reference, method)()),
        )


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("column", ["dt", "symbol", "weight", "price"])
@pytest.mark.parametrize("invalid", ["missing", "null", "all_null", "wrong_type"])
def test_invalid_required_columns_raise_catchable_value_error(path, column, invalid, tmp_path):
    df = input_frame()
    if invalid == "missing":
        df = df.drop(column)
    elif invalid == "null":
        df = df.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col(column)).alias(column))
    elif invalid == "all_null":
        df = df.with_columns(pl.lit(None).alias(column))
    else:
        value = 42 if column == "symbol" else "invalid"
        df = df.with_columns(pl.lit(value).alias(column))
    # PanicException derives from BaseException and would escape this handler.
    try:
        run_input(df, path, tmp_path)
    except Exception as exc:
        assert isinstance(exc, ValueError), repr(exc)
        assert column in str(exc)
    else:
        pytest.fail(f"{path} accepted {invalid} {column}")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("symbol", ["", "   "])
def test_empty_symbol_is_rejected(path, symbol, tmp_path):
    df = input_frame().with_columns(pl.lit(symbol).alias("symbol"))
    with pytest.raises(ValueError, match="symbol"):
        run_input(df, path, tmp_path)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("column", ["weight", "price"])
def test_boolean_is_not_a_numeric_input(path, column, tmp_path):
    df = input_frame().with_columns(pl.lit(True).alias(column))
    with pytest.raises(ValueError, match=column):
        run_input(df, path, tmp_path)
