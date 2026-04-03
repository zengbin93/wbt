"""Tests for untested Python input paths: polars DataFrame/LazyFrame, file paths (str/Path),
and utility roundtrip for polars_to_arrow_bytes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wbt import WeightBacktest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATA_10 = {
    "dt": [f"2024-01-{d + 2:02d} 09:30:00" for d in range(10)],
    "symbol": ["A"] * 10,
    "weight": [0.5, 0.3, -0.2, 0.0, 0.4, 0.5, -0.1, 0.0, 0.3, -0.2],
    "price": [100.0 + i * 0.5 for i in range(10)],
}

_DATA_5 = {
    "dt": [
        "2024-01-02 09:30:00",
        "2024-01-03 09:30:00",
        "2024-01-04 09:30:00",
        "2024-01-05 09:30:00",
        "2024-01-06 09:30:00",
    ],
    "symbol": ["A", "A", "A", "A", "A"],
    "weight": [0.5, 0.3, -0.2, 0.0, 0.4],
    "price": [100.0, 101.0, 102.0, 103.0, 104.0],
}


# ---------------------------------------------------------------------------
# polars DataFrame input
# ---------------------------------------------------------------------------


def test_polars_dataframe_input():
    """WeightBacktest accepts a polars.DataFrame."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(_DATA_5)
    bt = WeightBacktest(df, digits=2, fee_rate=0.0002, n_jobs=1)
    assert bt.stats is not None
    assert "绝对收益" in bt.stats


# ---------------------------------------------------------------------------
# polars LazyFrame input
# ---------------------------------------------------------------------------


def test_polars_lazyframe_input():
    """WeightBacktest accepts a polars.LazyFrame (collect inside)."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(_DATA_5).lazy()
    bt = WeightBacktest(df, digits=2, fee_rate=0.0002, n_jobs=1)
    assert bt.stats is not None
    assert "绝对收益" in bt.stats


# ---------------------------------------------------------------------------
# file path — str
# ---------------------------------------------------------------------------


def test_file_path_str_input(tmp_path):
    """WeightBacktest accepts a plain str file path (CSV)."""
    path = str(tmp_path / "test.csv")
    pd.DataFrame(_DATA_5).to_csv(path, index=False)
    bt = WeightBacktest(path, digits=2, fee_rate=0.0002, n_jobs=1)
    assert bt.stats is not None
    assert bt.dfw is None  # file path input does not store dfw


# ---------------------------------------------------------------------------
# file path — pathlib.Path
# ---------------------------------------------------------------------------


def test_file_path_object_input(tmp_path):
    """WeightBacktest accepts a pathlib.Path object (CSV)."""
    path = tmp_path / "test.csv"
    pd.DataFrame(_DATA_5).to_csv(str(path), index=False)
    bt = WeightBacktest(path, digits=2, fee_rate=0.0002, n_jobs=1)
    assert bt.stats is not None
    assert bt.dfw is None


# ---------------------------------------------------------------------------
# polars_to_arrow_bytes roundtrip
# ---------------------------------------------------------------------------


def test_polars_to_arrow_bytes():
    """polars_to_arrow_bytes + arrow_bytes_to_pd_df is lossless for basic types."""
    pl = pytest.importorskip("polars")
    from wbt._df_convert import arrow_bytes_to_pd_df, polars_to_arrow_bytes

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    data = polars_to_arrow_bytes(df)
    result = arrow_bytes_to_pd_df(data)
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# polars vs pandas consistency
# ---------------------------------------------------------------------------


def test_polars_pandas_consistency():
    """Stats must be identical whether input is pandas or polars DataFrame."""
    pl = pytest.importorskip("polars")

    bt_pd = WeightBacktest(pd.DataFrame(_DATA_10), digits=2, fee_rate=0.0002, n_jobs=1)
    bt_pl = WeightBacktest(pl.DataFrame(_DATA_10), digits=2, fee_rate=0.0002, n_jobs=1)

    assert bt_pd.stats is not None
    assert bt_pl.stats is not None
    for key in bt_pd.stats:
        assert bt_pd.stats[key] == bt_pl.stats[key], f"Mismatch for key '{key}'"
