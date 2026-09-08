"""Chronological position exposure using the Rust aggregation tree."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from wbt._df_convert import arrow_bytes_to_pd_df
from wbt._wbt import calculate_position_risk as _calculate_position_risk


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    result = frame.loc[:, ["dt", "symbol", "weight"]].copy()
    result["dt"] = pd.to_datetime(result["dt"])
    if result["dt"].isna().any():
        raise ValueError("dt must not contain nulls")
    if result["symbol"].isna().any():
        raise ValueError("symbol must not contain nulls")
    if pd.api.types.infer_dtype(result["symbol"].to_numpy()) not in ("string", "empty"):
        raise TypeError("symbol values must be strings")
    weights = result["weight"]
    # to_numeric accepts temporal dtypes and casting complex values drops their
    # imaginary parts. Check object/mixed values too, before either conversion.
    if weights.dtype.kind in "mMc" or (
        not pd.api.types.is_numeric_dtype(weights.dtype)
        and any(
            isinstance(value, (date, timedelta, np.datetime64, np.timedelta64, complex, np.complexfloating))
            for value in weights
        )
    ):
        raise TypeError("weight values must be real numbers, not datetime, timedelta or complex")
    result["weight"] = pd.to_numeric(result["weight"], errors="raise").to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(result["weight"].to_numpy()).any():
        raise ValueError("weight must not contain infinity")
    result["symbol"] = result["symbol"].astype(pd.StringDtype())
    return result


def _to_arrow(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def calculate_position_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Return seven exposure metrics on the sorted union of input timestamps.

    Missing weights carry the previous position (initially zero). Duplicate
    time/symbol rows use the last nonmissing weight in input order. Positions
    persist overnight; zero explicitly closes a position; leverage is not clipped.
    The long/short ratio is NaN when short exposure is zero. Input is not mutated.
    Datetime precision and timezone are preserved through Arrow IPC.
    """
    return arrow_bytes_to_pd_df(_calculate_position_risk(_to_arrow(_prepare_frame(frame))))
