import pandas as pd
import pytest
from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes


class TestRoundTrip:
    def test_basic_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3], "c": ["x", "y", "z"]})
        result = arrow_bytes_to_pd_df(pandas_to_arrow_bytes(df))
        pd.testing.assert_frame_equal(result, df)

    def test_datetime_roundtrip(self):
        df = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-01-01", "2024-01-02"]), "v": [1.0, 2.0]}
        )
        result = arrow_bytes_to_pd_df(pandas_to_arrow_bytes(df))
        assert "datetime" in str(result["dt"].dtype)
        assert len(result) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        result = arrow_bytes_to_pd_df(pandas_to_arrow_bytes(df))
        assert len(result) == 0
        assert "a" in result.columns

    def test_series_input(self):
        s = pd.Series([1.0, 2.0, 3.0], name="values")
        # Series gets converted to a single-column DataFrame by Arrow
        data = pandas_to_arrow_bytes(s.to_frame())
        result = arrow_bytes_to_pd_df(data)
        assert len(result) == 3
