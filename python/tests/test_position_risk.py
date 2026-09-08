import numpy as np
import pandas as pd
import pytest
from position_risk_oracle import dense_oracle
from test_position_risk_reference import load_reference

import wbt


@pytest.fixture(params=["rust", "python"])
def calculator(request):
    if request.param == "rust":
        return wbt.calculate_position_risk
    return load_reference()["calculate_position_risk"]


@pytest.mark.parametrize("seed", range(20))
def test_matches_independent_dense_history_oracle(calculator, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "dt": pd.Timestamp("2026-09-08") + pd.to_timedelta(rng.integers(0, 20, 90), unit="h"),
            "symbol": rng.choice(["甲", "B", "C", "D", "E"], 90),
            "weight": rng.choice([np.nan, 0.0, -2.5, 1.25, 0.1, -0.3], 90),
        }
    )
    pd.testing.assert_frame_equal(
        calculator(frame),
        dense_oracle(frame),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_duplicates_missing_values_and_explicit_close(calculator):
    frame = pd.DataFrame(
        {
            "dt": pd.to_datetime([1, 1, 1, 1, 2, 2, 3, 3, 4]),
            "symbol": ["A", "B", "A", "A", "A", "B", "A", "B", "C"],
            "weight": [2.0, -3.0, 1.0, np.nan, np.nan, 0.0, 0.0, np.nan, np.nan],
        }
    )
    result = calculator(frame)
    np.testing.assert_allclose(result.total_risk, [4, 1, 0, 0])
    np.testing.assert_allclose(result.max_single_risk, [3, 1, 0, 0])
    np.testing.assert_allclose(result.herfindahl, [10, 1, 0, 0])
    assert result.long_short_ratio.iloc[0] == pytest.approx(1 / 3)
    assert result.long_short_ratio.iloc[1:].isna().all()


@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai", "America/New_York"])
def test_nanosecond_timestamps_and_timezone_survive(calculator, timezone):
    times = pd.date_range("2026-09-08", periods=3, freq="ns", tz=timezone)
    frame = pd.DataFrame({"dt": times, "symbol": ["A"] * 3, "weight": [1, -1, 0]})
    result = calculator(frame)
    pd.testing.assert_series_equal(result.dt, frame.dt)
    np.testing.assert_allclose(result.net_exposure, [1, -1, 0])


def test_empty_has_typed_schema(calculator):
    frame = pd.DataFrame(
        {
            "dt": pd.Series([], dtype="datetime64[ns]"),
            "symbol": pd.Series([], dtype="str"),
            "weight": pd.Series([], dtype="float64"),
        }
    )
    result = calculator(frame)
    assert result.shape == (0, 8)
    assert str(result.dt.dtype) == "datetime64[ns]"
    assert (result.dtypes.iloc[1:] == "float64").all()


def test_closing_large_position_does_not_erase_small_position(calculator):
    frame = pd.DataFrame(
        {
            "dt": pd.to_datetime([1, 1, 2, 3]),
            "symbol": ["A", "B", "A", "B"],
            "weight": [1e16, 1.0, 0.0, 0.0],
        }
    )
    result = calculator(frame)
    np.testing.assert_allclose(result.total_risk.iloc[1:], [1.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(result.herfindahl.iloc[1:], [1.0, 0.0], rtol=0, atol=0)


def test_nullable_weight_means_missing_observation(calculator):
    frame = pd.DataFrame(
        {
            "dt": pd.to_datetime([1, 2]),
            "symbol": ["A", "A"],
            "weight": [1.0, pd.NA],
        }
    )
    np.testing.assert_allclose(calculator(frame).total_risk, [1.0, 1.0])


@pytest.mark.parametrize(
    "column,value",
    [
        ("dt", None),
        ("dt", "not-a-date"),
        ("symbol", None),
        ("symbol", 123),
        ("weight", float("inf")),
        ("weight", float("-inf")),
        ("weight", "bad"),
    ],
)
def test_invalid_input_is_rejected(calculator, column, value):
    row = {"dt": "2026-09-08", "symbol": "A", "weight": 1.0}
    row[column] = value
    with pytest.raises((ValueError, TypeError)):
        calculator(pd.DataFrame([row]))


def test_native_profile_has_identical_output():
    from wbt import _wbt
    from wbt.position_risk import _prepare_frame, _to_arrow

    assert hasattr(_wbt, "_profile_position_risk"), "native phase profiler is missing"
    data = _to_arrow(
        _prepare_frame(
            pd.DataFrame(
                {
                    "dt": ["2026-09-08"],
                    "symbol": ["A"],
                    "weight": [0.25],
                }
            )
        )
    )
    result, phases, is_debug = _wbt._profile_position_risk(data)
    from wbt._df_convert import arrow_bytes_to_pd_df

    pd.testing.assert_frame_equal(
        arrow_bytes_to_pd_df(result),
        arrow_bytes_to_pd_df(_wbt.calculate_position_risk(data)),
    )
    assert len(phases) == 3
    assert all(duration >= 0 for duration in phases)
    assert isinstance(is_debug, bool)


def test_native_entry_rejects_nonnumeric_weight():
    from wbt import _wbt
    from wbt.position_risk import _to_arrow

    frame = pd.DataFrame({"dt": pd.to_datetime([1]), "symbol": ["A"], "weight": ["invalid"]})
    with pytest.raises(ValueError, match="conversion|cast"):
        _wbt.calculate_position_risk(_to_arrow(frame))


def test_chronological_exposure_matches_hand_calculation():
    assert hasattr(wbt, "calculate_position_risk"), "public position-risk API is missing"
    frame = pd.DataFrame(
        [
            ("2026-09-08 09:30", "IF", 0.3),
            ("2026-09-08 09:00", "RB", -0.2),
            ("2026-09-08 09:30", "CU", 0.1),
            ("2026-09-09 09:00", "IF", 0.0),
        ],
        columns=["dt", "symbol", "weight"],
    )
    before = frame.copy(deep=True)
    result = wbt.calculate_position_risk(frame)
    assert list(result.columns) == [
        "dt",
        "total_risk",
        "long_risk",
        "short_risk",
        "net_exposure",
        "max_single_risk",
        "herfindahl",
        "long_short_ratio",
    ]
    assert result.dt.tolist() == list(
        pd.to_datetime(
            [
                "2026-09-08 09:00",
                "2026-09-08 09:30",
                "2026-09-09 09:00",
            ]
        )
    )
    np.testing.assert_allclose(
        result.iloc[:, 1:],
        [
            [0.2, 0.0, 0.2, -0.2, 0.2, 0.04, 0.0],
            [0.6, 0.4, 0.2, 0.2, 0.3, 0.14, 2.0],
            [0.3, 0.1, 0.2, -0.1, 0.2, 0.05, 0.5],
        ],
        rtol=1e-12,
        atol=1e-14,
    )
    pd.testing.assert_frame_equal(frame, before)
