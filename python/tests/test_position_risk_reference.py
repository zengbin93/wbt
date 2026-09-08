import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from position_risk_oracle import dense_oracle


@pytest.mark.parametrize("seed", range(12))
def test_python_tree_against_independent_dense_oracle(seed):
    reference = load_reference()
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "dt": pd.Timestamp("2026-09-08") + pd.to_timedelta(rng.integers(0, 15, 60), unit="h"),
            "symbol": rng.choice(["甲", "B", "C", "D", "E"], 60),
            "weight": rng.choice([np.nan, 0.0, -2.5, 1.25, 0.1, -0.3], 60),
        }
    )
    actual = reference["calculate_prepared"](frame)
    expected = dense_oracle(frame)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-12)


def load_reference():
    path = Path(__file__).parents[1] / "scripts" / "position_risk_reference.py"
    assert path.exists(), "same-algorithm Python reference is missing"
    return runpy.run_path(str(path))


def test_python_tree_matches_hand_calculation():
    reference = load_reference()
    frame = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2026-09-08 09:30", "2026-09-08 09:00"]),
            "symbol": ["IF", "RB"],
            "weight": [0.3, -0.2],
        }
    )
    result = reference["calculate_prepared"](frame)
    np.testing.assert_allclose(
        result.iloc[:, 1:],
        [
            [0.2, 0, 0.2, -0.2, 0.2, 0.04, 0],
            [0.5, 0.3, 0.2, 0.1, 0.3, 0.13, 1.5],
        ],
    )
