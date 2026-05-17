from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "w1": [1.0, -1.0, 0.5, 0.0],
            "w2": [-0.5, 1.0, 0.5, 0.0],
            "w3": [0.5, 0.0, -1.0, 1.0],
        }
    )


def test_mean_method() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = weights_simple_ensemble(_sample(), ["w1", "w2", "w3"], method="mean")
    assert np.allclose(df["weight"].tolist(), [1.0 / 3, 0.0, 0.0, 1.0 / 3])


def test_vote_method() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = weights_simple_ensemble(_sample(), ["w1", "w2", "w3"], method="vote")
    assert df["weight"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_sum_clip_method() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = weights_simple_ensemble(_sample(), ["w1", "w2", "w3"], method="sum_clip", clip_min=-0.5, clip_max=0.5)
    assert df["weight"].tolist() == [0.5, 0.0, 0.0, 0.5]


def test_only_long_zeroes_negatives() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = pd.DataFrame({"w1": [1.0, -1.0], "w2": [-2.0, 0.5]})
    out = weights_simple_ensemble(df, ["w1", "w2"], method="mean", only_long=True)
    assert out["weight"].tolist() == [0.0, 0.0]


def test_missing_col_asserts() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    with pytest.raises(AssertionError, match="缺失"):
        weights_simple_ensemble(_sample(), ["w1", "missing"])


def test_invalid_method_raises() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    with pytest.raises(ValueError, match="method 参数错误"):
        weights_simple_ensemble(_sample(), ["w1", "w2"], method="unknown")


def test_input_df_not_mutated() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = _sample()
    before_cols = list(df.columns)
    before_snapshot = df.copy()
    out = weights_simple_ensemble(df, ["w1", "w2", "w3"], method="mean")

    assert "weight" in out.columns
    assert "weight" not in df.columns, "weights_simple_ensemble 不应修改入参 df 的列"
    assert list(df.columns) == before_cols
    pd.testing.assert_frame_equal(df, before_snapshot)
    assert out is not df, "应返回新 DataFrame 而非原对象"
