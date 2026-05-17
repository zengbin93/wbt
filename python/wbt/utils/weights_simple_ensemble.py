"""多策略权重的朴素集成。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def weights_simple_ensemble(
    df: pd.DataFrame,
    weight_cols: list,
    method: str = "mean",
    only_long: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """用朴素的方法集成多个策略的权重。

    method: mean / vote / sum_clip；kwargs.clip_min / clip_max 仅对 sum_clip 生效。
    """
    method = method.lower()
    missing = set(weight_cols) - set(df.columns)
    assert not missing, f"数据中不包含全部权重列，缺失：{missing}"
    assert "weight" not in df.columns, "数据中已经包含 weight 列，请先删除"

    if method == "mean":
        df["weight"] = df[weight_cols].mean(axis=1).fillna(0)
    elif method == "vote":
        df["weight"] = np.sign(df[weight_cols].sum(axis=1)).fillna(0)
    elif method == "sum_clip":
        clip_min = kwargs.get("clip_min", -1)
        clip_max = kwargs.get("clip_max", 1)
        df["weight"] = df[weight_cols].sum(axis=1).clip(clip_min, clip_max).fillna(0)
    else:
        raise ValueError("method 参数错误，可选 mean / vote / sum_clip")

    if only_long:
        df["weight"] = np.where(df["weight"] > 0, df["weight"], 0)
    return df
