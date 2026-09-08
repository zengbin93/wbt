"""Independent, deliberately dense specification oracle (no update tree)."""

import math

import pandas as pd


def dense_oracle(frame):
    rows = []
    for timestamp in sorted(frame.dt.unique()):
        positions = []
        for symbol in frame.symbol.unique():
            history = frame[(frame.symbol == symbol) & (frame.dt <= timestamp)]
            history = history.sort_values("dt", kind="stable").dropna(subset=["weight"])
            positions.append(float(history.weight.iloc[-1]) if len(history) else 0.0)
        long_risk = math.fsum(weight for weight in positions if weight > 0)
        short_risk = math.fsum(-weight for weight in positions if weight < 0)
        rows.append(
            [
                timestamp,
                math.fsum(abs(weight) for weight in positions),
                long_risk,
                short_risk,
                math.fsum(positions),
                max(map(abs, positions), default=0.0),
                math.fsum(weight * weight for weight in positions),
                long_risk / short_risk if short_risk else math.nan,
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "dt",
            "total_risk",
            "long_risk",
            "short_risk",
            "net_exposure",
            "max_single_risk",
            "herfindahl",
            "long_short_ratio",
        ],
    )
