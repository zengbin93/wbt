"""Benchmark reference: the same stable-sort/aggregation-tree algorithm as Rust.

This is deliberately not the dense-grid Python implementation from the document.
It is a reproducible CPython algorithm comparison, not a fastest-pandas claim.
"""

from __future__ import annotations

import math

import pandas as pd

RISK_COLUMNS = [
    "total_risk",
    "long_risk",
    "short_risk",
    "net_exposure",
    "max_single_risk",
    "herfindahl",
    "long_short_ratio",
]


def calculate_prepared(frame: pd.DataFrame) -> pd.DataFrame:
    """Typed DataFrame in, DataFrame out; includes extraction, sort and tree work."""
    symbol_ids: dict[str, int] = {}
    events = []
    for timestamp, symbol, weight in zip(
        frame["dt"].array.asi8.tolist(), frame["symbol"].tolist(), frame["weight"].tolist(), strict=True
    ):
        symbol_id = symbol_ids.setdefault(symbol, len(symbol_ids))
        events.append((timestamp, symbol_id, weight))
    events.sort(key=lambda event: event[0])
    leaves = 1 << (max(1, len(symbol_ids)) - 1).bit_length()
    tree = [(0.0,) * 6] * (leaves * 2)
    output_times = []
    output_values = []
    for index, (timestamp, symbol_id, weight) in enumerate(events):
        if not math.isnan(weight):
            node = leaves + symbol_id
            tree[node] = (
                abs(weight),
                max(weight, 0.0),
                max(-weight, 0.0),
                weight,
                abs(weight),
                weight * weight,
            )
            while node > 1:
                node //= 2
                left, right = tree[2 * node], tree[2 * node + 1]
                tree[node] = (
                    left[0] + right[0],
                    left[1] + right[1],
                    left[2] + right[2],
                    left[3] + right[3],
                    max(left[4], right[4]),
                    left[5] + right[5],
                )
        if index + 1 == len(events) or events[index + 1][0] != timestamp:
            root = tree[1]
            ratio = math.nan if root[2] == 0.0 else root[1] / root[2]
            output_times.append(timestamp)
            output_values.append((*root, ratio))
    result = pd.DataFrame(output_values, columns=RISK_COLUMNS, dtype="float64")
    result.insert(0, "dt", pd.Series(output_times, dtype="int64").astype(frame["dt"].dtype))
    return result


def calculate_position_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Same public input normalization/output contract, without crossing into Rust."""
    from wbt.position_risk import _prepare_frame

    return calculate_prepared(_prepare_frame(frame))
