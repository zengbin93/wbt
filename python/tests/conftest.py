from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest


@pytest.fixture
def sample_dfw() -> pd.DataFrame:
    """2 symbols x 15 bars inline DataFrame with fixed random seed for reproducibility."""
    rng = np.random.default_rng(42)
    n_days = 15
    rows: list[dict] = []
    for sym in ["SYM_A", "SYM_B"]:
        for d in range(n_days):
            for h in range(4):
                dt = f"2024-01-{d + 1:02d} {9 + h:02d}:30:00"
                w = round(rng.uniform(-0.5, 0.5), 2)
                p = 100.0 + rng.normal(0, 2)
                rows.append({"dt": dt, "symbol": sym, "weight": w, "price": round(p, 4)})
    return pd.DataFrame(rows)


@pytest.fixture
def wb(sample_dfw: pd.DataFrame) -> WeightBacktest:
    """Pre-built WeightBacktest instance for reuse across tests."""
    return WeightBacktest(sample_dfw, digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252)
