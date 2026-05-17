from __future__ import annotations

import logging

import pandas as pd
import pytest


def test_short_span_returns_252_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.cal_yearly_days import cal_yearly_days

    dts = pd.date_range("2024-01-01", periods=50, freq="D").tolist()
    with caplog.at_level(logging.WARNING):
        result = cal_yearly_days(dts)
    assert result == 252
    assert any("时间跨度小于一年" in rec.message for rec in caplog.records)


def test_multi_year_returns_clamped_count() -> None:
    from wbt.utils.cal_yearly_days import cal_yearly_days

    dts = pd.date_range("2020-01-01", "2023-12-31", freq="B").tolist()
    result = cal_yearly_days(dts)
    assert 240 <= result <= 365


def test_accepts_series_and_index() -> None:
    from wbt.utils.cal_yearly_days import cal_yearly_days

    dr = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    assert cal_yearly_days(dr) == cal_yearly_days(pd.Series(dr))


def test_empty_raises() -> None:
    from wbt.utils.cal_yearly_days import cal_yearly_days

    with pytest.raises(Exception):
        cal_yearly_days([])
