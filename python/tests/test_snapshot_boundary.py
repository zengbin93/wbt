import json

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest


@pytest.mark.parametrize(
    "name,value", [("fee_rate", 0.1), ("digits", 6), ("weight_type", "cs"), ("yearly_days", 365), ("symbols", [])]
)
def test_effective_configuration_is_read_only(wb, name, value):
    original = getattr(wb, name)
    with pytest.raises(AttributeError):
        setattr(wb, name, value)
    assert getattr(wb, name) == original


def test_configuration_rejects_invalid_weight_type(sample_dfw):
    with pytest.raises(ValueError, match="invalid weight_type"):
        WeightBacktest(sample_dfw, weight_type="invalid")


@pytest.mark.parametrize("weight_type", ["ts", "cs"])
def test_configuration_reports_effective_fee_default(sample_dfw, weight_type):
    wb = WeightBacktest(sample_dfw, weight_type=weight_type, fee_rate=None)
    assert wb.weight_type == weight_type
    assert wb.fee_rate == 0.0002
    assert not wb.long_daily_return.empty


def test_input_and_table_mutation_do_not_change_snapshot(wb, sample_dfw):
    before = wb.to_result().to_dict(full=True)
    sample_dfw["price"] = 1.0
    detached_input = wb.dfw
    detached_input["price"] = 1.0
    wb.symbols.clear()
    for name in ["dailys", "daily_return", "pairs", "alpha", "aggregated_pairs"]:
        frame = getattr(wb, name)
        frame.drop(frame.index, inplace=True)
    assert wb.to_result().to_dict(full=True) == before


def test_snapshot_properties_keep_docstrings(wb):
    """快照属性的 docstring 不因 cached_property 包装而丢失（help/IDE 可见）。"""
    result = wb.to_result()
    for name in [
        "verdict",
        "verdict_recent",
        "drawdowns",
        "key_trades",
        "yearly_returns",
        "rolling",
        "segment_comparison",
        "curves_voladj",
    ]:
        doc = getattr(type(result), name).__doc__
        assert doc and doc.strip(), f"快照属性 {name} 的 docstring 丢失"


def test_nested_snapshot_is_read_only_and_export_is_detached(wb):
    result = wb.to_result()
    before = result.to_dict(full=True)
    with pytest.raises(AttributeError):
        result.yearly_days = 365
    with pytest.raises(AttributeError):
        result.verdict = {}
    with pytest.raises(AttributeError):
        del result.curves
    with pytest.raises(TypeError):
        result.stats["年化收益"] = 99
    with pytest.raises(TypeError):
        result.curves["多空"] = result.curves["多头"]
    with pytest.raises(ValueError):
        result.curves["多空"].daily[0] = 99
    with pytest.raises(ValueError):
        result.curves_voladj["多空"].daily.setflags(write=True)
    with pytest.raises(TypeError):
        result.verdict["is_good"] = False
    exported = result.to_dict(full=True)
    exported["monthly"]["years"].clear()
    exported["stats"].clear()
    assert result.to_dict(full=True) == before
    json.dumps(before, allow_nan=False)


def test_new_backtest_recomputes_without_changing_old_result(wb, sample_dfw):
    result = wb.to_result()
    before = result.to_dict(full=True)
    changed = WeightBacktest(sample_dfw, fee_rate=0.1, weight_type="cs", yearly_days=365)
    assert changed.to_result().yearly_days == 365
    assert not np.array_equal(changed.daily_return["total"], wb.daily_return["total"])
    assert result.to_dict(full=True) == before
    pd.testing.assert_frame_equal(wb.dfw, sample_dfw.astype({"weight": float, "price": float}))
