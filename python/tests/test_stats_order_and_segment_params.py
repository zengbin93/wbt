"""Test that all stats output functions preserve canonical field order.

Design reference: docs/desgin.md — stats output fields must follow a fixed order
for human readability.
"""
from __future__ import annotations

import pytest

from wbt import WeightBacktest

# Canonical field order from design doc
CANONICAL_ORDER = [
    "绝对收益",
    "年化收益",
    "夏普比率",
    "卡玛比率",
    "新高占比",
    "单笔盈亏比",
    "单笔收益",
    "日胜率",
    "周胜率",
    "月胜率",
    "季胜率",
    "年胜率",
    "最大回撤",
    "年化波动率",
    "下行波动率",
    "新高间隔",
    "交易次数",
    "年化交易次数",
    "持仓K线数",
    "交易胜率",
    "多头占比",
    "空头占比",
    "品种数量",
    "开始日期",
    "结束日期",
]


def _check_canonical_order(keys: list[str]) -> None:
    """Assert that keys follow the canonical relative order."""
    prev_index = -1
    for key in keys:
        assert key in CANONICAL_ORDER, f"Unexpected key '{key}' not in canonical order"
        idx = CANONICAL_ORDER.index(key)
        assert idx > prev_index, (
            f"Key '{key}' (canonical index {idx}) appeared after index {prev_index}, "
            f"violating canonical order.\nGot keys: {keys}"
        )
        prev_index = idx


class TestStatsFieldOrder:
    """Verify all stats outputs preserve canonical field order."""

    def test_stats_order(self, wb: WeightBacktest) -> None:
        _check_canonical_order(list(wb.stats.keys()))

    def test_long_stats_order(self, wb: WeightBacktest) -> None:
        _check_canonical_order(list(wb.long_stats.keys()))

    def test_short_stats_order(self, wb: WeightBacktest) -> None:
        _check_canonical_order(list(wb.short_stats.keys()))

    def test_segment_stats_order(self, wb: WeightBacktest) -> None:
        _check_canonical_order(list(wb.segment_stats().keys()))

    def test_long_alpha_stats_order(self, wb: WeightBacktest) -> None:
        _check_canonical_order(list(wb.long_alpha_stats.keys()))


class TestSegmentStatsStrParams:
    """Verify segment_stats accepts str/timestamp dates, not just int."""

    def test_sdt_edt_str_format(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats(sdt="2024-01-01", edt="2024-01-15")
        assert isinstance(stats, dict)
        assert "年化收益" in stats

    def test_sdt_edt_str_same_result_as_int(self, wb: WeightBacktest) -> None:
        """str date should produce identical result to equivalent int."""
        stats_str = wb.segment_stats(sdt="2024-01-01", edt="2024-01-15")
        stats_int = wb.segment_stats(sdt=20240101, edt=20240115)
        assert stats_str == stats_int

    def test_sdt_only(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats(sdt="2024-01-05")
        assert isinstance(stats, dict)
        assert "年化收益" in stats

    def test_edt_only(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats(edt="2024-01-10")
        assert isinstance(stats, dict)
        assert "年化收益" in stats

    def test_sdt_timestamp(self, wb: WeightBacktest) -> None:
        import pandas as pd

        stats = wb.segment_stats(
            sdt=pd.Timestamp("2024-01-01"),
            edt=pd.Timestamp("2024-01-15"),
        )
        assert isinstance(stats, dict)
        assert "年化收益" in stats

    def test_sdt_none(self, wb: WeightBacktest) -> None:
        stats = wb.segment_stats(sdt=None, edt=None)
        assert isinstance(stats, dict)
        assert "年化收益" in stats

    def test_sdt_str_yyyymmdd(self, wb: WeightBacktest) -> None:
        """支持 '20240101' 这种紧凑格式。"""
        stats = wb.segment_stats(sdt="20240101", edt="20240115")
        assert isinstance(stats, dict)
        assert "年化收益" in stats
