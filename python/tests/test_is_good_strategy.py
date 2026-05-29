"""is_good_strategy 端到端测试。

Rust 层（src/core/is_good_strategy.rs / src/core/backtest.rs）已覆盖算子细节与
"未跑 backtest 报错"分支。本文件只校验 Python 端薄转发的契约：返回 dict、字段齐全、
mode 字符串解析、阈值翻转。
"""

from __future__ import annotations

import pytest

from wbt import WeightBacktest

HISTORY_KEYS = {
    "mode",
    "is_good",
    "reason",
    "yearly_metrics",
    "complete_year_count",
    "history_alpha_max_drawdown",
    "cond_yearly_passed",
    "cond_history_dd_passed",
}

RECENT_KEYS = {
    "mode",
    "is_good",
    "reason",
    "recent_start_date",
    "recent_end_date",
    "recent_actual_days",
    "recent_abs_return",
    "recent_alpha_return",
    "recent_alpha_max_drawdown",
    "history_alpha_max_drawdown_excl_recent",
    "history_window_empty",
    "cond_recent_return_passed",
    "cond_recent_dd_passed",
}


def test_history_mode_returns_expected_keys(wb: WeightBacktest):
    r = wb.is_good_strategy(mode="history")
    assert isinstance(r, dict)
    assert r["mode"] == "history"
    assert isinstance(r["is_good"], bool)
    assert HISTORY_KEYS.issubset(r.keys()), f"missing keys: {HISTORY_KEYS - set(r.keys())}"
    assert isinstance(r["yearly_metrics"], list)
    for entry in r["yearly_metrics"]:
        assert {"year", "abs_return", "alpha_return", "days", "is_complete_year", "year_passed"}.issubset(entry.keys())


def test_recent_mode_returns_expected_keys(wb: WeightBacktest):
    r = wb.is_good_strategy(mode="recent")
    assert isinstance(r, dict)
    assert r["mode"] == "recent"
    assert isinstance(r["is_good"], bool)
    assert isinstance(r["history_window_empty"], bool)
    assert RECENT_KEYS.issubset(r.keys()), f"missing keys: {RECENT_KEYS - set(r.keys())}"


def test_invalid_mode_raises(wb: WeightBacktest):
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(mode="xxx")
    assert "xxx" in str(excinfo.value) or "invalid mode" in str(excinfo.value)


def test_threshold_extreme_changes_judgement(wb: WeightBacktest):
    """极小阈值必然让大多数策略 is_good=False，与默认 0.20 的判定可能不同。"""
    relaxed = wb.is_good_strategy(mode="history", max_dd_threshold=1.0)
    strict = wb.is_good_strategy(mode="history", max_dd_threshold=1e-9)
    # strict 几乎不可能通过条件 B（任何非零回撤都超阈）
    assert strict["cond_history_dd_passed"] is False
    # relaxed 一定通过条件 B（任何小于 100% 的回撤都通过）
    assert relaxed["cond_history_dd_passed"] is True


def test_default_parameters_work(wb: WeightBacktest):
    """不传任何参数应使用方案中的默认值，调用应成功。"""
    r = wb.is_good_strategy()
    assert r["mode"] == "history"
