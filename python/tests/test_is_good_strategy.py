"""is_good_strategy 端到端测试。

Rust 层（src/core/is_good_strategy.rs / src/core/backtest.rs）已覆盖算子细节、退化
路径、输入验证。本文件校验 Python 端薄转发的契约、字段类型、阈值响应。
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
    "alpha_degenerate",
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
    "alpha_degenerate",
    "cond_recent_return_passed",
    "cond_recent_dd_passed",
}


def test_history_mode_returns_expected_keys(wb: WeightBacktest):
    r = wb.is_good_strategy(mode="history")
    assert isinstance(r, dict)
    assert r["mode"] == "history"
    assert isinstance(r["is_good"], bool)
    assert set(r.keys()) == HISTORY_KEYS, (
        f"unexpected key set; missing: {HISTORY_KEYS - set(r.keys())}; extra: {set(r.keys()) - HISTORY_KEYS}"
    )
    assert isinstance(r["yearly_metrics"], list)
    for entry in r["yearly_metrics"]:
        assert {
            "year",
            "abs_return",
            "alpha_return",
            "days",
            "is_complete_year",
            "year_passed",
        } == set(entry.keys())
        # Nested bool fields must remain bool (not int) after PyO3 conversion.
        assert isinstance(entry["is_complete_year"], bool)
        assert isinstance(entry["year_passed"], bool)


def test_recent_mode_returns_expected_keys(wb: WeightBacktest):
    r = wb.is_good_strategy(mode="recent")
    assert isinstance(r, dict)
    assert r["mode"] == "recent"
    assert isinstance(r["is_good"], bool)
    assert isinstance(r["history_window_empty"], bool)
    assert set(r.keys()) == RECENT_KEYS, (
        f"unexpected key set; missing: {RECENT_KEYS - set(r.keys())}; extra: {set(r.keys()) - RECENT_KEYS}"
    )


def test_history_and_recent_key_sets_are_disjoint(wb: WeightBacktest):
    """文档约定：两个模式的 key 互斥（共享 mode/is_good/reason/alpha_degenerate）。"""
    history = set(wb.is_good_strategy(mode="history").keys())
    recent = set(wb.is_good_strategy(mode="recent").keys())
    shared = {"mode", "is_good", "reason", "alpha_degenerate"}
    history_only = history - shared
    recent_only = recent - shared
    assert history_only.isdisjoint(recent_only), (
        f"history-only ∩ recent-only must be empty; got history_only={history_only}, recent_only={recent_only}"
    )


def test_invalid_mode_raises(wb: WeightBacktest):
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(mode="xxx")
    msg = str(excinfo.value)
    # Now wrapped as WbtError::InvalidInput; Display prefix is "invalid input:".
    assert "invalid input" in msg or "invalid mode" in msg or "xxx" in msg, msg


def test_invalid_target_vol_raises(wb: WeightBacktest):
    """target_vol<=0 应在入口被拒绝。"""
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(target_vol=0.0)
    assert "target_vol" in str(excinfo.value) or "invalid input" in str(excinfo.value)


def test_invalid_max_dd_threshold_raises(wb: WeightBacktest):
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(max_dd_threshold=-0.1)
    assert "max_dd_threshold" in str(excinfo.value) or "invalid input" in str(excinfo.value)


def test_recent_zero_recent_days_raises(wb: WeightBacktest):
    """recent 模式下 recent_days=0 应抛错（不是 panic）。"""
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(mode="recent", recent_days=0)
    msg = str(excinfo.value)
    assert "recent_days" in msg or "invalid input" in msg


def test_threshold_extreme_changes_judgement(wb: WeightBacktest):
    """极小阈值让 cond_history_dd_passed=False；极大阈值让其 True。"""
    relaxed = wb.is_good_strategy(mode="history", max_dd_threshold=1.0)
    strict = wb.is_good_strategy(mode="history", max_dd_threshold=1e-9)
    assert strict["cond_history_dd_passed"] is False
    assert relaxed["cond_history_dd_passed"] is True


def test_default_parameters_work(wb: WeightBacktest):
    """不传任何参数应使用约定默认值。"""
    r = wb.is_good_strategy()
    assert r["mode"] == "history"
    assert isinstance(r["alpha_degenerate"], bool)


def test_min_history_days_zero_is_accepted(wb: WeightBacktest):
    """min_history_days=0 表示关闭 floor，调用应成功。"""
    r = wb.is_good_strategy(mode="recent", min_history_days=0)
    assert r["mode"] == "recent"


def test_dict_key_order_is_deterministic(wb: WeightBacktest):
    """hashmap_to_pydict 按 key 字母序插入，两次调用顺序一致。"""
    a = list(wb.is_good_strategy(mode="history").keys())
    b = list(wb.is_good_strategy(mode="history").keys())
    assert a == b == sorted(a), f"keys must be sorted: {a}"


def test_reason_is_string_and_empty_on_success(wb: WeightBacktest):
    """reason 永远是 str；is_good=True 时为空字符串。"""
    r = wb.is_good_strategy(mode="history", max_dd_threshold=1.0)
    assert isinstance(r["reason"], str)
    if r["is_good"]:
        assert r["reason"] == ""
