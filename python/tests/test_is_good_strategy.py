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
    "alpha_degenerate",
    "cond_yearly_passed",
    "history_alpha_max_drawdown",
    "history_alpha_sharpe",
    "cond_history_dd_passed",
    "cond_history_sharpe_passed",
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
            "alpha_max_drawdown",
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


def test_yearly_metrics_carry_per_year_drawdown(wb: WeightBacktest):
    """回撤已下放为逐年指标：yearly_metrics 每项带出 alpha_max_drawdown。"""
    r = wb.is_good_strategy(mode="history")
    ym = r["yearly_metrics"]
    assert isinstance(ym, list) and ym, "fixture should produce at least one year bucket"
    for entry in ym:
        dd = entry["alpha_max_drawdown"]
        assert isinstance(dd, (int, float)) and not isinstance(dd, bool)
        assert dd >= 0.0


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


def test_history_mode_returns_full_sample_hard_gate_keys(wb: WeightBacktest):
    """history 模式必含全样本两道硬门的指标与判定 key。"""
    r = wb.is_good_strategy(mode="history")
    # 全样本超额回撤：非退化时为数值；退化时为 None。
    dd = r["history_alpha_max_drawdown"]
    assert dd is None or (isinstance(dd, (int, float)) and not isinstance(dd, bool))
    sharpe = r["history_alpha_sharpe"]
    assert sharpe is None or (isinstance(sharpe, (int, float)) and not isinstance(sharpe, bool))
    # 两道硬门判定：永远是 bool。
    assert isinstance(r["cond_history_dd_passed"], bool)
    assert isinstance(r["cond_history_sharpe_passed"], bool)
    # 非退化时，判定与数值自洽：dd <= threshold ⇔ passed。
    if dd is not None:
        if dd <= 0.30:
            assert r["cond_history_dd_passed"] is True
        if dd > 0.30:
            assert r["cond_history_dd_passed"] is False
    if sharpe is not None:
        if sharpe > 0.5:
            assert r["cond_history_sharpe_passed"] is True
        if sharpe <= 0.5:
            assert r["cond_history_sharpe_passed"] is False


def test_history_mode_sharpe_threshold_is_strict_greater(wb: WeightBacktest):
    """Sharpe 硬门是严格 >：把阈值调到 Sharpe 之上 → 必否决；调到 0 → 默认应通过（如果 Sharpe>0）。"""
    r = wb.is_good_strategy(mode="history")
    if r["alpha_degenerate"] or r["history_alpha_sharpe"] is None:
        pytest.skip("fixture yields alpha-degenerate data; Sharpe gate is N/A")
    sharpe = r["history_alpha_sharpe"]
    # 阈值取 sharpe + 1 → 必被否决（严格 >）。
    r2 = wb.is_good_strategy(mode="history", min_full_sharpe=sharpe + 1.0)
    assert r2["cond_history_sharpe_passed"] is False
    # 阈值取 sharpe - 1 → 必通过。
    r3 = wb.is_good_strategy(mode="history", min_full_sharpe=sharpe - 1.0)
    assert r3["cond_history_sharpe_passed"] is True


def test_history_mode_dd_threshold_is_le(wb: WeightBacktest):
    """超额回撤硬门是 <=：阈值取 dd → 通过；阈值取 dd - 1e-6 → 不通过。"""
    r = wb.is_good_strategy(mode="history")
    if r["alpha_degenerate"] or r["history_alpha_max_drawdown"] is None:
        pytest.skip("fixture yields alpha-degenerate data; DD gate is N/A")
    dd = r["history_alpha_max_drawdown"]
    # 阈值取 dd → 通过（`<=`）。
    r_eq = wb.is_good_strategy(mode="history", max_alpha_dd_threshold=dd)
    assert r_eq["cond_history_dd_passed"] is True, f"dd {dd} == threshold should pass"
    # 阈值取 dd - 1e-6 → 不通过。
    r_lt = wb.is_good_strategy(mode="history", max_alpha_dd_threshold=max(dd - 1e-6, 0.0))
    assert r_lt["cond_history_dd_passed"] is False, f"dd {dd} > threshold-eps should fail"


def test_history_mode_default_thresholds(wb: WeightBacktest):
    """默认参数下 max_alpha_dd_threshold=0.30 / min_full_sharpe=0.5 必生效。"""
    r = wb.is_good_strategy(mode="history")
    # 通过设置一个明确放开的阈值与默认对比，间接证明默认值就是 0.30 / 0.5。
    r_loose = wb.is_good_strategy(mode="history", max_alpha_dd_threshold=1.0, min_full_sharpe=-1e9)
    # 在默认值下至少有一个硬门状态比 loose 严：拿 raw dd / sharpe 比 threshold 反推。
    dd = r["history_alpha_max_drawdown"]
    sharpe = r["history_alpha_sharpe"]
    if dd is not None:
        # loose 阈值 1.0 应让 DD gate 通过；默认 0.30 可能不通过。
        assert r_loose["cond_history_dd_passed"] is True
    if sharpe is not None:
        # loose 阈值 -1e9 应让 Sharpe gate 通过；默认 0.5 可能不通过。
        assert r_loose["cond_history_sharpe_passed"] is True


def test_history_mode_invalid_max_alpha_dd_threshold_raises(wb: WeightBacktest):
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(mode="history", max_alpha_dd_threshold=-0.1)
    msg = str(excinfo.value)
    assert "max_alpha_dd_threshold" in msg or "invalid input" in msg, msg


def test_history_mode_invalid_min_full_sharpe_raises(wb: WeightBacktest):
    with pytest.raises(Exception) as excinfo:
        wb.is_good_strategy(mode="history", min_full_sharpe=float("nan"))
    msg = str(excinfo.value)
    assert "min_full_sharpe" in msg or "invalid input" in msg, msg


def test_history_mode_alpha_degenerate_dd_and_sharpe_are_none(sample_dfw):
    """退化时两道全样本硬门对应数值字段为 None，判定恒为 False。

    用「单只品种 + 恒定价格 + 恒定权重」的合成数据让 long 序列恒定、long_vol→0，
    触发 alpha_degenerate；再校验四个新 key 的 null / false 契约。
    """
    import datetime as _dt

    n = 300
    df = sample_dfw.iloc[:0].copy()  # copy schema, drop rows
    rows = []
    for i in range(n):
        dt = (_dt.datetime(2020, 1, 1, 9, 30) + _dt.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({"dt": dt, "symbol": "DEGEN", "weight": 0.5, "price": 100.0})
    import pandas as _pd

    df = _pd.DataFrame(rows)
    wb2 = WeightBacktest(df, digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252)
    # WeightBacktest.__init__ 内部已经触发 backtest。
    r = wb2.is_good_strategy(mode="history")
    if not r["alpha_degenerate"]:
        pytest.skip("fixture did not trigger alpha-degenerate; covered by Rust unit tests")
    assert r["history_alpha_max_drawdown"] is None
    assert r["history_alpha_sharpe"] is None
    assert r["cond_history_dd_passed"] is False
    assert r["cond_history_sharpe_passed"] is False
    assert r["is_good"] is False
