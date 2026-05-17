# 从 czsc 迁移 5 个函数到 wbt 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 czsc 库的 5 个函数（`cal_yearly_days`、`rolling_daily_performance`、`weights_simple_ensemble`、`cal_trade_price`、`log_strategy_info`）迁移到 wbt，前两个用 Rust 实现并通过 PyO3 暴露，后三个纯 Python。

**Architecture:** 业务逻辑全部下沉到 Rust 层（含 warning、兜底、排序、缺失值处理），Python 仅做 pandas↔Arrow 格式适配，参照现有 `wbt/top_drawdowns.py` 模式。Rust warning 通过 `log` + `pyo3-log` 桥接到 Python logging（loguru 用户可一行接管）。

**Tech Stack:** Rust (PyO3 0.28, polars, chrono, log, pyo3-log), Python (pandas, numpy, pyarrow, loguru), pytest, ruff, basedpyright, maturin.

---

## 任务概览

| # | 任务 | 产出文件 |
|---|------|---------|
| 1 | 加依赖 | `Cargo.toml`、`python/pyproject.toml` |
| 2 | Rust `cal_yearly_days` 核心 + 单测 | `src/core/cal_yearly_days.rs`、`src/core/mod.rs` |
| 3 | PyO3 暴露 `cal_yearly_days` + pyo3-log 初始化 + stub | `src/lib.rs`、`python/wbt/_wbt.pyi` |
| 4 | Python `cal_yearly_days` adapter + 测试 | `python/wbt/utils/__init__.py`、`python/wbt/utils/cal_yearly_days.py`、`python/tests/test_cal_yearly_days.py` |
| 5 | Rust `rolling_daily_performance` 核心 + 单测 | `src/core/rolling_daily_performance.rs`、`src/core/mod.rs` |
| 6 | PyO3 暴露 `rolling_daily_performance` + stub | `src/lib.rs`、`python/wbt/_wbt.pyi` |
| 7 | Python `rolling_daily_performance` adapter + 测试 | `python/wbt/utils/rolling_daily_performance.py`、`python/tests/test_rolling_daily_performance.py` |
| 8 | `weights_simple_ensemble` + 测试 | `python/wbt/utils/weights_simple_ensemble.py`、`python/tests/test_weights_simple_ensemble.py` |
| 9 | `cal_trade_price` + 测试 | `python/wbt/utils/cal_trade_price.py`、`python/tests/test_cal_trade_price.py` |
| 10 | `log_strategy_info` + 测试 | `python/wbt/utils/log_strategy_info.py`、`python/tests/test_log_strategy_info.py` |
| 11 | 顶层 `__init__.py` 透传 + 烟雾测试 | `python/wbt/__init__.py`、`python/tests/test_imports.py` |
| 12 | 质量收尾 | 全仓 cargo + pytest + ruff + basedpyright |

---

### Task 1: 加依赖

**Files:**
- Modify: `Cargo.toml`
- Modify: `python/pyproject.toml`

- [ ] **Step 1: 在 Cargo.toml 的 `[dependencies]` 段追加 log 与 pyo3-log**

打开 `Cargo.toml`，找到 `[dependencies]` 段（约 21 行），在 `hashbrown = "0.17"` 下方追加：

```toml
log = "0.4"
pyo3-log = "0.13"
```

- [ ] **Step 2: 在 python/pyproject.toml 的 dependencies 列表追加 loguru**

打开 `python/pyproject.toml`，找到第 12 行：

```toml
dependencies = ["numpy", "pandas", "pyarrow", "polars", "plotly"]
```

替换为：

```toml
dependencies = ["numpy", "pandas", "pyarrow", "polars", "plotly", "loguru"]
```

- [ ] **Step 3: 验证 Cargo 依赖能拉取**

Run: `cargo fetch`
Expected: 退出码 0，新增 `log` 和 `pyo3-log` crates 下载成功

- [ ] **Step 4: 提交**

```bash
git add Cargo.toml python/pyproject.toml
git commit -m "build: add log+pyo3-log (rust) and loguru (python) deps"
```

---

### Task 2: Rust `cal_yearly_days` 核心 + 单测

**Files:**
- Create: `src/core/cal_yearly_days.rs`
- Modify: `src/core/mod.rs`

- [ ] **Step 1: 写失败的 Rust 单测（先创建文件并填好 tests 模块）**

创建 `src/core/cal_yearly_days.rs`：

```rust
use chrono::{Datelike, NaiveDate};
use std::collections::{BTreeSet, HashMap};

/// 计算年度交易日数量（与 czsc.eda.cal_yearly_days 行为完全一致）。
///
/// 业务规则（全部在 Rust 内）：
/// 1. 输入为空 → panic（PyO3 wrapper 转 PyException）；
/// 2. 去重后样本跨度 < 365 天 → 通过 `log::warn!` 提示，返回 252；
/// 3. 否则按自然年聚合取交易日数 max，钳制到 [1, 365]。
pub fn cal_yearly_days(dates: &[NaiveDate]) -> i64 {
    assert!(!dates.is_empty(), "输入的日期数量必须大于0");

    let dedup: BTreeSet<NaiveDate> = dates.iter().copied().collect();
    let min = *dedup.iter().next().unwrap();
    let max = *dedup.iter().next_back().unwrap();

    if (max - min).num_days() < 365 {
        log::warn!("时间跨度小于一年，直接返回 252");
        return 252;
    }

    let mut per_year: HashMap<i32, i64> = HashMap::new();
    for d in &dedup {
        *per_year.entry(d.year()).or_insert(0) += 1;
    }
    per_year.values().copied().max().unwrap_or(0).min(365)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    #[should_panic(expected = "输入的日期数量必须大于0")]
    fn empty_input_panics() {
        cal_yearly_days(&[]);
    }

    #[test]
    fn span_less_than_one_year_returns_252() {
        let dates: Vec<NaiveDate> = (1..=200).map(|i| d(2024, 1, 1) + chrono::Duration::days(i)).collect();
        assert_eq!(cal_yearly_days(&dates), 252);
    }

    #[test]
    fn multi_year_returns_max_year_count() {
        // 2022 有 260 个交易日，2023 有 250 个，跨度 > 365
        let mut dates = Vec::new();
        for i in 0..260 {
            dates.push(d(2022, 1, 1) + chrono::Duration::days(i));
        }
        for i in 0..250 {
            dates.push(d(2023, 1, 1) + chrono::Duration::days(i));
        }
        // 加一个 2024 年初的日期保证 max-min > 365
        dates.push(d(2024, 2, 1));
        let result = cal_yearly_days(&dates);
        assert!(result >= 250 && result <= 260, "got {}", result);
    }

    #[test]
    fn clamped_to_365() {
        // 极端情况：连续每天都有数据（>365 天/年时仍钳到 365）
        let dates: Vec<NaiveDate> = (0..800).map(|i| d(2020, 1, 1) + chrono::Duration::days(i)).collect();
        assert!(cal_yearly_days(&dates) <= 365);
    }
}
```

- [ ] **Step 2: 注册模块**

修改 `src/core/mod.rs`，在 `pub mod yearly_return;` 同级位置追加：

```rust
pub mod cal_yearly_days;
```

- [ ] **Step 3: 运行单测验证通过**

Run: `cargo test --lib core::cal_yearly_days::tests`
Expected: 4 tests passed (empty_input_panics, span_less_than_one_year_returns_252, multi_year_returns_max_year_count, clamped_to_365)

- [ ] **Step 4: 提交**

```bash
git add src/core/cal_yearly_days.rs src/core/mod.rs
git commit -m "feat(core): add cal_yearly_days with span fallback and warning"
```

---

### Task 3: PyO3 暴露 `cal_yearly_days` + pyo3-log 初始化 + stub

**Files:**
- Modify: `src/lib.rs`
- Modify: `python/wbt/_wbt.pyi`

- [ ] **Step 1: 在 src/lib.rs 添加 `#[pyfunction] cal_yearly_days`**

在 `src/lib.rs` 末尾的 `#[pymodule] fn _wbt` 函数之前（约第 377 行附近），插入：

```rust
// ---------------------------------------------------------------------------
// cal_yearly_days standalone function
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (timestamps_ms))]
pub fn cal_yearly_days(timestamps_ms: Vec<i64>) -> PyResult<i64> {
    use chrono::DateTime;
    if timestamps_ms.is_empty() {
        return Err(PyException::new_err("输入的日期数量必须大于0"));
    }
    let dates: Vec<chrono::NaiveDate> = timestamps_ms
        .iter()
        .filter_map(|ms| DateTime::from_timestamp_millis(*ms).map(|d| d.naive_utc().date()))
        .collect();
    if dates.is_empty() {
        return Err(PyException::new_err("输入的日期数量必须大于0"));
    }
    Ok(crate::core::cal_yearly_days::cal_yearly_days(&dates))
}
```

- [ ] **Step 2: 在 `_wbt` pymodule 顶部初始化 pyo3-log，并注册 cal_yearly_days**

把 `src/lib.rs` 中的 `_wbt` 函数（约第 381 行）：

```rust
#[pymodule]
fn _wbt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWeightBacktest>()?;
    m.add_function(wrap_pyfunction!(daily_performance, m)?)?;
    m.add_function(wrap_pyfunction!(top_drawdowns, m)?)?;
    Ok(())
}
```

替换为：

```rust
#[pymodule]
fn _wbt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bridge Rust log::warn! → Python logging (loguru 用户可一行接管)
    let _ = pyo3_log::try_init();

    m.add_class::<PyWeightBacktest>()?;
    m.add_function(wrap_pyfunction!(daily_performance, m)?)?;
    m.add_function(wrap_pyfunction!(top_drawdowns, m)?)?;
    m.add_function(wrap_pyfunction!(cal_yearly_days, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: 在 python/wbt/_wbt.pyi 末尾追加 stub**

打开 `python/wbt/_wbt.pyi`，在 `def top_drawdowns(...)` 行下方追加：

```python
def cal_yearly_days(timestamps_ms: list[int]) -> int: ...
```

- [ ] **Step 4: 运行 maturin develop 重建 wheel**

Run: `cd python && maturin develop --release`
Expected: 编译成功，未出现 warning 之外的报错

- [ ] **Step 5: 验证 Python 可导入（仅烟雾测试，数值正确性由 cargo test 保证）**

Run: `python -c "from wbt._wbt import cal_yearly_days; print(cal_yearly_days([1704067200000, 1735689600000]))"`
Expected: 打印一个整数（不抛异常即可，本步只验证 PyO3 入口可达）

- [ ] **Step 6: 提交**

```bash
git add src/lib.rs python/wbt/_wbt.pyi
git commit -m "feat(py): expose cal_yearly_days via PyO3 + init pyo3-log bridge"
```

---

### Task 4: Python `cal_yearly_days` adapter + 测试

**Files:**
- Create: `python/wbt/utils/__init__.py`
- Create: `python/wbt/utils/cal_yearly_days.py`
- Create: `python/tests/test_cal_yearly_days.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_cal_yearly_days.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败（模块不存在）**

Run: `pytest python/tests/test_cal_yearly_days.py -v`
Expected: 4 个 test 全部 FAIL，错误信息提示 `wbt.utils.cal_yearly_days` 模块不存在

- [ ] **Step 3: 创建 `python/wbt/utils/__init__.py`（空文件）**

创建空文件 `python/wbt/utils/__init__.py`（内容为单个空行）：

```python
```

- [ ] **Step 4: 写实现 `python/wbt/utils/cal_yearly_days.py`**

```python
"""Minimal Python adapter for the Rust ``_wbt.cal_yearly_days`` symbol.

All business rules (span check, 252 fallback, warning) live in Rust.
This module only converts Python date-like iterables to unix-ms i64 lists.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from wbt._wbt import cal_yearly_days as _cal_yearly_days_rs


def cal_yearly_days(dts: Iterable) -> int:
    """计算年度交易日数量。

    业务规则（跨度判定、252 兜底、warning）由 Rust 层完成；
    本函数仅负责把 Python 端的日期序列转成 Rust 接收的 unix-ms 列表。
    """
    ts_ms = (pd.to_datetime(pd.Series(list(dts))).astype("int64") // 1_000_000).tolist()
    return int(_cal_yearly_days_rs(ts_ms))
```

- [ ] **Step 5: 运行测试验证全部通过**

Run: `pytest python/tests/test_cal_yearly_days.py -v`
Expected: 4 tests passed

- [ ] **Step 6: 提交**

```bash
git add python/wbt/utils/__init__.py python/wbt/utils/cal_yearly_days.py python/tests/test_cal_yearly_days.py
git commit -m "feat(utils): add cal_yearly_days adapter calling Rust core"
```

---

### Task 5: Rust `rolling_daily_performance` 核心 + 单测

**Files:**
- Create: `src/core/rolling_daily_performance.rs`
- Modify: `src/core/mod.rs`

- [ ] **Step 1: 创建 Rust 核心实现**

创建 `src/core/rolling_daily_performance.rs`：

```rust
use crate::core::cal_yearly_days::cal_yearly_days;
use crate::core::daily_performance::{DailyPerformance, daily_performance};
use chrono::NaiveDate;
use polars::prelude::*;

/// 滚动日度绩效（与 czsc.utils.analysis.stats.rolling_daily_performance 行为一致）。
///
/// 业务规则（全部在 Rust 内）：
/// 1. dates / returns 长度需一致；
/// 2. 按 dates 升序排序（不要求调用方预排）；
/// 3. returns 中的 NaN 视为 0；
/// 4. yearly_days 未提供时，调用 cal_yearly_days 自动推断；
/// 5. 滚动窗口：跳过前 min_periods 个点，每个 edt 取 (edt-window, edt] 区间。
pub fn rolling_daily_performance(
    dates: Vec<NaiveDate>,
    returns: Vec<f64>,
    window: i64,
    min_periods: usize,
    yearly_days: Option<usize>,
) -> PolarsResult<DataFrame> {
    assert_eq!(dates.len(), returns.len(), "dates 与 returns 长度必须一致");

    let mut indexed: Vec<(NaiveDate, f64)> = dates
        .into_iter()
        .zip(returns)
        .map(|(d, r)| (d, if r.is_nan() { 0.0 } else { r }))
        .collect();
    indexed.sort_by_key(|(d, _)| *d);

    let dates: Vec<NaiveDate> = indexed.iter().map(|(d, _)| *d).collect();
    let returns: Vec<f64> = indexed.iter().map(|(_, r)| *r).collect();

    let yd = yearly_days.unwrap_or_else(|| cal_yearly_days(&dates) as usize);
    let n = dates.len();

    let mut sdt_vec: Vec<NaiveDate> = Vec::with_capacity(n);
    let mut edt_vec: Vec<NaiveDate> = Vec::with_capacity(n);
    let mut perfs: Vec<DailyPerformance> = Vec::with_capacity(n);
    if min_periods < n {
        for end_idx in min_periods..n {
            let edt = dates[end_idx];
            let sdt = edt - chrono::Duration::days(window);
            let start_idx = dates.partition_point(|d| *d < sdt);
            let slice = &returns[start_idx..=end_idx];
            let perf = daily_performance(slice, Some(yd))
                .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?;
            perfs.push(perf);
            sdt_vec.push(sdt);
            edt_vec.push(edt);
        }
    }

    build_dataframe(&sdt_vec, &edt_vec, &perfs)
}

fn build_dataframe(
    sdt: &[NaiveDate],
    edt: &[NaiveDate],
    perfs: &[DailyPerformance],
) -> PolarsResult<DataFrame> {
    let to_date_series = |name: &str, v: &[NaiveDate]| {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let days: Vec<i32> = v.iter().map(|d| (*d - epoch).num_days() as i32).collect();
        Series::new(name.into(), days)
            .cast(&DataType::Date)
            .map(|s| s.into_column())
    };

    let abs_ret: Vec<f64> = perfs.iter().map(|p| p.absolute_return).collect();
    let ann_ret: Vec<f64> = perfs.iter().map(|p| p.annual_returns).collect();
    let sharpe: Vec<f64> = perfs.iter().map(|p| p.sharpe_ratio).collect();
    let mdd: Vec<f64> = perfs.iter().map(|p| p.max_drawdown).collect();
    let calmar: Vec<f64> = perfs.iter().map(|p| p.calmar_ratio).collect();
    let win_rate: Vec<f64> = perfs.iter().map(|p| p.daily_win_rate).collect();
    let pl_ratio: Vec<f64> = perfs.iter().map(|p| p.daily_profit_loss_ratio).collect();
    let win_prob: Vec<f64> = perfs.iter().map(|p| p.daily_win_probability).collect();
    let ann_vol: Vec<f64> = perfs.iter().map(|p| p.annual_volatility).collect();
    let down_vol: Vec<f64> = perfs.iter().map(|p| p.downside_volatility).collect();
    let nz_cov: Vec<f64> = perfs.iter().map(|p| p.non_zero_coverage).collect();
    let bep: Vec<f64> = perfs.iter().map(|p| p.break_even_point).collect();
    let nh_int: Vec<i64> = perfs.iter().map(|p| p.new_high_interval).collect();
    let nh_ratio: Vec<f64> = perfs.iter().map(|p| p.new_high_ratio).collect();
    let dd_risk: Vec<f64> = perfs.iter().map(|p| p.drawdown_risk).collect();
    let ann_lr: Vec<f64> = perfs.iter().map(|p| p.annual_lin_reg_cumsum_return).collect();
    let la_mdd: Vec<f64> = perfs.iter().map(|p| p.length_adjusted_average_max_drawdown).collect();

    DataFrame::new_infer_height(vec![
        Series::new("绝对收益".into(), abs_ret).into_column(),
        Series::new("年化".into(), ann_ret).into_column(),
        Series::new("夏普".into(), sharpe).into_column(),
        Series::new("最大回撤".into(), mdd).into_column(),
        Series::new("卡玛".into(), calmar).into_column(),
        Series::new("日胜率".into(), win_rate).into_column(),
        Series::new("日盈亏比".into(), pl_ratio).into_column(),
        Series::new("日赢面".into(), win_prob).into_column(),
        Series::new("年化波动率".into(), ann_vol).into_column(),
        Series::new("下行波动率".into(), down_vol).into_column(),
        Series::new("非零覆盖".into(), nz_cov).into_column(),
        Series::new("盈亏平衡点".into(), bep).into_column(),
        Series::new("新高间隔".into(), nh_int).into_column(),
        Series::new("新高占比".into(), nh_ratio).into_column(),
        Series::new("回撤风险".into(), dd_risk).into_column(),
        Series::new("回归年度回报率".into(), ann_lr).into_column(),
        Series::new("长度调整平均最大回撤".into(), la_mdd).into_column(),
        to_date_series("sdt", sdt)?,
        to_date_series("edt", edt)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    #[should_panic(expected = "dates 与 returns 长度必须一致")]
    fn mismatched_lengths_panic() {
        let _ = rolling_daily_performance(vec![d(2024, 1, 1)], vec![0.01, 0.02], 252, 1, Some(252));
    }

    #[test]
    fn min_periods_skips_warmup() {
        let dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        let returns: Vec<f64> = (0..400).map(|i| (i as f64) * 0.0001).collect();
        let df = rolling_daily_performance(dates, returns, 252, 100, Some(252)).unwrap();
        assert_eq!(df.height(), 300); // 400 - 100
    }

    #[test]
    fn nan_returns_treated_as_zero() {
        let dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        let mut returns: Vec<f64> = vec![0.001; 400];
        returns[10] = f64::NAN;
        returns[200] = f64::NAN;
        // 不应 panic
        let df = rolling_daily_performance(dates, returns, 252, 100, Some(252)).unwrap();
        assert_eq!(df.height(), 300);
    }

    #[test]
    fn yearly_days_auto_inferred_when_none() {
        let dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        let returns: Vec<f64> = vec![0.001; 400];
        let df = rolling_daily_performance(dates, returns, 252, 100, None).unwrap();
        assert!(df.height() > 0);
    }

    #[test]
    fn unsorted_input_is_sorted_internally() {
        let mut dates: Vec<NaiveDate> = (0..400).map(|i| d(2022, 1, 1) + chrono::Duration::days(i)).collect();
        dates.reverse();
        let returns: Vec<f64> = vec![0.001; 400];
        let df = rolling_daily_performance(dates, returns, 252, 100, Some(252)).unwrap();
        let edt_col = df.column("edt").unwrap();
        let edts: Vec<i32> = edt_col
            .as_materialized_series()
            .date()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!(edts.windows(2).all(|w| w[0] <= w[1]));
    }
}
```

- [ ] **Step 2: 注册模块**

修改 `src/core/mod.rs`，在 `pub mod cal_yearly_days;` 下方追加：

```rust
pub mod rolling_daily_performance;
```

- [ ] **Step 3: 运行 Rust 测试**

Run: `cargo test --lib core::rolling_daily_performance::tests`
Expected: 5 tests passed

- [ ] **Step 4: 提交**

```bash
git add src/core/rolling_daily_performance.rs src/core/mod.rs
git commit -m "feat(core): add rolling_daily_performance with auto sort/nan/yearly_days"
```

---

### Task 6: PyO3 暴露 `rolling_daily_performance` + stub

**Files:**
- Modify: `src/lib.rs`
- Modify: `python/wbt/_wbt.pyi`

- [ ] **Step 1: 在 src/lib.rs 添加 `rolling_daily_performance` pyfunction**

在 `src/lib.rs` 的 `cal_yearly_days` pyfunction 之后追加：

```rust
// ---------------------------------------------------------------------------
// rolling_daily_performance standalone function
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, ret_col, window=252, min_periods=100, yearly_days=None))]
pub fn rolling_daily_performance<'py>(
    py: Python<'py>,
    data: Bound<'py, PyBytes>,
    ret_col: &str,
    window: i64,
    min_periods: usize,
    yearly_days: Option<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let df_in = pyarrow_to_df(data.as_bytes())?;

    let dt_col = df_in
        .column("dt")
        .map_err(|e| PyException::new_err(format!("missing 'dt' column: {e}")))?;
    let dates: Vec<chrono::NaiveDate> = match dt_col.dtype() {
        DataType::Datetime(_, _) => dt_col
            .datetime()
            .map_err(|e| PyException::new_err(e.to_string()))?
            .as_datetime_iter()
            .flatten()
            .map(|d| d.date())
            .collect(),
        DataType::Date => dt_col
            .date()
            .map_err(|e| PyException::new_err(e.to_string()))?
            .as_date_iter()
            .flatten()
            .collect(),
        other => {
            return Err(PyException::new_err(format!(
                "Unsupported dt dtype: {other:?} (expected Date or Datetime)"
            )));
        }
    };

    let returns: Vec<f64> = df_in
        .column(ret_col)
        .map_err(|e| PyException::new_err(format!("missing '{ret_col}' column: {e}")))?
        .f64()
        .map_err(|e| PyException::new_err(e.to_string()))?
        .into_iter()
        .map(|opt| opt.unwrap_or(f64::NAN))
        .collect();

    let mut df_out = crate::core::rolling_daily_performance::rolling_daily_performance(
        dates,
        returns,
        window,
        min_periods,
        yearly_days,
    )
    .map_err(|e| PyException::new_err(e.to_string()))?;
    let bytes = df_to_pyarrow(&mut df_out)?;
    Ok(PyBytes::new(py, &bytes))
}
```

- [ ] **Step 2: 在 `_wbt` pymodule 中注册**

在 `_wbt` 函数最后一个 `add_function` 之后追加：

```rust
    m.add_function(wrap_pyfunction!(rolling_daily_performance, m)?)?;
```

- [ ] **Step 3: 在 _wbt.pyi 追加 stub**

打开 `python/wbt/_wbt.pyi`，在 `def cal_yearly_days(...)` 行下方追加：

```python
def rolling_daily_performance(
    data: bytes,
    ret_col: str,
    window: int = 252,
    min_periods: int = 100,
    yearly_days: int | None = None,
) -> bytes: ...
```

- [ ] **Step 4: maturin 重建**

Run: `cd python && maturin develop --release`
Expected: 编译通过

- [ ] **Step 5: 验证 Python 可导入**

Run: `python -c "from wbt._wbt import rolling_daily_performance; print(rolling_daily_performance)"`
Expected: 打印 `<built-in function rolling_daily_performance>` 或类似

- [ ] **Step 6: 提交**

```bash
git add src/lib.rs python/wbt/_wbt.pyi
git commit -m "feat(py): expose rolling_daily_performance via PyO3"
```

---

### Task 7: Python `rolling_daily_performance` adapter + 测试

**Files:**
- Create: `python/wbt/utils/rolling_daily_performance.py`
- Create: `python/tests/test_rolling_daily_performance.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_rolling_daily_performance.py`：

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dts = pd.date_range("2022-01-01", periods=n, freq="D")
    rets = rng.normal(0.0005, 0.01, size=n)
    return pd.DataFrame({"dt": dts, "ret": rets})


def test_dt_column_input() -> None:
    from wbt.utils.rolling_daily_performance import rolling_daily_performance

    df = _sample_df()
    out = rolling_daily_performance(df, "ret", window=252, min_periods=100)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 300
    for col in ("sdt", "edt", "年化", "夏普", "最大回撤"):
        assert col in out.columns


def test_dt_index_input() -> None:
    from wbt.utils.rolling_daily_performance import rolling_daily_performance

    df = _sample_df().set_index("dt")
    out = rolling_daily_performance(df, "ret", window=252, min_periods=100)
    assert len(out) == 300


def test_explicit_yearly_days() -> None:
    from wbt.utils.rolling_daily_performance import rolling_daily_performance

    df = _sample_df()
    out = rolling_daily_performance(df, "ret", window=252, min_periods=100, yearly_days=252)
    assert len(out) == 300
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest python/tests/test_rolling_daily_performance.py -v`
Expected: 3 tests FAIL，提示模块不存在

- [ ] **Step 3: 写实现**

创建 `python/wbt/utils/rolling_daily_performance.py`：

```python
"""Minimal Python adapter for the Rust ``_wbt.rolling_daily_performance`` symbol.

All business logic (sorting, NaN handling, yearly_days inference, rolling loop)
lives in Rust. This module only adapts pandas DataFrame ↔ Arrow IPC bytes.
"""

from __future__ import annotations

import pandas as pd

from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes
from wbt._wbt import rolling_daily_performance as _rolling_rs


def rolling_daily_performance(
    df: pd.DataFrame,
    ret_col: str,
    window: int = 252,
    min_periods: int = 100,
    yearly_days: int | None = None,
) -> pd.DataFrame:
    """计算滚动日收益的各项指标（业务逻辑在 Rust 内）。

    :param df: 日收益数据，columns 含 ['dt', ret_col]，或 index 为 datetime
    :param ret_col: 收益列名
    :param window: 滚动窗口（自然天数）
    :param min_periods: 最小样本数
    :param yearly_days: 年度交易日数；None 时由 Rust 调 cal_yearly_days 推断
    """
    if isinstance(df.index, pd.DatetimeIndex):
        work = pd.DataFrame({"dt": df.index, ret_col: df[ret_col].values})
    else:
        work = df[["dt", ret_col]].copy()
        work["dt"] = pd.to_datetime(work["dt"])

    data = pandas_to_arrow_bytes(work)
    out_bytes = _rolling_rs(data, ret_col, window, min_periods, yearly_days)
    return arrow_bytes_to_pd_df(out_bytes)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest python/tests/test_rolling_daily_performance.py -v`
Expected: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add python/wbt/utils/rolling_daily_performance.py python/tests/test_rolling_daily_performance.py
git commit -m "feat(utils): add rolling_daily_performance adapter"
```

---

### Task 8: `weights_simple_ensemble` + 测试

**Files:**
- Create: `python/wbt/utils/weights_simple_ensemble.py`
- Create: `python/tests/test_weights_simple_ensemble.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_weights_simple_ensemble.py`：

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "w1": [1.0, -1.0, 0.5, 0.0],
            "w2": [-0.5, 1.0, 0.5, 0.0],
            "w3": [0.5, 0.0, -1.0, 1.0],
        }
    )


def test_mean_method() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = weights_simple_ensemble(_sample(), ["w1", "w2", "w3"], method="mean")
    assert np.allclose(df["weight"].tolist(), [1.0 / 3, 0.0, 0.0, 1.0 / 3])


def test_vote_method() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = weights_simple_ensemble(_sample(), ["w1", "w2", "w3"], method="vote")
    assert df["weight"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_sum_clip_method() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = weights_simple_ensemble(
        _sample(), ["w1", "w2", "w3"], method="sum_clip", clip_min=-0.5, clip_max=0.5
    )
    assert df["weight"].tolist() == [0.5, 0.0, 0.0, 0.5]


def test_only_long_zeroes_negatives() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    df = pd.DataFrame({"w1": [1.0, -1.0], "w2": [-2.0, 0.5]})
    out = weights_simple_ensemble(df, ["w1", "w2"], method="mean", only_long=True)
    assert out["weight"].tolist() == [0.0, 0.0]


def test_missing_col_asserts() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    with pytest.raises(AssertionError, match="缺失"):
        weights_simple_ensemble(_sample(), ["w1", "missing"])


def test_invalid_method_raises() -> None:
    from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

    with pytest.raises(ValueError, match="method 参数错误"):
        weights_simple_ensemble(_sample(), ["w1", "w2"], method="unknown")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest python/tests/test_weights_simple_ensemble.py -v`
Expected: 6 tests FAIL

- [ ] **Step 3: 写实现**

创建 `python/wbt/utils/weights_simple_ensemble.py`：

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest python/tests/test_weights_simple_ensemble.py -v`
Expected: 6 tests passed

- [ ] **Step 5: 提交**

```bash
git add python/wbt/utils/weights_simple_ensemble.py python/tests/test_weights_simple_ensemble.py
git commit -m "feat(utils): add weights_simple_ensemble (mean/vote/sum_clip)"
```

---

### Task 9: `cal_trade_price` + 测试

**Files:**
- Create: `python/wbt/utils/cal_trade_price.py`
- Create: `python/tests/test_cal_trade_price.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_cal_trade_price.py`：

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def _bars(symbol: str = "A", n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "dt": pd.date_range("2024-01-01", periods=n, freq="min"),
            "open": np.arange(n, dtype=float) + 100,
            "close": np.arange(n, dtype=float) + 101,
            "vol": np.arange(n, dtype=float) + 1,
        }
    )


def test_columns_present_for_default_windows() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    out = cal_trade_price(_bars())
    expected = {"TP_CLOSE", "TP_NEXT_OPEN", "TP_NEXT_CLOSE"}
    for w in (5, 10, 15, 20, 30, 60):
        expected.add(f"TP_TWAP{w}")
        expected.add(f"TP_VWAP{w}")
    assert expected.issubset(set(out.columns))


def test_multi_symbol_concat_order_preserved() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    bars = pd.concat([_bars("A"), _bars("B")], ignore_index=True)
    out = cal_trade_price(bars)
    assert set(out["symbol"].unique()) == {"A", "B"}


def test_custom_windows() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    out = cal_trade_price(_bars(), windows=(3, 7))
    assert "TP_TWAP3" in out.columns
    assert "TP_VWAP7" in out.columns
    assert "TP_TWAP5" not in out.columns


def test_tp_close_equals_close() -> None:
    from wbt.utils.cal_trade_price import cal_trade_price

    bars = _bars()
    out = cal_trade_price(bars)
    assert (out["TP_CLOSE"] == bars["close"]).all()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest python/tests/test_cal_trade_price.py -v`
Expected: 4 tests FAIL

- [ ] **Step 3: 写实现**

创建 `python/wbt/utils/cal_trade_price.py`：

```python
"""按 symbol 分组生成 TWAP / VWAP 等下根 K 线交易价表。

🟡 Experimental：windows 默认值可能调整。
"""

from __future__ import annotations

import pandas as pd


def cal_trade_price(df: pd.DataFrame, digits: int | None = None, **kwargs) -> pd.DataFrame:
    """计算给定品种基础周期 K 线数据的交易价格表。

    :param df: 基础周期 K 线，必须包含 symbol/dt/open/close/vol 列
    :param digits: 保留小数位数；None 时按各品种 close 列推断
    :param kwargs:
        - windows: TWAP/VWAP 窗口列表，默认 (5, 10, 15, 20, 30, 60)
        - copy: 是否复制输入，默认 True
    """
    assert "symbol" in df.columns, "数据中必须包含 symbol 列"
    for col in ("dt", "open", "close", "vol"):
        assert col in df.columns, f"数据中必须包含 {col} 列"

    if kwargs.get("copy", True):
        df = df.copy()

    symbols = df["symbol"].unique().tolist()
    windows = kwargs.get("windows", (5, 10, 15, 20, 30, 60))

    dfs: list[pd.DataFrame] = []
    for symbol in symbols:
        sub = df[df["symbol"] == symbol].sort_values("dt").reset_index(drop=True)

        sym_digits = digits
        if sym_digits is None:
            sym_digits = sub["close"].astype(str).str.split(".").str[1].str.len().max()
            if pd.isna(sym_digits):
                sym_digits = 0
            sym_digits = int(sym_digits)

        sub["TP_CLOSE"] = sub["close"]
        sub["TP_NEXT_OPEN"] = sub["open"].shift(-1)
        sub["TP_NEXT_CLOSE"] = sub["close"].shift(-1)
        price_cols = ["TP_CLOSE", "TP_NEXT_OPEN", "TP_NEXT_CLOSE"]

        sub["_vcp"] = sub["vol"] * sub["close"]
        for t in windows:
            sub[f"TP_TWAP{t}"] = sub["close"].rolling(t).mean().shift(-t)
            vol_sum = sub["vol"].rolling(t).sum()
            vcp_sum = sub["_vcp"].rolling(t).sum()
            sub[f"TP_VWAP{t}"] = (vcp_sum / vol_sum).shift(-t)
            price_cols.extend([f"TP_TWAP{t}", f"TP_VWAP{t}"])
        sub.drop(columns=["_vcp"], inplace=True)

        for pc in price_cols:
            sub[pc] = sub[pc].fillna(sub["close"])
        sub[price_cols] = sub[price_cols].round(sym_digits)
        dfs.append(sub)

    return pd.concat(dfs, ignore_index=True)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest python/tests/test_cal_trade_price.py -v`
Expected: 4 tests passed

- [ ] **Step 5: 提交**

```bash
git add python/wbt/utils/cal_trade_price.py python/tests/test_cal_trade_price.py
git commit -m "feat(utils): add cal_trade_price with TWAP/VWAP columns"
```

---

### Task 10: `log_strategy_info` + 测试

**Files:**
- Create: `python/wbt/utils/log_strategy_info.py`
- Create: `python/tests/test_log_strategy_info.py`

- [ ] **Step 1: 写失败的测试**

创建 `python/tests/test_log_strategy_info.py`：

```python
from __future__ import annotations

import logging

import pandas as pd
import pytest
from loguru import logger as _loguru_logger


@pytest.fixture
def loguru_to_caplog(caplog: pytest.LogCaptureFixture):
    """把 loguru 输出桥到标准 logging 以便 caplog 捕获。"""

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = _loguru_logger.add(_Sink(), level="DEBUG")
    yield caplog
    _loguru_logger.remove(handler_id)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A"] * 3 + ["B"] * 3,
            "dt": pd.date_range("2024-01-01", periods=3).tolist() * 2,
            "weight": [0.1, -0.2, 0.0, 0.3, 0.0, None],
        }
    )


def test_normal_df_logs_basic_info(loguru_to_caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.log_strategy_info import log_strategy_info

    with loguru_to_caplog.at_level(logging.INFO):
        log_strategy_info("S1", _sample_df())
    text = "\n".join(rec.message for rec in loguru_to_caplog.records)
    assert "策略 S1 数据详情" in text
    assert "品种数量: 2" in text


def test_empty_df_warns_only(loguru_to_caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.log_strategy_info import log_strategy_info

    with loguru_to_caplog.at_level(logging.WARNING):
        log_strategy_info("S2", pd.DataFrame(columns=["symbol", "dt", "weight"]))
    assert any("数据为空" in rec.message for rec in loguru_to_caplog.records)


def test_quality_warning_for_nan_and_zero(loguru_to_caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.log_strategy_info import log_strategy_info

    with loguru_to_caplog.at_level(logging.WARNING):
        log_strategy_info("S3", _sample_df())
    assert any("数据质量提醒" in rec.message for rec in loguru_to_caplog.records)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest python/tests/test_log_strategy_info.py -v`
Expected: 3 tests FAIL

- [ ] **Step 3: 写实现**

创建 `python/wbt/utils/log_strategy_info.py`：

```python
"""打印策略数据的详细信息。"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def log_strategy_info(strategy: str, df: pd.DataFrame) -> None:
    """打印策略数据详情，包括每个品种的数据详情。"""
    logger.info("-" * 100)
    if df.empty:
        logger.warning(f"策略 {strategy} 数据为空！")
        return

    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)

    logger.info(f"策略 {strategy} 数据详情：")
    logger.info(f"  总记录数: {len(df)}")
    logger.info(f"  时间范围: {df['dt'].min()} ~ {df['dt'].max()}; 时间点数: {df['dt'].nunique()}")
    logger.info(f"  品种数量: {df['symbol'].nunique()}")
    logger.info("  品种详情:")
    for symbol in sorted(df["symbol"].unique()):
        sub = df[df["symbol"] == symbol]
        if "weight" in sub.columns:
            ws = sub["weight"].describe()
            logger.info(
                f"    {symbol}: 记录数={len(sub)}, 时间={sub['dt'].min()}~{sub['dt'].max()}, "
                f"权重范围=[{ws['min']:.3f}, {ws['max']:.3f}], 平均={ws['mean']:.3f}"
            )
        else:
            logger.info(f"    {symbol}: 记录数={len(sub)}, 时间={sub['dt'].min()}~{sub['dt'].max()}")

    if "weight" in df.columns:
        null_w = int(df["weight"].isnull().sum())
        zero_w = int((df["weight"] == 0).sum())
        if null_w or zero_w:
            logger.warning(f"  数据质量提醒: 空权重={null_w}, 零权重={zero_w}")
    logger.info("-" * 100)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest python/tests/test_log_strategy_info.py -v`
Expected: 3 tests passed

- [ ] **Step 5: 提交**

```bash
git add python/wbt/utils/log_strategy_info.py python/tests/test_log_strategy_info.py
git commit -m "feat(utils): add log_strategy_info using loguru"
```

---

### Task 11: 顶层 `__init__.py` 透传 + 烟雾测试

**Files:**
- Modify: `python/wbt/utils/__init__.py`
- Modify: `python/wbt/__init__.py`
- Modify: `python/tests/test_imports.py`

- [ ] **Step 1: 填充 `python/wbt/utils/__init__.py`**

把 `python/wbt/utils/__init__.py` 替换为：

```python
from wbt.utils.cal_trade_price import cal_trade_price
from wbt.utils.cal_yearly_days import cal_yearly_days
from wbt.utils.log_strategy_info import log_strategy_info
from wbt.utils.rolling_daily_performance import rolling_daily_performance
from wbt.utils.weights_simple_ensemble import weights_simple_ensemble

__all__ = [
    "cal_trade_price",
    "cal_yearly_days",
    "log_strategy_info",
    "rolling_daily_performance",
    "weights_simple_ensemble",
]
```

- [ ] **Step 2: 修改 `python/wbt/__init__.py` 透传**

替换为：

```python
from wbt._wbt import daily_performance
from wbt.backtest import WeightBacktest, backtest
from wbt.mock import mock_symbol_kline, mock_weights
from wbt.report import generate_backtest_report
from wbt.top_drawdowns import top_drawdowns
from wbt.utils import (
    cal_trade_price,
    cal_yearly_days,
    log_strategy_info,
    rolling_daily_performance,
    weights_simple_ensemble,
)

__all__ = [
    "WeightBacktest",
    "backtest",
    "cal_trade_price",
    "cal_yearly_days",
    "daily_performance",
    "generate_backtest_report",
    "log_strategy_info",
    "mock_symbol_kline",
    "mock_weights",
    "rolling_daily_performance",
    "top_drawdowns",
    "weights_simple_ensemble",
]
```

- [ ] **Step 3: 追加烟雾测试**

修改 `python/tests/test_imports.py`，把 `test_public_exports` 改为：

```python
from __future__ import annotations

import wbt


def test_public_exports() -> None:
    """验证包公开导出的核心对象可用。"""
    assert wbt.WeightBacktest is not None
    assert wbt.daily_performance is not None


def test_migrated_czsc_exports() -> None:
    """5 个从 czsc 迁移过来的 API 都能从顶层导入。"""
    assert callable(wbt.cal_yearly_days)
    assert callable(wbt.rolling_daily_performance)
    assert callable(wbt.weights_simple_ensemble)
    assert callable(wbt.cal_trade_price)
    assert callable(wbt.log_strategy_info)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest python/tests/test_imports.py -v`
Expected: 2 tests passed

- [ ] **Step 5: 提交**

```bash
git add python/wbt/utils/__init__.py python/wbt/__init__.py python/tests/test_imports.py
git commit -m "feat(wbt): expose 5 migrated czsc APIs at top level"
```

---

### Task 12: 质量收尾

**Files:**
- 无新增；运行全仓质量门

- [ ] **Step 1: 运行全部 Rust 测试**

Run: `cargo test --lib`
Expected: 全部 tests passed（含 Task 2 / Task 5 新增的）

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy --lib -- -D warnings`
Expected: 无 warning，无错误

- [ ] **Step 3: 运行全部 Python 测试**

Run: `cd python && pytest tests -v`
Expected: 所有原有测试 + 5 个新增测试文件全部通过

- [ ] **Step 4: ruff 格式化与检查**

Run: `cd python && ruff format wbt tests && ruff check wbt tests`
Expected: 无 lint error

- [ ] **Step 5: 类型检查**

Run: `cd python && basedpyright wbt`
Expected: 无 error（warning 可接受）

- [ ] **Step 6: 提交质量修复（若有）**

```bash
git add -A
git status  # 确认仅 lint/format 修复
git commit -m "chore: lint/format fixes after migration" || echo "no fixes needed"
```

---

## 完成标准

- 所有 12 个任务的 checkbox 全部勾选
- `cargo test` 全绿
- `pytest python/tests` 全绿
- `cargo clippy -- -D warnings` 无 warning
- `ruff check` / `basedpyright` 无 error
- 顶层 `import wbt; wbt.<五个新名字>` 全部可用
