# wbt Standalone Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract weight_backtest from rs_czsc into a standalone Rust+Python library at `/Users/0xjun/Documents/vsPro/wbt`.

**Architecture:** Workspace with `wbt-core` (pure Rust lib) and root crate (PyO3 cdylib via maturin). All czsc-* dependencies inlined. Python wrapper replicates original API exactly.

**Tech Stack:** Rust 1.94, Polars 0.46, PyO3 0.25, maturin 1.12, rayon, chrono, thiserror, anyhow

---

## File Structure

```
wbt/
├── Cargo.toml                    # workspace root + PyO3 cdylib crate
├── pyproject.toml                # maturin config
├── .gitignore                    # updated for Rust+Python
├── crates/
│   └── wbt-core/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── errors.rs
│           ├── utils.rs          # WeightType, RoundToNthDigit, min_max, Quantile
│           ├── trade_dir.rs
│           ├── daily_performance.rs
│           ├── evaluate_pairs.rs
│           ├── native_engine.rs
│           ├── calc_symbol.rs
│           ├── backtest.rs
│           └── report.rs
├── src/
│   └── lib.rs                    # PyO3 bindings
└── python/
    └── wbt/
        ├── __init__.py
        ├── _df_convert.py
        └── backtest.py
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/wbt-core/Cargo.toml`
- Create: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Create workspace root Cargo.toml**

```toml
[workspace]
members = ["crates/wbt-core"]

[package]
name = "wbt"
version = "0.1.0"
edition = "2024"

[lib]
name = "_wbt"
crate-type = ["cdylib"]

[dependencies]
wbt-core = { path = "crates/wbt-core" }
pyo3 = { version = "0.25", features = ["extension-module"] }
numpy = "0.25"
polars = { version = "0.46", default-features = false, features = ["ipc"] }

[workspace.dependencies]
polars = { version = "0.46", default-features = false }
```

- [ ] **Step 2: Create wbt-core Cargo.toml**

```toml
[package]
name = "wbt-core"
version = "0.1.0"
edition = "2024"

[dependencies]
polars = { workspace = true, features = ["lazy", "dtype-datetime", "dtype-date", "round_series", "pearson_corr"] }
rayon = "1.10"
chrono = "0.4"
thiserror = "2"
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
strum = "0.26"
strum_macros = "0.26"
hashbrown = "0.15"
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "wbt"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["pandas", "pyarrow", "numpy"]

[tool.maturin]
python-source = "python"
module-name = "wbt._wbt"
features = []
```

- [ ] **Step 4: Update .gitignore for Rust+Python**

Append Rust patterns to existing Python .gitignore:

```
# Rust
/target/
Cargo.lock
```

- [ ] **Step 5: Create directory structure**

```bash
mkdir -p crates/wbt-core/src src python/wbt
```

- [ ] **Step 6: Commit scaffolding**

```bash
git add -A
git commit -m "feat: project scaffolding for wbt standalone library"
```

---

### Task 2: wbt-core - errors.rs & utils.rs (基础模块)

**Files:**
- Create: `crates/wbt-core/src/errors.rs`
- Create: `crates/wbt-core/src/utils.rs`

- [ ] **Step 1: Create errors.rs**

Replace `CZSCErrorDerive` + `expand_error_chain` with plain `thiserror`:

```rust
use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WbtError {
    #[error("expected value for {0}, got None")]
    NoneValue(String),

    #[error("polars: {0}")]
    Polars(#[from] PolarsError),

    #[error("returns should not be empty")]
    ReturnsEmpty,

    #[error("{0:#}")]
    Unexpected(#[from] anyhow::Error),
}
```

- [ ] **Step 2: Create utils.rs**

Inline `RoundToNthDigit`, `min_max`, `Quantile`, and `WeightType`:

```rust
use strum_macros::{AsRefStr, Display, EnumString};

#[derive(Debug, Clone, Copy, PartialEq, EnumString, AsRefStr, Display)]
pub enum WeightType {
    #[strum(serialize = "ts")]
    TS,
    #[strum(serialize = "cs")]
    CS,
}

pub trait RoundToNthDigit {
    fn round_to_nth_digit(&self, nth: usize) -> Self;
    fn round_to_2_digit(&self) -> Self;
    fn round_to_3_digit(&self) -> Self;
    fn round_to_4_digit(&self) -> Self;
}

impl RoundToNthDigit for f64 {
    fn round_to_nth_digit(&self, nth: usize) -> f64 {
        let scale = 10_f64.powi(nth as i32);
        (self * scale).round() / scale
    }
    fn round_to_2_digit(&self) -> f64 { self.round_to_nth_digit(2) }
    fn round_to_3_digit(&self) -> f64 { self.round_to_nth_digit(3) }
    fn round_to_4_digit(&self) -> f64 { self.round_to_nth_digit(4) }
}

pub fn min_max(x: f64, min_val: f64, max_val: f64) -> f64 {
    if x < min_val { min_val } else if x > max_val { max_val } else { x }
}

pub trait Quantile {
    fn quantile(&self, q: f64) -> Option<f64>;
}

impl Quantile for [f64] {
    fn quantile(&self, q: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&q) { return None; }
        let n = self.len();
        if n == 0 { return None; }
        let mut sorted = self.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = q * (n as f64 - 1.0);
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let fraction = pos - lower as f64;
        if lower == upper { Some(sorted[lower]) }
        else { Some(sorted[lower] + fraction * (sorted[upper] - sorted[lower])) }
    }
}
```

- [ ] **Step 3: Verify compilation**

```bash
# Create minimal lib.rs first
echo 'pub mod errors;\npub mod utils;' > crates/wbt-core/src/lib.rs
cd /Users/0xjun/Documents/vsPro/wbt && cargo check -p wbt-core
```

Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add crates/wbt-core/src/errors.rs crates/wbt-core/src/utils.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add errors and utils modules"
```

---

### Task 3: wbt-core - trade_dir.rs

**Files:**
- Create: `crates/wbt-core/src/trade_dir.rs`

- [ ] **Step 1: Create trade_dir.rs**

Copy from source verbatim. Only change: remove `use czsc_trader::` doc paths. Keep all enums, impls, static strings, and tests exactly as-is from the original `/Users/0xjun/Documents/cursorPro/rs_czsc/crates/czsc-trader/src/weight_backtest/trade_dir.rs`.

The file is self-contained — no czsc-* imports needed.

- [ ] **Step 2: Add module to lib.rs and verify**

Add `pub mod trade_dir;` to lib.rs.

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo test -p wbt-core
```

Expected: 3 trade_dir tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/wbt-core/src/trade_dir.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add trade_dir module"
```

---

### Task 4: wbt-core - daily_performance.rs (内联)

**Files:**
- Create: `crates/wbt-core/src/daily_performance.rs`

- [ ] **Step 1: Create daily_performance.rs**

Inline the following into one file:
- `DailyPerformance` struct + Default impl (from `czsc-utils/src/daily_performance.rs`)
- `daily_performance()` function (same file, lines 99-317)
- `daily_performance_drawdown()` function (same file, lines 319-358)
- `calc_underwater()`, `calc_underwater_valley()`, `calc_underwater_peak()`, `calc_underwater_recovery()` (from `czsc-utils/src/top_drawdowns.rs`, lines 1-50)

**Import replacements:**
- `use crate::errors::UtilsError;` → `use crate::errors::WbtError;`
- `use czsc_core::utils::rounded::{RoundToNthDigit, min_max};` → `use crate::utils::{RoundToNthDigit, min_max};`
- All `UtilsError` → `WbtError`
- `UtilsError::ReturnsEmpty` → `WbtError::ReturnsEmpty`

Also bring over the `test_daily_performance` test from the source.

- [ ] **Step 2: Add module and verify**

Add `pub mod daily_performance;` to lib.rs.

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo test -p wbt-core -- daily_performance
```

Expected: test passes.

- [ ] **Step 3: Commit**

```bash
git add crates/wbt-core/src/daily_performance.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add daily_performance module (inlined from czsc-utils)"
```

---

### Task 5: wbt-core - evaluate_pairs.rs

**Files:**
- Create: `crates/wbt-core/src/evaluate_pairs.rs`

- [ ] **Step 1: Create evaluate_pairs.rs**

Copy from source `/Users/0xjun/Documents/cursorPro/rs_czsc/crates/czsc-trader/src/weight_backtest/evaluate_pairs.rs`.

**Import replacements:**
- `use super::{errors::WeightBackTestError, trade_dir::TradeDir};` → `use crate::{errors::WbtError, trade_dir::TradeDir};`
- `use crate::weight_backtest::native_engine::PairsSoA;` → `use crate::native_engine::PairsSoA;`
- `use czsc_core::utils::rounded::RoundToNthDigit;` → `use crate::utils::RoundToNthDigit;`
- All `WeightBackTestError` → `WbtError`

Note: This file references `PairsSoA` which will be created in Task 7. For now, add the module declaration but allow it to be checked after Task 7.

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod evaluate_pairs;` to lib.rs. Do NOT cargo check yet — depends on native_engine.

- [ ] **Step 3: Commit (will verify after Task 7)**

```bash
git add crates/wbt-core/src/evaluate_pairs.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add evaluate_pairs module"
```

---

### Task 6: wbt-core - report.rs

**Files:**
- Create: `crates/wbt-core/src/report.rs`

- [ ] **Step 1: Create report.rs**

Copy from source `/Users/0xjun/Documents/cursorPro/rs_czsc/crates/czsc-trader/src/weight_backtest/report.rs`.

**Import replacements:**
- `use super::evaluate_pairs::EvaluatePairs;` → `use crate::evaluate_pairs::EvaluatePairs;`
- `use crate::weight_backtest::native_engine::DailyTotals;` → `use crate::native_engine::DailyTotals;`
- `use czsc_utils::daily_performance::DailyPerformance;` → `use crate::daily_performance::DailyPerformance;`

Everything else (struct definitions, From<Report> for Value, From<StatsReport> for Value) stays the same.

- [ ] **Step 2: Add module to lib.rs**

Add `pub mod report;` to lib.rs. Do NOT cargo check yet — depends on native_engine.

- [ ] **Step 3: Commit**

```bash
git add crates/wbt-core/src/report.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add report module"
```

---

### Task 7: wbt-core - native_engine.rs

**Files:**
- Create: `crates/wbt-core/src/native_engine.rs`

- [ ] **Step 1: Create native_engine.rs**

Copy from source `/Users/0xjun/Documents/cursorPro/rs_czsc/crates/czsc-trader/src/weight_backtest/native_engine.rs` (1052 lines).

**Import replacements:**
- `use crate::weight_backtest::errors::WeightBackTestError;` → `use crate::errors::WbtError;`
- `use crate::weight_backtest::report::SymbolsReport;` → `use crate::report::SymbolsReport;`
- `use crate::weight_backtest::trade_dir::{TradeAction, TradeDir};` → `use crate::trade_dir::{TradeAction, TradeDir};`
- All `WeightBackTestError` → `WbtError`

All structs (`DailyTotals`, `DailysSoA`, `PairsSoA`, `LotsSoA`, `SymbolDailysSoA`, `SymbolPairsSoA`, `NativeEngine`) and their impls stay identical. Keep all inline functions (`dt_to_date_key_fast`, `dt_to_day_ordinal`, `dt_to_days_since_epoch`).

- [ ] **Step 2: Add module and verify compilation**

Add `pub mod native_engine;` to lib.rs.

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo check -p wbt-core
```

Expected: compiles (first time all modules can resolve cross-references).

- [ ] **Step 3: Run tests**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo test -p wbt-core
```

Expected: all existing tests pass (trade_dir + daily_performance + native_engine tests).

- [ ] **Step 4: Commit**

```bash
git add crates/wbt-core/src/native_engine.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add native_engine module"
```

---

### Task 8: wbt-core - calc_symbol.rs

**Files:**
- Create: `crates/wbt-core/src/calc_symbol.rs`

- [ ] **Step 1: Create calc_symbol.rs**

Copy from source `/Users/0xjun/Documents/cursorPro/rs_czsc/crates/czsc-trader/src/weight_backtest/calc_symbol.rs`.

**Import replacements:**
- `use super::WeightBacktest;` → `use crate::WeightBacktest;`
- `use crate::weight_backtest::TradeDir;` → `use crate::trade_dir::TradeDir;`
- `use crate::weight_backtest::{WeightBackTestError, trade_dir::TradeAction};` → `use crate::{errors::WbtError, trade_dir::TradeAction};`
- All `WeightBackTestError` → `WbtError`

Keep all the `impl WeightBacktest` methods: `calc_all_dailys`, `calc_symbol_pairs`, `get_symbol_str_from_a_symbol_df`.

- [ ] **Step 2: Add module to lib.rs**

Add `mod calc_symbol;` to lib.rs (private module, methods are on WeightBacktest).

- [ ] **Step 3: Commit**

```bash
git add crates/wbt-core/src/calc_symbol.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add calc_symbol module"
```

---

### Task 9: wbt-core - backtest.rs

**Files:**
- Create: `crates/wbt-core/src/backtest.rs`

- [ ] **Step 1: Create backtest.rs**

Copy from source `/Users/0xjun/Documents/cursorPro/rs_czsc/crates/czsc-trader/src/weight_backtest/backtest.rs`.

**Import replacements:**
- The entire `use super::` block → use crate-level imports:
  ```rust
  use crate::{
      WeightBacktest,
      errors::WbtError,
      evaluate_pairs::evaluate_pairs_soa,
      report::{Report, StatsReport, SymbolsReport},
      utils::{WeightType, RoundToNthDigit},
      native_engine::{DailyTotals, DailysSoA, PairsSoA},
      trade_dir::TradeDir,
  };
  ```
- `use czsc_core::utils::rounded::RoundToNthDigit;` → already covered above
- `use czsc_utils::daily_performance::daily_performance;` → `use crate::daily_performance::daily_performance;`
- `use crate::weight_backtest::native_engine::*;` → `use crate::native_engine::*;`
- `use crate::weight_backtest::trade_dir::TradeDir;` → already covered
- `czsc_bail!(...)` → `return Err(anyhow::anyhow!(...).into())`
- All `WeightBackTestError` → `WbtError`

Keep: `date_key_to_naive_date`, `pearson_corr_inline`, `std_inline`, `do_backtest`, `process_symbols`.

- [ ] **Step 2: Add module to lib.rs**

Add `mod backtest;` to lib.rs (private — methods are on WeightBacktest).

- [ ] **Step 3: Commit**

```bash
git add crates/wbt-core/src/backtest.rs crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): add backtest module"
```

---

### Task 10: wbt-core - lib.rs (主入口 + WeightBacktest 结构体)

**Files:**
- Modify: `crates/wbt-core/src/lib.rs`

- [ ] **Step 1: Write complete lib.rs**

Migrate from source `mod.rs`. This contains the `WeightBacktest` struct definition, `new()`, `backtest()`, `dailys_df()`, `pairs_df()`, `alpha_df()`, and the `utils.rs` methods (`unique_symbols`, `convert_datetime`, `round_weight`, `calc_stats_by_alpha`, `alpha`).

```rust
pub mod errors;
pub mod utils;
pub mod trade_dir;
pub mod daily_performance;
pub mod evaluate_pairs;
pub mod native_engine;
pub mod report;
mod backtest;
mod calc_symbol;

use anyhow::Context;
use errors::WbtError;
use native_engine::{DailysSoA, PairsSoA};
use polars::prelude::*;
use report::Report;

pub use utils::WeightType;

pub struct WeightBacktest {
    pub dfw: DataFrame,
    pub digits: i64,
    pub fee_rate: f64,
    pub symbols: Vec<Arc<str>>,
    dailys_soa: Option<DailysSoA>,
    pairs_soa: Option<PairsSoA>,
    dailys_cache: Option<DataFrame>,
    pairs_cache: Option<DataFrame>,
    pub report: Option<Report>,
}
```

Then impl blocks for `new()`, `backtest()`, `dailys_df()`, `pairs_df()`, `alpha_df()` — migrated from `mod.rs`.

Also move the `unique_symbols`, `convert_datetime`, `round_weight`, `calc_stats_by_alpha`, `alpha` methods from `utils.rs` source into a separate `impl WeightBacktest` block in lib.rs (or keep them in a private module).

**Key replacement in `convert_datetime`:**
- `czsc_bail!("Unsupported datetime type: {:?}", dt_type)` → `return Err(anyhow::anyhow!("Unsupported datetime type: {:?}", dt_type).into())`

Also move the `#[cfg(test)] mod tests` from `mod.rs` with `raw_example_data()` and the 3 unit tests.

- [ ] **Step 2: Full compilation check**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo check -p wbt-core
```

Expected: compiles.

- [ ] **Step 3: Run all tests**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo test -p wbt-core
```

Expected: all tests pass (trade_dir::3, daily_performance, native_engine, mod::3).

- [ ] **Step 4: Commit**

```bash
git add crates/wbt-core/src/lib.rs
git commit -m "feat(wbt-core): complete WeightBacktest struct and public API"
```

---

### Task 11: PyO3 Bindings (src/lib.rs)

**Files:**
- Create: `src/lib.rs`

- [ ] **Step 1: Create PyO3 bindings**

Migrate from `/Users/0xjun/Documents/cursorPro/rs_czsc/python/src/trader/weight_backtest.rs` + `/python/src/utils/daily_performance.rs` + `/python/src/utils/df_convert.rs`.

```rust
use std::io::Cursor;
use std::str::FromStr;

use numpy::PyReadonlyArray1;
use pyo3::{
    prelude::*,
    types::{PyBytes, PyBytesMethods, PyDict},
    exceptions::PyException,
};
use polars::prelude::*;
use wbt_core::{WeightBacktest, WeightType};

// --- Arrow conversion (inlined from df_convert.rs) ---

fn pyarrow_to_df(data: &[u8]) -> PyResult<DataFrame> {
    let cursor = Cursor::new(data);
    IpcReader::new(cursor)
        .finish()
        .map_err(|e| PyException::new_err(e.to_string()))
}

fn df_to_pyarrow(dataframe: &mut DataFrame) -> PyResult<Vec<u8>> {
    let mut buffer = Cursor::new(Vec::new());
    IpcWriter::new(&mut buffer)
        .finish(dataframe)
        .map_err(|e| PyException::new_err(e.to_string()))?;
    Ok(buffer.into_inner())
}

// --- PyWeightBacktest ---

#[pyclass(module = "wbt._wbt")]
pub struct PyWeightBacktest {
    inner: WeightBacktest,
}

#[pymethods]
impl PyWeightBacktest {
    #[staticmethod]
    #[pyo3(signature = (data, digits=2, fee_rate=Some(0.0002), n_jobs=Some(4), weight_type="ts", yearly_days=252))]
    fn from_arrow<'py>(
        py: Python<'py>,
        data: Bound<'py, PyBytes>,
        digits: i64,
        fee_rate: Option<f64>,
        n_jobs: Option<usize>,
        weight_type: &str,
        yearly_days: usize,
    ) -> PyResult<Self> {
        let data = data.as_bytes();
        let df = pyarrow_to_df(data)?;
        let weight_type_enum = WeightType::from_str(weight_type).unwrap_or(WeightType::TS);

        let mut inner = WeightBacktest::new(df, digits, fee_rate)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        py.allow_threads(|| {
            inner.backtest(n_jobs, weight_type_enum, yearly_days)
                .map_err(|e| PyException::new_err(e.to_string()))
        })?;
        Ok(Self { inner })
    }

    // stats(), daily_return(), dailys(), alpha(), pairs(), symbol_dict()
    // — exact same as source, replacing PythonError with PyException::new_err
}

// --- daily_performance Python function ---

#[pyfunction]
#[pyo3(signature = (daily_returns, yearly_days=None))]
fn daily_performance<'py>(
    py: Python<'py>,
    daily_returns: PyReadonlyArray1<'py, f64>,
    yearly_days: Option<usize>,
) -> PyResult<PyObject> {
    let daily_returns = daily_returns.as_slice()
        .map_err(|e| PyException::new_err(e.to_string()))?;
    let dp = wbt_core::daily_performance::daily_performance(daily_returns, yearly_days)
        .map_err(|e| PyException::new_err(e.to_string()))?;

    let py_dict = PyDict::new(py);
    // ... set all 17 keys exactly as source ...
    Ok(py_dict.into())
}

// --- Module ---

#[pymodule]
fn _wbt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWeightBacktest>()?;
    m.add_function(wrap_pyfunction!(daily_performance, m)?)?;
    Ok(())
}
```

Complete the `stats()`, `daily_return()`, `dailys()`, `alpha()`, `pairs()`, `symbol_dict()` methods — each one is a direct port from the source with `PythonError` → `PyException::new_err(e.to_string())`.

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo check
```

Expected: compiles (including PyO3 cdylib).

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs
git commit -m "feat: add PyO3 bindings for Python module"
```

---

### Task 12: Python Wrapper

**Files:**
- Create: `python/wbt/__init__.py`
- Create: `python/wbt/_df_convert.py`
- Create: `python/wbt/backtest.py`

- [ ] **Step 1: Create __init__.py**

```python
from wbt.backtest import WeightBacktest
from wbt._wbt import daily_performance

__all__ = ["WeightBacktest", "daily_performance"]
```

- [ ] **Step 2: Create _df_convert.py**

Copy verbatim from `/Users/0xjun/Documents/cursorPro/rs_czsc/python/rs_czsc/_utils/_df_convert.py`.

```python
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import Union

def pandas_to_arrow_bytes(df: Union[pd.DataFrame, pd.Series]) -> bytes:
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()

def arrow_bytes_to_pd_df(arrow_bytes: bytes) -> pd.DataFrame:
    buffer = pa.BufferReader(arrow_bytes)
    with ipc.open_file(buffer) as reader:
        table = reader.read_all()
    return table.to_pandas()
```

- [ ] **Step 3: Create backtest.py**

Copy from `/Users/0xjun/Documents/cursorPro/rs_czsc/python/rs_czsc/_trader/weight_backtest.py`.

**Import replacements only:**
- `from rs_czsc._rs_czsc import PyWeightBacktest, daily_performance` → `from wbt._wbt import PyWeightBacktest, daily_performance`
- `from rs_czsc._utils._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes` → `from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes`

All class body, methods, properties, docstrings — **identical** to original.

- [ ] **Step 4: Commit**

```bash
git add python/wbt/__init__.py python/wbt/_df_convert.py python/wbt/backtest.py
git commit -m "feat: add Python wrapper with identical API to original"
```

---

### Task 13: Build & Integration Test

- [ ] **Step 1: Build with maturin (develop mode)**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && maturin develop
```

Expected: builds successfully, installs wbt into current Python environment.

- [ ] **Step 2: Verify Python import**

```bash
python3 -c "from wbt import WeightBacktest, daily_performance; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 3: Run Rust tests**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo test
```

Expected: all tests pass.

- [ ] **Step 4: Fix any compilation/import issues**

If any errors occur, diagnose and fix. Common issues:
- Polars API changes between 0.42 and 0.46 (e.g., `Series::new` may need `PlSmallStr` or `&str`)
- `cont_slice` → may need `as_slice()` in newer Polars
- `StringChunkedBuilder` API changes

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "fix: resolve build issues and verify integration"
```

---

### Task 14: Polars 0.46 API Compatibility Fixes

> This task exists because the source uses Polars 0.42 and we target 0.46. Known breaking changes need to be addressed.

**Likely changes needed:**

- [ ] **Step 1: Audit and fix Series::new signature**

Polars 0.46 may require `PlSmallStr` instead of `&str` for column names. If so, use `Series::new(PlSmallStr::from("name"), ...)` or the `column` macro.

Check: `cargo check -p wbt-core 2>&1 | head -50`

- [ ] **Step 2: Audit ChunkedArray::cont_slice**

In newer Polars, `cont_slice()` may be renamed or require different access. Check if `as_slice()` or `to_vec()` is needed.

- [ ] **Step 3: Audit StringChunkedBuilder**

The builder API may have changed. Check `StringChunkedBuilder::new` signature.

- [ ] **Step 4: Audit DataFrame::new and column access**

`df.column("name")` may return `&Column` instead of `&Series` in newer Polars. May need `.as_series()` or `.as_materialized_series()`.

- [ ] **Step 5: Fix all issues and verify tests pass**

```bash
cd /Users/0xjun/Documents/vsPro/wbt && cargo test -p wbt-core && cargo check
```

- [ ] **Step 6: Commit fixes**

```bash
git add -A
git commit -m "fix: polars 0.46 API compatibility"
```
