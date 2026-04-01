# wbt — 持仓权重回测独立库设计

## 概述

将 `rs_czsc/crates/czsc-trader/src/weight_backtest/` 提取为完全独立的 Rust 库 `wbt`，同时支持 Rust crate 引用和 Python (`pip install`) 使用。

## 来源

- Rust 核心：`rs_czsc/crates/czsc-trader/src/weight_backtest/` (~2,947 行，9 文件)
- PyO3 绑定：`rs_czsc/python/src/trader/weight_backtest.rs` (~134 行)
- Python 封装：`rs_czsc/python/rs_czsc/_trader/weight_backtest.py` (~317 行)
- Arrow 转换：`rs_czsc/python/rs_czsc/_utils/_df_convert.py` (~47 行)

## 项目结构

```
wbt/
├── Cargo.toml              # workspace root
├── pyproject.toml           # maturin 构建配置
├── crates/
│   └── wbt-core/            # 纯 Rust 库 (lib crate, 无 PyO3)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs                # pub mod + re-export WeightBacktest
│           ├── errors.rs             # WbtError (thiserror, 不依赖 error-macros)
│           ├── utils.rs              # WeightType, RoundToNthDigit, min_max
│           ├── trade_dir.rs          # TradeDir, TradeAction
│           ├── backtest.rs           # do_backtest 编排
│           ├── native_engine.rs      # NativeEngine, DailysSoA, PairsSoA, LotsSoA
│           ├── calc_symbol.rs        # calc_all_dailys, calc_symbol_pairs
│           ├── evaluate_pairs.rs     # evaluate_pairs_soa
│           ├── report.rs             # Report, StatsReport, EvaluatePairs
│           └── daily_performance.rs  # 内联 daily_performance + top_drawdowns
├── src/                     # PyO3 cdylib (maturin 入口)
│   └── lib.rs               # #[pymodule] wbt: PyWeightBacktest, daily_performance
└── python/
    └── wbt/
        ├── __init__.py       # from wbt._wbt import PyWeightBacktest, daily_performance
        ├── _df_convert.py    # pandas_to_arrow_bytes, arrow_bytes_to_pd_df
        └── backtest.py       # WeightBacktest 封装类 (API 与原版完全一致)
```

## 关键决策

### 1. 错误处理：去掉 error-macros proc macro

原版使用 `CZSCErrorDerive` proc macro + `expand_error_chain` + `czsc_bail!`。

替代方案：直接用 `thiserror::Error` derive + `anyhow::Error` 处理意外错误。

```rust
#[derive(Debug, thiserror::Error)]
pub enum WbtError {
    #[error("expected value for {0}, got None")]
    NoneValue(String),
    #[error("polars: {0}")]
    Polars(#[from] polars::error::PolarsError),
    #[error("returns should not be empty")]
    ReturnsEmpty,
    #[error("{0}")]
    Unexpected(#[from] anyhow::Error),
}
```

**为什么**：proc macro 仅做 `From` 转换 + `Display` 格式化，`thiserror` 原生支持，无需自定义。

### 2. 内联依赖

| 原依赖 | 内联到 | 代码量 |
|--------|--------|--------|
| `czsc_utils::daily_performance` + `top_drawdowns` | `daily_performance.rs` | ~350 行 |
| `czsc_core::RoundToNthDigit` | `utils.rs` | ~20 行 |
| `czsc_core::min_max` | `utils.rs` | ~8 行 |
| `pyarrow_to_df` / `df_to_pyarrow` | `src/lib.rs` (PyO3 层) | ~15 行 |

### 3. wbt-core 依赖清单

```toml
[dependencies]
polars = { version = "0.46", features = ["lazy", "dtype-datetime", "dtype-date"] }
rayon = "1.10"
chrono = "0.4"
thiserror = "2"
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
strum = "0.26"
strum_macros = "0.26"
hashbrown = "0.15"
arrayvec = "0.7"
```

### 4. PyO3 绑定层依赖

```toml
[dependencies]
wbt-core = { path = "crates/wbt-core" }
pyo3 = { version = "0.25", features = ["extension-module"] }
polars = { version = "0.46", features = ["ipc"] }
```

### 5. Python API 兼容性

`python/wbt/backtest.py` 的 `WeightBacktest` 类完全复刻原版 API：

**构造函数**：`__init__(self, dfw, digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252)`

**属性/方法**（全部保留）：
- `stats` → dict
- `daily_return` → DataFrame (pivot, 含 total 列)
- `dailys` → DataFrame
- `alpha` → DataFrame
- `alpha_stats` → dict
- `bench_stats` → dict
- `long_daily_return` → DataFrame
- `short_daily_return` → DataFrame
- `long_stats` → dict
- `short_stats` → dict
- `pairs` → DataFrame
- `symbol_dict` → list
- `get_symbol_daily(symbol)` → DataFrame
- `get_symbol_pairs(symbol)` → DataFrame
- `get_top_symbols(n, kind)` → list

**关键**：`daily_performance` 也通过 PyO3 导出为 Python 函数，供 `alpha_stats`/`bench_stats`/`long_stats`/`short_stats` 调用。

### 6. maturin 构建

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "wbt"
requires-python = ">=3.9"
dependencies = ["pandas", "pyarrow"]

[tool.maturin]
features = ["python"]
python-source = "python"
module-name = "wbt._wbt"
```

Cargo.toml root 使用 feature flag 控制 PyO3：

```toml
[features]
default = []
python = ["pyo3"]
```

## Rust 使用方式

```rust
// Cargo.toml
// [dependencies]
// wbt-core = { git = "..." }  或 path

use wbt_core::WeightBacktest;
use polars::prelude::*;

let mut wb = WeightBacktest::new(dfw, 2, Some(0.0002))?;
wb.backtest(Some(4), WeightType::TS, 252)?;
let stats = wb.report.unwrap().stats;
```

## Python 使用方式

```python
from wbt import WeightBacktest

wb = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=4)
print(wb.stats)
print(wb.daily_return)
print(wb.alpha)
```

## 不做的事

- 不改动算法逻辑，逐文件迁移
- 不优化现有 Python API 接口（沿用 Arrow bytes）
- 不添加新功能
- 不添加 stub 生成（pyo3-stub-gen），后续按需加
