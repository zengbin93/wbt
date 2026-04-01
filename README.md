# wbt

High-performance position-weighted backtesting engine, written in Rust with Python bindings.

[中文文档](README_CN.md)

## Overview

**wbt** (Weight Back Test) is a standalone library for backtesting trading strategies based on position weights. It computes daily P&L attribution, trade pair matching (FIFO), and comprehensive performance statistics — all powered by a zero-copy, cache-friendly Rust core with rayon-based parallelism.

- **Rust crate** (`wbt-core`): pure Rust library, no Python dependency
- **Python package** (`wbt`): PyO3 bindings + pandas-friendly API via Arrow IPC

## Installation

### Python

```bash
# Requires: Rust toolchain, maturin, Python >= 3.9
git clone <repo-url> && cd wbt
uv venv .venv && source .venv/bin/activate
uv pip install pandas pyarrow numpy maturin
maturin develop --release
```

### Rust

```toml
# Cargo.toml
[dependencies]
wbt-core = { path = "path/to/wbt/crates/wbt-core" }
```

## Quick Start

### Python

```python
import pandas as pd
from wbt import WeightBacktest

# Prepare input: DataFrame with columns [dt, symbol, weight, price]
dfw = pd.DataFrame({
    "dt":     ["2024-01-02 09:01:00", "2024-01-02 09:02:00", "2024-01-03 09:01:00", "2024-01-03 09:02:00"],
    "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
    "weight": [0.5, 0.5, 0.0, 0.0],
    "price":  [185.0, 186.0, 187.0, 185.5],
})

wb = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts")

wb.stats          # dict: Sharpe, Calmar, max drawdown, win rate, etc.
wb.daily_return   # DataFrame: per-symbol daily returns with 'total' column
wb.dailys         # DataFrame: per-symbol daily detail (edge, return, cost, turnover, ...)
wb.alpha          # DataFrame: strategy excess return vs. benchmark
wb.pairs          # DataFrame: all matched trade pairs (FIFO)
```

### Rust

```rust
use wbt_core::{WeightBacktest, WeightType};
use polars::prelude::*;

let dfw: DataFrame = /* build your DataFrame with [dt, symbol, weight, price] */;
let mut wb = WeightBacktest::new(dfw, 2, Some(0.0002))?;
wb.backtest(Some(4), WeightType::TS, 252)?;

let report = wb.report.as_ref().unwrap();
println!("Sharpe: {}", report.stats.daily_performance.sharpe_ratio);
```

## Input Format

| Column   | Type     | Description                                                                 |
|----------|----------|-----------------------------------------------------------------------------|
| `dt`     | datetime | Bar end time; must be a continuous trading time series with no gaps          |
| `symbol` | str      | Instrument code                                                             |
| `weight` | float    | Position weight at bar end; independent across symbols; positive = long, negative = short, 0 = flat |
| `price`  | float    | Trade price (close, next-bar open, TWAP/VWAP, etc.)                         |

## Parameters

| Parameter     | Default  | Description                                         |
|---------------|----------|-----------------------------------------------------|
| `digits`      | `2`      | Decimal places for weight rounding                  |
| `fee_rate`    | `0.0002` | One-way transaction cost (commission + slippage)    |
| `n_jobs`      | `1`      | Number of parallel threads (rayon thread pool)      |
| `weight_type` | `"ts"`   | `"ts"` (time-series: equal-weight average across symbols) or `"cs"` (cross-section: sum across symbols) |
| `yearly_days` | `252`    | Trading days per year for annualization             |

## Output Properties

| Property            | Type      | Description                                           |
|---------------------|-----------|-------------------------------------------------------|
| `stats`             | dict      | Full performance report (Sharpe, Calmar, drawdown, win rate, trade metrics, ...) |
| `daily_return`      | DataFrame | Pivoted daily returns per symbol + `total` column     |
| `dailys`            | DataFrame | Per-symbol daily detail: `n1b`, `edge`, `return`, `cost`, `turnover`, long/short splits |
| `alpha`             | DataFrame | Strategy vs. benchmark excess return                  |
| `alpha_stats`       | dict      | Performance statistics on the excess return series    |
| `pairs`             | DataFrame | All FIFO-matched trade pairs with P&L, hold bars, direction |
| `long_daily_return` | DataFrame | Long-only daily returns                               |
| `short_daily_return`| DataFrame | Short-only daily returns                              |
| `long_stats`        | dict      | Long-only performance statistics                      |
| `short_stats`       | dict      | Short-only performance statistics                     |
| `bench_stats`       | dict      | Benchmark (equal-weight n1b) performance statistics   |

## Architecture

```
wbt/
├── crates/wbt-core/    # Pure Rust library
│   └── src/
│       ├── lib.rs                # WeightBacktest struct & public API
│       ├── native_engine.rs      # Zero-copy parallel engine (rayon)
│       ├── daily_performance.rs  # Sharpe, Calmar, drawdown, etc.
│       ├── evaluate_pairs.rs     # Trade pair statistics
│       ├── backtest.rs           # Orchestration & alpha computation
│       ├── calc_symbol.rs        # Per-symbol daily & pair calculation
│       ├── report.rs             # Report structs & JSON serialization
│       ├── trade_dir.rs          # Trade direction & action enums
│       ├── errors.rs             # Error types
│       └── utils.rs              # WeightType, rounding, quantile
├── src/lib.rs          # PyO3 bindings (Arrow IPC in/out)
└── python/wbt/         # Python API wrapper
    ├── backtest.py     # WeightBacktest class (pandas-friendly)
    └── _df_convert.py  # Arrow <-> pandas conversion
```

### Performance Design

- **O(N) counting sort** for symbol grouping (replaces generic sort)
- **Struct-of-Arrays (SoA)** layout for cache-locality in hot loops
- **Lazy DataFrame materialization** — internal computation uses raw vectors; DataFrames built only on demand
- **FIFO lot matching** via stack-based SoA with dynamic resizing
- **rayon** data-parallelism across symbols with configurable thread pool
- **Zero-copy slicing** from Polars DataFrames into contiguous memory blocks
- **Arrow IPC** for zero-serialization-overhead Python <-> Rust data transfer

## License

[MIT](LICENSE)
