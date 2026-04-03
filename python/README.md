# wbt Python Package

Python API for the wbt Rust backtesting engine.

[中文文档](README_CN.md)

## Development Objectives

This Python subproject aims to provide a practical research-facing interface for weight-based backtesting while keeping the heavy computation in Rust.

Design priorities:

1. Keep data input flexible for common research formats.
2. Return analysis-friendly outputs as pandas objects.
3. Preserve one consistent metric schema across stats outputs.
4. Provide plotting utilities that work directly on backtest outputs.

## Project Layout

This directory is an independent Python subproject.

```text
python/
|-- pyproject.toml
|-- README.md
|-- scripts/
|-- tests/
`-- wbt/
```

The Rust crate remains one level up at ../Cargo.toml. maturin builds the extension module from there.

## Installation And Local Setup

Requirements:

- Rust toolchain
- Python 3.10+
- uv

Setup:

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
```

## Quick Start

```python
import pandas as pd
from wbt import WeightBacktest

df = pd.DataFrame(
    {
        "dt": [
            "2024-01-02 09:01:00",
            "2024-01-02 09:02:00",
            "2024-01-02 09:03:00",
            "2024-01-02 09:04:00",
        ],
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        "weight": [0.5, 0.2, 0.0, -0.3],
        "price": [185.0, 186.0, 186.5, 184.5],
    }
)

wb = WeightBacktest(
    df,
    digits=2,
    fee_rate=0.0002,
    n_jobs=4,
    weight_type="ts",   # "ts" or "cs"
    yearly_days=252,
)

print("all:", wb.stats)
print("long:", wb.long_stats)
print("short:", wb.short_stats)

print(wb.daily_return.head())
print(wb.dailys.head())
print(wb.pairs.head())

print(wb.segment_stats("2024-01-01", "2024-12-31", kind="多空"))
print(wb.long_alpha_stats)
```

## Accepted Inputs

The data argument accepts:

- pandas.DataFrame
- polars.DataFrame
- polars.LazyFrame
- file path as str or Path

Supported file formats from path input:

- csv
- parquet
- feather
- arrow

Required columns:

| Column | Type | Meaning |
|---|---|---|
| dt | datetime-like | Bar end time |
| symbol | str | Instrument code |
| weight | float | Target position weight |
| price | float | Price used for return calculation |

Notes:

- Null values are not allowed.
- weight is rounded by digits before backtest.

## Main API Surface

Imports:

```python
from wbt import WeightBacktest, backtest, daily_performance
```

Primary class and helper:

- WeightBacktest(...): main entry.
- backtest(...): convenience wrapper returning WeightBacktest.
- daily_performance(...): standalone metric utility.

Core properties and methods:

- stats, long_stats, short_stats
- daily_return, long_daily_return, short_daily_return
- dailys, pairs
- alpha, alpha_stats, bench_stats
- segment_stats(sdt, edt, kind)
- long_alpha_stats
- get_symbol_daily(symbol), get_symbol_pairs(symbol)

## Plotting Utilities

All plotting helpers are under wbt.plotting.

```python
from wbt.plotting import (
    plot_backtest_overview,
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_long_short_comparison,
    plot_monthly_heatmap,
    plot_pairs_analysis,
    plot_symbol_returns,
)
```

Typical usage:

```python
fig1 = plot_cumulative_returns(wb.daily_return)
fig2 = plot_drawdown(wb.daily_return)
fig3 = plot_pairs_analysis(wb.pairs)

# Optional HTML export
html = plot_backtest_overview(wb.daily_return, to_html=True)
```

## Quality And Testing

Run checks from python/:

```bash
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## Architecture Snapshot

```text
repo-root/
|-- Cargo.toml
|-- src/
`-- python/
    `-- wbt/
        |-- __init__.py
        |-- _df_convert.py
        |-- _wbt.pyi
        |-- backtest.py
        `-- plotting/
            |-- __init__.py
            |-- _common.py
            |-- returns.py
            |-- risk.py
            |-- trades.py
            `-- overview.py
```

## License

[MIT](../LICENSE)
