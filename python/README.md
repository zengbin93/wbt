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

Top-level imports (all reachable from `import wbt`):

```python
from wbt import (
    # Backtest engine
    WeightBacktest, backtest,
    # Performance metrics (Rust-backed)
    daily_performance,
    top_drawdowns,
    rolling_daily_performance,
    cal_yearly_days,
    # Strategy utilities (pure Python)
    weights_simple_ensemble,
    cal_trade_price,
    log_strategy_info,
    # Reporting
    generate_backtest_report,
    # Test data
    mock_symbol_kline, mock_weights,
)
```

Primary class and helpers:

- `WeightBacktest(...)`: main backtest engine entry.
- `backtest(...)`: convenience wrapper returning a `WeightBacktest`.
- `daily_performance(returns, yearly_days=252)`: standalone metric utility on a daily-return array.
- `top_drawdowns(returns, top=10)`: top-N drawdown windows.
- `rolling_daily_performance(df, ret_col, window=252, min_periods=100, yearly_days=None)`: rolling-window daily performance.
- `cal_yearly_days(dts)`: infer yearly trading-day count from a date series.
- `weights_simple_ensemble(df, weight_cols, method="mean", only_long=False, **kwargs)`: ensemble multiple strategy weights (`mean` / `vote` / `sum_clip`).
- `cal_trade_price(df, digits=None, windows=(5, 10, 15, 20, 30, 60))`: TWAP / VWAP and next-bar trade-price table grouped by symbol.
- `log_strategy_info(strategy, df)`: pretty-print per-symbol weight summaries via loguru.
- `generate_backtest_report(wb, output_path)`: render a self-contained HTML report.
- `mock_symbol_kline(...)` / `mock_weights(...)`: generators for quick experiments.

Core `WeightBacktest` properties and methods:

- `stats`, `long_stats`, `short_stats`
- `daily_return`, `long_daily_return`, `short_daily_return`
- `dailys`, `pairs`
- `alpha`, `alpha_stats`, `bench_stats`
- `segment_stats(sdt, edt, kind)`
- `long_alpha_stats`
- `get_symbol_daily(symbol)`, `get_symbol_pairs(symbol)`

### Logging Note

`cal_yearly_days` and `rolling_daily_performance` emit warnings from Rust (e.g. short-span fallback) via the `log` crate. The package initializes `pyo3-log` at module load, so those warnings show up through Python's standard `logging`. If you use loguru, install an [`InterceptHandler`](https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging) once to route them into your loguru sinks.

## Plotting Utilities

All plotting functions are **single-purpose figures** that consume a
`BacktestResult` (from `wb.to_result()`) with zero data transformation — each
field maps straight to a plotly trace. There are no composite (subplot) charts;
the HTML report composes single figures into a CSS grid instead.

```python
from wbt.plotting import (
    plot_colored_table,        # stats as a colored table
    plot_cumulative_returns,   # cumulative curves (voladj=True for vol-normalized)
    plot_daily_return_dist,    # daily-return histogram
    plot_drawdown,             # drawdown + cumulative (dual-axis single figure)
    plot_drawdowns_table,      # top-drawdowns detail table
    plot_key_trades,           # yearly best/worst key trades
    plot_monthly_heatmap,      # monthly-return heatmap
    plot_pairs_hold_dist,      # holding-bars distribution by direction
    plot_pairs_pnl_dist,       # pnl-ratio distribution by direction
    plot_stats_comparison,     # 多空/多头/空头/基准/超额 metric comparison table
    plot_symbol_returns,       # per-symbol cumulative returns
    plot_verdict,              # is_good_strategy verdict + yearly metrics
)
```

Typical usage:

```python
result = wb.to_result()

fig1 = plot_cumulative_returns(result, keys=["多空", "多头", "空头", "基准"])
fig2 = plot_cumulative_returns(result, voladj=True)   # vol-normalized
fig3 = plot_drawdown(result)
fig4 = plot_pairs_pnl_dist(result)

# Optional HTML export
html = plot_cumulative_returns(result, to_html=True)

# Full HTML report file (composes single figures into a tabbed CSS grid)
generate_backtest_report(df, "report.html")
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
|   |-- lib.rs                       # PyO3 bindings (pyfunctions, _wbt pymodule)
|   `-- core/
|       |-- cal_yearly_days.rs       # Rust core for cal_yearly_days
|       |-- daily_performance.rs
|       |-- rolling_daily_performance.rs
|       |-- top_drawdowns.rs
|       `-- ...                      # backtest engine internals
`-- python/
    `-- wbt/
        |-- __init__.py              # top-level exports
        |-- _df_convert.py           # pandas <-> Arrow IPC helpers
        |-- _wbt.pyi                 # Rust extension stubs
        |-- backtest.py              # WeightBacktest class
        |-- mock.py                  # mock_symbol_kline / mock_weights
        |-- top_drawdowns.py         # adapter for _wbt.top_drawdowns
        |-- utils/                   # adapters + pure-Python utilities
        |   |-- __init__.py
        |   |-- cal_yearly_days.py
        |   |-- rolling_daily_performance.py
        |   |-- weights_simple_ensemble.py
        |   |-- cal_trade_price.py
        |   `-- log_strategy_info.py
        |-- plotting/                # single-purpose plotly charts
        |   |-- __init__.py
        |   |-- _common.py
        |   |-- returns.py
        |   |-- risk.py
        |   |-- trades.py
        |   `-- overview.py
        `-- report/                  # HTML report + composite charts
            |-- __init__.py
            |-- _generator.py
            |-- _plot_backtest.py
            `-- html_builder.py
```

## License

[MIT](../LICENSE)
