# wbt

Position-weighted backtesting engine for quantitative strategies, with a Rust core and Python bindings.

[中文说明](README_CN.md)

## Why This Project Exists

Many strategy teams use position weights as the canonical interface between signal generation and execution simulation. Existing backtesting tools often focus on order-level simulation or are too slow for large, multi-symbol weight datasets.

The goals of wbt are:

1. Keep one consistent data contract for weight-based strategies.
2. Provide fast and deterministic computation with Rust.
3. Expose a Python-first API for research workflows.
4. Offer built-in evaluation outputs and plotting-ready data structures.

## What wbt Is Good At

- Time-series and cross-sectional weight backtests.
- Multi-symbol daily performance attribution.
- Long/short decomposition and segment-level metrics.
- High-throughput computation from pandas, polars, or file inputs.

## What wbt Is Not Trying To Solve

- Tick-level order book simulation.
- Exchange matching-engine microstructure.
- Broker-specific execution modeling.

If your strategy logic is naturally represented as target weights over time, wbt is a strong fit.

## Repository Layout

- Rust crate: repository root
- Python package: python/

```text
wbt/
|-- Cargo.toml
|-- src/
`-- python/
    |-- pyproject.toml
    |-- README.md
    |-- tests/
    `-- wbt/
```

## Quick Start (Python Users)

The Python package is in python/ and keeps the import path as import wbt.

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
uv run pytest -v
```

Then in Python:

```python
import pandas as pd
from wbt import WeightBacktest

df = pd.DataFrame(
    {
        "dt": ["2024-01-02 09:01:00", "2024-01-02 09:02:00", "2024-01-02 09:03:00"],
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "weight": [0.5, 0.0, -0.3],
        "price": [185.0, 186.0, 184.5],
    }
)

wb = WeightBacktest(df, digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts")

print(wb.stats)
print(wb.long_stats)
print(wb.short_stats)
```

For complete Python guide, see python/README.md.

## Quick Start (Rust Developers)

Run tests from repository root:

```bash
cargo test
```

Use as dependency:

```toml
[dependencies]
wbt = "0.1"
```

## Data Contract (Core Idea)

wbt expects four essential columns:

- dt: bar end timestamp
- symbol: instrument identifier
- weight: target position weight at bar end
- price: trade/mark price

Accepted Python inputs:

- pandas.DataFrame
- polars.DataFrame or polars.LazyFrame
- file path (csv, parquet, feather, arrow)

## Outputs You Can Use Immediately

- wb.stats: full long-short evaluation summary.
- wb.long_stats and wb.short_stats: directional breakdown.
- wb.daily_return and wb.dailys: daily series for analytics.
- wb.alpha and wb.alpha_stats: strategy-vs-benchmark excess analysis.
- wb.pairs: trade-pair table for per-trade evaluation.
- wb.segment_stats(...): metrics for arbitrary date windows.
- wb.long_alpha_stats: volatility-scaled long-side alpha metrics.

## Standalone Utility Functions

Beyond the `WeightBacktest` class, wbt exposes several stand-alone helpers at the top level:

- `daily_performance(returns, yearly_days=252)`: full performance metrics on a daily return series (Rust core).
- `top_drawdowns(returns, top=10)`: top-N drawdown windows (Rust core).
- `rolling_daily_performance(df, ret_col, window=252, min_periods=100, yearly_days=None)`: rolling-window daily performance (Rust core).
- `cal_yearly_days(dts)`: infer yearly trading-day count from a date series (Rust core).
- `weights_simple_ensemble(df, weight_cols, method="mean", only_long=False)`: ensemble multiple strategy weights (`mean` / `vote` / `sum_clip`).
- `cal_trade_price(df, digits=None, windows=(5, 10, 15, 20, 30, 60))`: TWAP / VWAP and next-bar trade-price table grouped by symbol.
- `log_strategy_info(strategy, df)`: pretty-print per-symbol weight summaries via loguru.
- `mock_symbol_kline(...)` / `mock_weights(...)`: generators for quick experiments.

The Rust-backed helpers emit warnings (e.g. short-span fallback in `cal_yearly_days`) via the `log` crate; `pyo3-log` bridges them into Python's standard `logging` module, so any loguru `InterceptHandler` setup will receive them transparently.

## HTML Report Generation

`wbt.generate_backtest_report(wb, output_path)` produces a self-contained HTML report combining the `wbt.report._plot_backtest` chart family (cumulative returns, drawdown analysis, daily return distribution, monthly heatmap, backtest stats overview, colored metric table, long/short comparison).

## Plotting

Two plotting surfaces are available in the Python package:

- `wbt.plotting`: focused single-purpose figures — `plot_cumulative_returns`, `plot_drawdown`, `plot_daily_return_dist`, `plot_monthly_heatmap`, `plot_symbol_returns`, `plot_pairs_analysis`, `plot_backtest_overview`, `plot_colored_table`, `plot_long_short_comparison`.
- `wbt.report`: report-grade composite charts (used internally by `generate_backtest_report`) — `plot_backtest_stats`, `plot_drawdown_analysis`, `plot_daily_return_distribution`, plus reusable helpers like `HtmlReportBuilder` and `get_performance_metrics_cards`.

## Development Workflow

- Rust checks run from repository root.
- Python checks run from python/.
- CI validates both layers.

Typical local quality checks:

```bash
# repository root
cargo test

# python subproject
cd python
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## Related Docs

- English Python guide: python/README.md
- Chinese Python guide: python/README_CN.md
- Design notes: docs/desgin.md

## License

[MIT](LICENSE)
