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
- wb.is_good_strategy(mode="history" | "recent", ...): objective verdict on whether a strategy is worth pursuing. Returns a dict with `is_good` (bool), `reason`, per-year breakdown (history mode) or recent-window metrics (recent mode), and condition flags. Adjustable parameters: `target_vol`, `max_dd_threshold`, `min_year_days`, `recent_days`. In `recent` mode, the historical max drawdown is computed on the segment **excluding** the recent window, so the two never overlap by construction.

## Standalone Utility Functions

Beyond the `WeightBacktest` class, wbt exposes several stand-alone helpers at the top level:

- `daily_performance(returns, yearly_days=252)`: full performance metrics on a daily return series (Rust core).
- `top_drawdowns(returns, top=10)`: top-N drawdown windows (Rust core).
- `rolling_daily_performance(df, ret_col, window=252, min_periods=100, yearly_days=None)`: rolling-window daily performance (Rust core).
- `cal_yearly_days(dts)`: infer yearly trading-day count from a date series (Rust core).
- `weights_simple_ensemble(df, weight_cols, method="mean", only_long=False, **kwargs)`: ensemble multiple strategy weights (`mean` / `vote` / `sum_clip`). Returns a new DataFrame (input `df` is not mutated). `sum_clip` mode additionally accepts `clip_min=-1, clip_max=1` via kwargs.
- `cal_trade_price(df, digits=None, **kwargs)`: TWAP / VWAP and next-bar trade-price table grouped by symbol. Accepts `windows=(5, 10, 15, 20, 30, 60)` and `copy=True` via kwargs.
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

## Related Open-Source Projects

wbt sits in a small ecosystem of quantitative-research tools. The most closely related projects:

- [**czsc**](https://github.com/waditu/czsc) — A comprehensive Python framework for Chan Theory (缠论) quantitative trading: signals, strategies, traders, EDA, and plotting. Since v1.0.x its core algorithms are implemented in Rust and exposed via PyO3 (`czsc._native`). **Relation to wbt:** wbt migrated 5 evaluation/utility functions from czsc (`cal_yearly_days`, `rolling_daily_performance`, `weights_simple_ensemble`, `cal_trade_price`, `log_strategy_info`) and keeps numerical results aligned with the czsc reference (see `python/tests/test_compare_with_czsc_script.py`). czsc strategies naturally emit the weight tables that wbt consumes.

- [**wmr**](https://github.com/zengbin93/wmr) — A strategy weight management system backed by ClickHouse and DuckDB, focused on persisting, versioning, and querying per-strategy position weights at scale. **Relation to wbt:** wmr is the data layer for weight tables (storage / retrieval); wbt is the compute layer that turns those tables into backtest metrics, daily series, and HTML reports.

- [**talib-rs**](https://github.com/0xcjun/talib-rs) — A pure-Rust technical-analysis library, designed as a drop-in replacement for the classic C TA-Lib (bit-exact results, SIMD-accelerated, no C dependency). **Relation to wbt:** a peer project on the Rust side — wbt focuses on weight-driven backtesting and performance metrics, while talib-rs covers canonical TA indicators. The two compose well when a strategy needs both indicator computation and weight-based backtesting inside the same Rust/Python pipeline.

Together they sketch a typical research-to-evaluation pipeline: **czsc** (signals & strategies) → **wmr** (weight storage) → **wbt** (backtest & metrics), with **talib-rs** providing reusable Rust-native indicator computation along the way.

## License

[MIT](LICENSE)
