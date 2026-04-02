# wbt Python Package

Python bindings for the `wbt` Rust backtesting engine.

[中文文档](README_CN.md)

## Project Layout

This directory is the standalone Python subproject.

```text
python/
|-- pyproject.toml
|-- README.md
|-- docs/
|-- tests/
`-- wbt/
```

The Rust crate remains one level up at `../Cargo.toml`, and maturin builds the extension module from there.

## Development Setup

Requirements:

- Rust toolchain
- Python 3.10+
- `uv`

Commands:

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
```

## Quality Checks

Run all Python checks from `python/`:

```bash
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## Quick Start

```python
import pandas as pd
from wbt import WeightBacktest

# pandas DataFrame input
df = pd.DataFrame(
    {
        "dt": ["2024-01-02 09:01:00", "2024-01-02 09:02:00"],
        "symbol": ["AAPL", "AAPL"],
        "weight": [0.5, 0.0],
        "price": [185.0, 186.0],
    }
)
wb = WeightBacktest(df, digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts")
print(wb.stats)

# File path input (csv, parquet, feather)
wb = WeightBacktest("path/to/data.parquet", digits=2)

# Polars DataFrame input
import polars as pl
df_pl = pl.read_parquet("path/to/data.parquet")
wb = WeightBacktest(df_pl, digits=2)
```

## Input Format

The `data` parameter accepts `pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`, or a file path (`str` / `Path`).

| Column   | Type     | Description |
|----------|----------|-------------|
| `dt`     | datetime | Bar end time |
| `symbol` | str      | Instrument code |
| `weight` | float    | Position weight at bar end |
| `price`  | float    | Trade price |

## Architecture

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
