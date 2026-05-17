# wbt Python 包

wbt Rust 回测引擎的 Python 接口项目。

[English](README.md)

## 开发目标

这个子项目的定位是：让研究侧可以用熟悉的 Python 数据结构快速接入回测，同时把核心计算留在 Rust 里保证性能。

主要目标：

1. 输入数据结构尽量兼容常见研究栈。
2. 输出结果直接可用于分析和绘图。
3. 所有 stats 类输出保持统一字段体系。
4. 提供开箱即用的可视化函数。

## 目录结构

当前目录是独立 Python 子工程：

```text
python/
|-- pyproject.toml
|-- README.md
|-- scripts/
|-- tests/
`-- wbt/
```

Rust crate 在上一层 ../Cargo.toml，通过 maturin 构建扩展模块。

## 安装与本地环境

要求：

- Rust 工具链
- Python 3.10+
- uv

安装：

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
```

## 快速开始

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
    weight_type="ts",   # "ts" 时序策略 / "cs" 截面策略
    yearly_days=252,
)

print("综合:", wb.stats)
print("多头:", wb.long_stats)
print("空头:", wb.short_stats)

print(wb.daily_return.head())
print(wb.dailys.head())
print(wb.pairs.head())

print(wb.segment_stats("2024-01-01", "2024-12-31", kind="多空"))
print(wb.long_alpha_stats)
```

## 输入规范

data 参数支持：

- pandas.DataFrame
- polars.DataFrame
- polars.LazyFrame
- str 或 Path 文件路径

路径输入支持格式：

- csv
- parquet
- feather
- arrow

必需字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| dt | datetime-like | K 线结束时间 |
| symbol | str | 标的代码 |
| weight | float | 目标持仓权重 |
| price | float | 收益计算价格 |

说明：

- 输入数据不允许空值。
- weight 会按 digits 进行小数截断后再参与回测。

## 核心 API

顶层导入（都可以从 `import wbt` 直接访问）：

```python
from wbt import (
    # 回测引擎
    WeightBacktest, backtest,
    # 绩效指标（Rust 实现）
    daily_performance,
    top_drawdowns,
    rolling_daily_performance,
    cal_yearly_days,
    # 策略工具（纯 Python）
    weights_simple_ensemble,
    cal_trade_price,
    log_strategy_info,
    # 报告生成
    generate_backtest_report,
    # 测试数据
    mock_symbol_kline, mock_weights,
)
```

主要对象与函数：

- `WeightBacktest(...)`: 回测主入口。
- `backtest(...)`: 便捷封装，返回 `WeightBacktest`。
- `daily_performance(returns, yearly_days=252)`: 基于日收益数组的完整绩效指标。
- `top_drawdowns(returns, top=10)`: Top-N 回撤窗口。
- `rolling_daily_performance(df, ret_col, window=252, min_periods=100, yearly_days=None)`: 滚动窗口日度绩效。
- `cal_yearly_days(dts)`: 根据日期序列自动推断年度交易日数。
- `weights_simple_ensemble(df, weight_cols, method="mean", only_long=False, **kwargs)`: 多策略权重集成（`mean` / `vote` / `sum_clip`）。
- `cal_trade_price(df, digits=None, windows=(5, 10, 15, 20, 30, 60))`: 按品种生成 TWAP / VWAP 与下根 K 线交易价表。
- `log_strategy_info(strategy, df)`: 用 loguru 打印每个品种的权重摘要。
- `generate_backtest_report(wb, output_path)`: 输出自包含的 HTML 报告。
- `mock_symbol_kline(...)` / `mock_weights(...)`: 快速实验用的模拟数据生成器。

常用 `WeightBacktest` 属性与方法：

- `stats`, `long_stats`, `short_stats`
- `daily_return`, `long_daily_return`, `short_daily_return`
- `dailys`, `pairs`
- `alpha`, `alpha_stats`, `bench_stats`
- `segment_stats(sdt, edt, kind)`
- `long_alpha_stats`
- `get_symbol_daily(symbol)`, `get_symbol_pairs(symbol)`

### 日志说明

`cal_yearly_days` 和 `rolling_daily_performance` 在 Rust 端通过 `log` crate 发出 warning（例如跨度不足时的 252 兜底）。包加载时已经初始化 `pyo3-log`，warning 会自动出现在 Python 标准 `logging` 中。如果使用 loguru，配置一次 [`InterceptHandler`](https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging) 即可路由到 loguru sink。

## 可视化函数

提供两套绘图接口：

### `wbt.plotting` — 单图模块

```python
from wbt.plotting import (
    plot_backtest_overview,
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_long_short_comparison,
    plot_monthly_heatmap,
    plot_pairs_analysis,
    plot_symbol_returns,
)
```

### `wbt.report` — 报告级组合图

```python
from wbt.report import (
    HtmlReportBuilder,
    LongShortComparisonChart,
    generate_backtest_report,
    get_performance_metrics_cards,
    plot_backtest_stats,
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_distribution,
    plot_drawdown_analysis,
    plot_long_short_comparison,
    plot_monthly_heatmap,
)
```

示例：

```python
fig1 = plot_cumulative_returns(wb.daily_return)
fig2 = plot_drawdown(wb.daily_return)
fig3 = plot_pairs_analysis(wb.pairs)

# 三图组合的绩效概览
fig4 = plot_backtest_stats(wb.daily_return, ret_col="total")

# 可选：导出 html 字符串
html = plot_backtest_overview(wb.daily_return, to_html=True)

# 完整 HTML 报告文件
generate_backtest_report(wb, "report.html")
```

## 质量检查

在 python/ 目录执行：

```bash
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## 架构总览

```text
repo-root/
|-- Cargo.toml
|-- src/
|   |-- lib.rs                       # PyO3 绑定（pyfunction、_wbt pymodule）
|   `-- core/
|       |-- cal_yearly_days.rs       # cal_yearly_days 的 Rust 核心
|       |-- daily_performance.rs
|       |-- rolling_daily_performance.rs
|       |-- top_drawdowns.rs
|       `-- ...                      # 回测引擎内部模块
`-- python/
    `-- wbt/
        |-- __init__.py              # 顶层导出
        |-- _df_convert.py           # pandas <-> Arrow IPC 工具
        |-- _wbt.pyi                 # Rust 扩展类型 stub
        |-- backtest.py              # WeightBacktest 类
        |-- mock.py                  # mock_symbol_kline / mock_weights
        |-- top_drawdowns.py         # _wbt.top_drawdowns 的适配层
        |-- utils/                   # 适配层 + 纯 Python 工具
        |   |-- __init__.py
        |   |-- cal_yearly_days.py
        |   |-- rolling_daily_performance.py
        |   |-- weights_simple_ensemble.py
        |   |-- cal_trade_price.py
        |   `-- log_strategy_info.py
        |-- plotting/                # 单图（plotly）
        |   |-- __init__.py
        |   |-- _common.py
        |   |-- returns.py
        |   |-- risk.py
        |   |-- trades.py
        |   `-- overview.py
        `-- report/                  # HTML 报告 + 组合图
            |-- __init__.py
            |-- _generator.py
            |-- _plot_backtest.py
            `-- html_builder.py
```

## License

[MIT](../LICENSE)
