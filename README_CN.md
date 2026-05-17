# wbt

一个面向量化策略的持仓权重回测引擎，使用 Rust 提供高性能核心，并通过 Python 绑定提供易用接口。

[English](README.md)

## 项目目标

很多策略研究流程都以持仓权重作为策略表达的核心接口，但常见回测工具要么偏向订单级撮合模拟，要么在大规模多品种数据上速度不足。

wbt 的开发目标是：

1. 用统一的数据契约承接权重策略输入。
2. 用 Rust 保证计算性能与结果稳定性。
3. 提供 Python 友好的研究接口，降低接入成本。
4. 内置可直接用于评估与可视化的数据输出。

## 适用场景

- 时序策略与截面策略的权重回测。
- 多品种日收益拆解与归因分析。
- 多空拆分、分段统计和交易对评估。
- pandas、polars、文件输入等多种数据通路。

## 非目标

- 逐笔撮合与盘口微观结构仿真。
- 交易所撮合机制级别的高频细节模拟。
- 券商特定执行细节建模。

如果你的策略天然可表示为“随时间变化的目标权重”，wbt 会更合适。

## 仓库结构

- Rust crate：仓库根目录
- Python 包：python/

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

## Python 快速开始

Python 子项目位于 python/，导入路径保持为 import wbt。

```bash
cd python
uv sync --extra dev
uv run maturin develop --release
uv run pytest -v
```

示例：

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

完整 Python 指南见 python/README_CN.md。

## Rust 快速开始

在仓库根目录执行：

```bash
cargo test
```

依赖方式：

```toml
[dependencies]
wbt = "0.1"
```

## 输入数据契约

wbt 的核心输入字段为：

- dt：K 线结束时间
- symbol：标的代码
- weight：该时点目标持仓权重
- price：成交或估值价格

Python 侧支持输入：

- pandas.DataFrame
- polars.DataFrame / polars.LazyFrame
- 文件路径（csv、parquet、feather、arrow）

## 关键输出能力

- wb.stats：多空综合绩效指标。
- wb.long_stats / wb.short_stats：多头与空头拆分指标。
- wb.daily_return / wb.dailys：日度收益明细序列。
- wb.alpha / wb.alpha_stats：相对基准超额分析。
- wb.pairs：交易对级别评估数据。
- wb.segment_stats(...)：任意时间区间统计。
- wb.long_alpha_stats：波动率调整后的多头超额指标。

## 独立工具函数

除了 `WeightBacktest` 类，wbt 顶层还导出一组独立工具：

- `daily_performance(returns, yearly_days=252)`：基于日收益序列的完整绩效指标（Rust 核心）。
- `top_drawdowns(returns, top=10)`：Top-N 回撤窗口（Rust 核心）。
- `rolling_daily_performance(df, ret_col, window=252, min_periods=100, yearly_days=None)`：滚动窗口日度绩效（Rust 核心）。
- `cal_yearly_days(dts)`：根据日期序列自动推断年度交易日数（Rust 核心）。
- `weights_simple_ensemble(df, weight_cols, method="mean", only_long=False)`：多策略权重集成（`mean` / `vote` / `sum_clip`）。
- `cal_trade_price(df, digits=None, windows=(5, 10, 15, 20, 30, 60))`：按品种计算 TWAP / VWAP 与下根 K 线交易价表。
- `log_strategy_info(strategy, df)`：用 loguru 打印每个品种的权重摘要。
- `mock_symbol_kline(...)` / `mock_weights(...)`：快速实验的模拟数据生成器。

Rust 端发出的 warning（如 `cal_yearly_days` 跨度不足时回退到 252）通过 `log` crate 触发，再由 `pyo3-log` 桥接到 Python 标准 `logging`，loguru 用户配置一次 `InterceptHandler` 即可接管。

## HTML 报告生成

`wbt.generate_backtest_report(wb, output_path)` 输出一个自包含的 HTML 报告，集成 `wbt.report._plot_backtest` 的图表（累计收益、回撤分析、日收益分布、月度热力图、绩效概览、彩色指标表、多空对比）。

## 可视化

Python 子项目提供两个绘图模块：

- `wbt.plotting`：单图模块 — `plot_cumulative_returns` / `plot_drawdown` / `plot_daily_return_dist` / `plot_monthly_heatmap` / `plot_symbol_returns` / `plot_pairs_analysis` / `plot_backtest_overview` / `plot_colored_table` / `plot_long_short_comparison`。
- `wbt.report`：报告级组合图（被 `generate_backtest_report` 调用）— `plot_backtest_stats` / `plot_drawdown_analysis` / `plot_daily_return_distribution`，以及可复用的 `HtmlReportBuilder` 和 `get_performance_metrics_cards`。

## 开发与质量检查

- Rust 相关命令在仓库根目录执行。
- Python 相关命令在 python/ 目录执行。
- CI 会同时校验这两部分。

常用命令：

```bash
# 仓库根目录
cargo test

# python 子项目
cd python
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## 相关文档

- Python 英文文档：python/README.md
- Python 中文文档：python/README_CN.md
- 设计记录：docs/desgin.md

## License

[MIT](LICENSE)
