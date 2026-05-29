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
- wb.aggregated_pairs / wb.key_trades(top=3)：按 (品种, 开仓时间, 平仓时间) 聚合去重的开平记录，以及每年最赚/最亏各 N 笔关键交易（Rust 计算）。
- wb.to_result(target_vol=0.20) → BacktestResult：绘图与审核页面的标准输入数据对象（详见下文「可视化」）。
- wb.segment_stats(...)：任意时间区间统计。
- wb.long_alpha_stats：波动率调整后的多头超额指标。
- wb.is_good_strategy(mode="history" | "recent", ...)：客观判定一个策略能不能搞。返回 dict，含 `is_good`（bool）、`reason`、`alpha_degenerate`（bool）、年度明细（history 模式）或最近窗口指标（recent 模式）以及各条件通过标记。可调参数：`target_vol`、`max_dd_threshold`、`min_year_days`、`recent_days`、`min_history_days`。`recent` 模式下，历史最大回撤在**剔除 recent 窗口后**的样本上计算（带可配置的 `min_history_days` floor），与 recent 窗口在时间上完全错开。Alpha 退化（NaN/Inf 或 long/bench 零方差）通过 `alpha_degenerate=True` 报告，所有 alpha 派生字段为 `None`，`is_good=False`——不会假阳性"零回撤通过"。返回 dict 的 key 按字母序稳定排列；`history` 与 `recent` 模式返回**互斥**的 key 集合（按 `mode` dispatch）。

## 独立工具函数

除了 `WeightBacktest` 类，wbt 顶层还导出一组独立工具：

- `daily_performance(returns, yearly_days=252)`：基于日收益序列的完整绩效指标（Rust 核心）。
- `top_drawdowns(returns, top=10)`：Top-N 回撤窗口（Rust 核心）。
- `rolling_daily_performance(df, ret_col, window=252, min_periods=100, yearly_days=None)`：滚动窗口日度绩效（Rust 核心）。
- `cal_yearly_days(dts)`：根据日期序列自动推断年度交易日数（Rust 核心）。
- `weights_simple_ensemble(df, weight_cols, method="mean", only_long=False, **kwargs)`：多策略权重集成（`mean` / `vote` / `sum_clip`）。返回新 DataFrame（不修改入参 `df`）。`sum_clip` 模式可通过 kwargs 传 `clip_min=-1, clip_max=1`。
- `cal_trade_price(df, digits=None, **kwargs)`：按品种计算 TWAP / VWAP 与下根 K 线交易价表。kwargs 支持 `windows=(5, 10, 15, 20, 30, 60)` 与 `copy=True`。
- `log_strategy_info(strategy, df)`：用 loguru 打印每个品种的权重摘要。
- `mock_symbol_kline(...)` / `mock_weights(...)`：快速实验的模拟数据生成器。

Rust 端发出的 warning（如 `cal_yearly_days` 跨度不足时回退到 252）通过 `log` crate 触发，再由 `pyo3-log` 桥接到 Python 标准 `logging`，loguru 用户配置一次 `InterceptHandler` 即可接管。

## HTML 报告生成

`wbt.generate_backtest_report(df, output_path)` 输出一个自包含的 HTML 报告（回测概览、多空对比、关键交易等标签页）。内部仅做一次 `wb.to_result()` 预处理，再交由 `wbt.plotting` 绘图。

## 可视化

所有绘图函数以 **`BacktestResult`** 为标准输入——一次性算好绘图所需的全部数据，绘图函数零数据转换：

```python
result = wb.to_result()            # 标准输入数据对象
from wbt.plotting import plot_cumulative_returns, plot_key_trades
fig = plot_cumulative_returns(result, keys=["多空", "多头", "空头"])
plot_key_trades(result, to_html=True)
result.to_dict(full=True)          # JSON 安全，供审核页面走 HTTP
```

- `BacktestResult` 字段：`dates` / `year_starts` / `curves`（原始曲线，键 多空/多头/空头/基准/超额）/ `curves_voladj`（波动率归一，按需）/ `return_dist` / `monthly` / `symbol_returns` / `pairs_dist` / `stats` / `stats_by_side`，以及审核字段 `drawdowns` / `key_trades` / `verdict`（均为按需 `cached_property`）。
- `wbt.plotting`（均为单一职责单图，无组合图）：`plot_cumulative_returns`（`voladj=True` 为波动率归一）/ `plot_drawdown` / `plot_daily_return_dist` / `plot_monthly_heatmap` / `plot_symbol_returns` / `plot_pairs_pnl_dist` / `plot_pairs_hold_dist` / `plot_colored_table` / `plot_stats_comparison` / `plot_key_trades` / `plot_drawdowns_table` / `plot_verdict`。
- `wbt.report`：`generate_backtest_report` / `HtmlReportBuilder` / `get_performance_metrics_cards`。

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

## 相关开源项目

wbt 处于一组量化研究工具的生态中，几个最相关的项目如下：

- [**czsc**](https://github.com/waditu/czsc) — 基于缠论的综合性量化交易 Python 框架，覆盖信号、策略、Trader、EDA 与可视化；自 v1.0.x 起将核心算法用 Rust 实现并通过 PyO3 暴露（`czsc._native`）。**与 wbt 的关系：** wbt 从 czsc 迁移了 5 个评估 / 工具函数（`cal_yearly_days`、`rolling_daily_performance`、`weights_simple_ensemble`、`cal_trade_price`、`log_strategy_info`），并保持数值口径与 czsc 对齐（见 `python/tests/test_compare_with_czsc_script.py`）；czsc 端的策略天然产出可被 wbt 消费的权重表。

- [**wmr**](https://github.com/zengbin93/wmr) — 基于 ClickHouse 与 DuckDB 的策略持仓权重管理系统，专注于大规模权重数据的持久化、版本管理与查询。**与 wbt 的关系：** wmr 是权重表的数据层（存储 / 检索），wbt 是把权重表转化为回测指标、日序列与 HTML 报告的计算层。

- [**talib-rs**](https://github.com/0xcjun/talib-rs) — 纯 Rust 实现的技术分析库，定位为经典 C 版 TA-Lib 的 drop-in 替代（结果逐位对齐、SIMD 加速、无 C 依赖）。**与 wbt 的关系：** Rust 侧的同侪项目——wbt 聚焦权重回测与绩效指标，talib-rs 覆盖标准技术指标。当策略需要"指标计算 + 权重回测"同时在 Rust/Python 栈里完成时，两者可组合使用。

整体上，三者勾勒出一条典型的研究到评估管线：**czsc**（信号与策略）→ **wmr**（权重存储）→ **wbt**（回测与指标），**talib-rs** 则在沿线提供可复用的 Rust 原生指标计算能力。

## License

[MIT](LICENSE)
