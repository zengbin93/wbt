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

导入：

```python
from wbt import WeightBacktest, backtest, daily_performance
```

主要对象与函数：

- WeightBacktest(...): 主入口。
- backtest(...): 便捷封装，返回 WeightBacktest。
- daily_performance(...): 独立绩效指标函数。

常用属性与方法：

- stats, long_stats, short_stats
- daily_return, long_daily_return, short_daily_return
- dailys, pairs
- alpha, alpha_stats, bench_stats
- segment_stats(sdt, edt, kind)
- long_alpha_stats
- get_symbol_daily(symbol), get_symbol_pairs(symbol)

## 可视化函数

绘图函数位于 wbt.plotting：

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

示例：

```python
fig1 = plot_cumulative_returns(wb.daily_return)
fig2 = plot_drawdown(wb.daily_return)
fig3 = plot_pairs_analysis(wb.pairs)

# 可选：导出 html 字符串
html = plot_backtest_overview(wb.daily_return, to_html=True)
```

## 质量检查

在 python/ 目录执行：

```bash
uv run pytest -v
uv run ruff format --check .
uv run ruff check . --no-fix
uv run basedpyright
```

## License

[MIT](../LICENSE)
