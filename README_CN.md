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

## 可视化

Python 子项目提供 wbt.plotting 模块，内置：

- 累计收益曲线
- 月度热力图
- 回撤分析图
- 日收益分布图
- 交易对分析图
- 回测总览图

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
