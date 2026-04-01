# wbt

高性能持仓权重回测引擎，Rust 编写，支持 Python 调用。

[English](README.md)

## 概述

**wbt**（Weight Back Test）是一个独立的策略持仓权重回测库。基于持仓权重序列，计算每日盈亏归因、交易对撮合（FIFO）及全面的绩效统计指标。底层采用零拷贝、缓存友好的 Rust 引擎，配合 rayon 实现多品种并行计算。

- **Rust crate**（`wbt-core`）：纯 Rust 库，无 Python 依赖
- **Python 包**（`wbt`）：通过 PyO3 绑定 + Arrow IPC 提供 pandas 友好的 API

## 安装

### Python

```bash
# 前置条件：Rust 工具链、maturin、Python >= 3.9
git clone <repo-url> && cd wbt
uv venv .venv && source .venv/bin/activate
uv pip install pandas pyarrow numpy maturin
maturin develop --release
```

### Rust

```toml
# Cargo.toml
[dependencies]
wbt-core = { path = "path/to/wbt/crates/wbt-core" }
```

## 快速开始

### Python

```python
import pandas as pd
from wbt import WeightBacktest

# 准备输入数据：DataFrame 包含 [dt, symbol, weight, price] 四列
dfw = pd.DataFrame({
    "dt":     ["2024-01-02 09:01:00", "2024-01-02 09:02:00", "2024-01-03 09:01:00", "2024-01-03 09:02:00"],
    "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
    "weight": [0.5, 0.5, 0.0, 0.0],
    "price":  [185.0, 186.0, 187.0, 185.5],
})

wb = WeightBacktest(dfw, digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts")

wb.stats          # dict: 夏普、卡玛、最大回撤、胜率等
wb.daily_return   # DataFrame: 各品种每日收益率 + total 汇总列
wb.dailys         # DataFrame: 各品种每日明细（edge、return、cost、turnover 等）
wb.alpha          # DataFrame: 策略超额收益（相对基准）
wb.pairs          # DataFrame: 全部 FIFO 撮合交易对
```

### Rust

```rust
use wbt_core::{WeightBacktest, WeightType};
use polars::prelude::*;

let dfw: DataFrame = /* 构建包含 [dt, symbol, weight, price] 的 DataFrame */;
let mut wb = WeightBacktest::new(dfw, 2, Some(0.0002))?;
wb.backtest(Some(4), WeightType::TS, 252)?;

let report = wb.report.as_ref().unwrap();
println!("夏普比率: {}", report.stats.daily_performance.sharpe_ratio);
```

## 输入格式

| 列名     | 类型     | 说明                                                                         |
|----------|----------|-----------------------------------------------------------------------------|
| `dt`     | datetime | K线结束时间；必须是连续的交易时间序列，不允许有时间断层                             |
| `symbol` | str      | 合约代码                                                                     |
| `weight` | float    | 该时刻的持仓权重；品种之间独立；正值=多头，负值=空头，0=空仓                       |
| `price`  | float    | 交易价格（收盘价、下一根K线开盘价、TWAP/VWAP 等均可）                             |

## 参数说明

| 参数          | 默认值   | 说明                                                  |
|---------------|----------|-------------------------------------------------------|
| `digits`      | `2`      | 权重列保留小数位数                                      |
| `fee_rate`    | `0.0002` | 单边交易成本（手续费 + 冲击成本）                         |
| `n_jobs`      | `1`      | 并行线程数（rayon 线程池）                               |
| `weight_type` | `"ts"`   | `"ts"`（时序策略：品种等权平均）或 `"cs"`（截面策略：品种求和） |
| `yearly_days` | `252`    | 年化交易日数量                                          |

## 输出属性

| 属性                | 类型      | 说明                                                    |
|---------------------|-----------|--------------------------------------------------------|
| `stats`             | dict      | 完整绩效报告（夏普、卡玛、回撤、胜率、交易指标等）            |
| `daily_return`      | DataFrame | 各品种每日收益率（透视表）+ `total` 汇总列                  |
| `dailys`            | DataFrame | 各品种每日明细：`n1b`、`edge`、`return`、`cost`、`turnover`、多空拆分 |
| `alpha`             | DataFrame | 策略超额收益（策略 vs. 基准）                              |
| `alpha_stats`       | dict      | 超额收益序列的绩效统计                                    |
| `pairs`             | DataFrame | 全部 FIFO 撮合交易对（盈亏比例、持仓K线数、方向等）          |
| `long_daily_return` | DataFrame | 多头每日收益率                                           |
| `short_daily_return`| DataFrame | 空头每日收益率                                           |
| `long_stats`        | dict      | 多头绩效统计                                             |
| `short_stats`       | dict      | 空头绩效统计                                             |
| `bench_stats`       | dict      | 基准（品种等权 n1b）绩效统计                               |

## 架构

```
wbt/
├── crates/wbt-core/    # 纯 Rust 库
│   └── src/
│       ├── lib.rs                # WeightBacktest 结构体与公开 API
│       ├── native_engine.rs      # 零拷贝并行引擎（rayon）
│       ├── daily_performance.rs  # 夏普、卡玛、回撤等绩效指标
│       ├── evaluate_pairs.rs     # 交易对统计评估
│       ├── backtest.rs           # 回测编排与 Alpha 计算
│       ├── report.rs             # 报告结构体与 JSON 序列化
│       ├── trade_dir.rs          # 交易方向与动作枚举
│       ├── errors.rs             # 错误类型
│       └── utils.rs              # WeightType、四舍五入、分位数
├── src/lib.rs          # PyO3 绑定（Arrow IPC 输入输出）
└── python/wbt/         # Python API 封装层
    ├── backtest.py     # WeightBacktest 类（pandas 友好）
    └── _df_convert.py  # Arrow <-> pandas 转换
```

### 实测性能

测试环境：全A股日线数据（2017-01 至 2025-04），Apple M 系列芯片，8 线程，release + LTO 编译：

| 数据集 | 行数 | 品种数 | 耗时 | 吞吐量 |
|--------|------|--------|------|--------|
| 全A股日线 | 8,440,404 | 5,351 | **0.63s** | 1340 万行/秒 |

单次运行产出：

| 输出 | 规模 | 说明 |
|------|------|------|
| `dailys` | 8,435,053 行 × 15 列 | 各品种每日盈亏归因 |
| `pairs` | 942,679 行 × 11 列 | FIFO 撮合交易对 |
| `daily_return` | 2,023 行 | 品种等权组合每日收益 |
| `alpha` | 2,023 行 × 4 列 | 策略超额收益 |
| `stats` | 29 项指标 | 夏普、卡玛、回撤、胜率等 |

<details>
<summary><code>stats</code> 输出样例（点击展开）</summary>

```python
{
  "开始日期": "2016-12-27",       "结束日期": "2025-04-28",
  "绝对收益": 0.4431,             "年化": 0.0552,
  "夏普": 0.345,                  "最大回撤": 0.2234,
  "卡玛": 0.2471,                 "日胜率": 0.5136,
  "日盈亏比": 1.0219,             "日赢面": 0.0384,
  "年化波动率": 0.16,              "下行波动率": 0.1167,
  "非零覆盖": 0.9802,             "盈亏平衡点": 0.9975,
  "新高间隔": 542.0,              "新高占比": 0.0277,
  "回撤风险": 1.3963,             "回归年度回报率": 0.0395,
  "长度调整平均最大回撤": 0.9516,   "交易胜率": 0.2697,
  "单笔收益": 17.51,              "持仓K线数": 9.79,
  "多头占比": 0.4611,             "空头占比": 0.5269,
  "与基准相关性": -0.2835,         "与基准空头相关性": -0.374,
  "波动比": 0.6665,               "与基准波动相关性": 0.1974,
  "品种数量": 5351
}
```

</details>

### 性能设计

- **O(N) 计数排序**：品种分组，替代通用排序
- **Struct-of-Arrays (SoA)**：热循环中的缓存局部性优化
- **延迟 DataFrame 物化**：内部计算使用原始向量，仅在访问时构建 DataFrame
- **FIFO 撮合引擎**：基于栈式 SoA 的动态扩容配对
- **rayon 数据并行**：跨品种并行，可配置线程池
- **零拷贝切片**：从 Polars DataFrame 直接提取连续内存块
- **Arrow IPC**：Python <-> Rust 零序列化开销数据传输

## 许可证

[MIT](LICENSE)
