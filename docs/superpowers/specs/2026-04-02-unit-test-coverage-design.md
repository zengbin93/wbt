# Unit Test Coverage Design

## Goal

为 Rust 和 Python 分别添加单元测试，覆盖所有公共接口和关键内部函数，目标覆盖率 90%+。

## Strategy

**方案 B：按模块逐步补齐**，按依赖顺序从底层到上层逐模块添加测试。

- 测试数据：全部内联构造小型数据，不依赖外部文件
- Rust 测试位置：内部函数用 `#[cfg(test)] mod tests`，pub API 用 `tests/` 集成测试
- Python 测试位置：`python/tests/` 下 pytest 文件

## Rust Inline Tests (`#[cfg(test)]`)

### 1. `src/core/utils.rs`

| 函数/trait | 测试点 |
|-----------|--------|
| `WeightType` | FromStr "ts"→TS, "cs"→CS, 无效字符串报错 |
| `RoundToNthDigit` for f64 | `round_to_nth_digit(0..6)`, `round_to_2/3/4_digit`, 负数, 0.0, 大数 |
| `Quantile` for Vec<f64> | 空 vec→None, 单元素, q=0/0.5/1, 已排序/未排序 |
| `date_key_to_naive_date` | 已知日期 key→NaiveDate 对照, 边界日期 (epoch, 远未来) |
| `pearson_corr_inline` | 完全正相关 [1,2,3]&[2,4,6]→1.0, 负相关→-1.0, 无相关, 常数序列 |
| `std_inline` | 全相同→0, 已知序列对照, 单元素 |
| `min_max` | 值在范围内/低于 min/高于 max |

### 2. `src/core/trade_dir.rs`

| 函数 | 测试点 |
|-----|--------|
| `TradeDir` Display/FromStr | "多头"↔Long, "空头"↔Short, "多空"↔LongShort |
| `TradeAction` Display/FromStr | 四个方向的序列化/反序列化 |
| `TradeAction::first_create` | 已有 3 个测试，补充极大/极小 vol |
| `TradeAction::get_event_seq` | 所有 4×4 组合中的有效路径 |

### 3. `src/core/errors.rs`

| 变体 | 测试点 |
|-----|--------|
| `NoneValue` | Display 包含字段名 |
| `ReturnsEmpty` | Display 固定文本 |
| `Polars` / `Unexpected` | From 转换正确 |

### 4. `src/core/daily_performance.rs`

| 函数 | 测试点 |
|-----|--------|
| `daily_performance()` | 空数组→默认值, 全零→合理默认值, 全正/全负/混合小数据集→手算对照 |
| `daily_performance_drawdown` | 已知回撤序列→手算 top-N 回撤均值和最长回撤 |
| `calc_underwater` | 累积收益→水下曲线对照 |
| `calc_underwater_valley/peak/recovery` | 构造有明确谷底/峰值/恢复点的序列 |

### 5. `src/core/evaluate_pairs.rs`

| 函数 | 测试点 |
|-----|--------|
| `evaluate_pairs_soa()` | 构造 PairsSoA (2-3 笔交易)→验证胜率/盈亏比/持仓天数等 |
| `evaluate_pairs_soa()` | 空 pairs→合理处理, 全盈/全亏 |
| `compute_break_even_point` | 已知盈亏对序列→手算对照 |

### 6. `src/core/native_engine.rs`

| 函数/结构 | 测试点 |
|----------|--------|
| `dt_to_date_key_fast` | 已知 datetime i64→date_key 对照 (微秒/毫秒/纳秒) |
| `dt_to_days_since_epoch` | 同上 |
| `DailysSoA::to_dataframe` | 构造小型 SoA→验证 DataFrame 列名、行数、类型 |
| `PairsSoA::to_dataframe` | 同上 |
| `LotsSoA` | push/pop/peek/is_empty, 容量边界 |

### 7. `src/core/mod.rs`

已有 `test_round_weight`, `test_convert_datetime`, `test_unique_symbols`。补充：

| 函数 | 测试点 |
|-----|--------|
| `WeightBacktest::new` | 正常创建, 缺失列报错, fee_rate=None 使用默认值 |
| `convert_datetime` | 已有测试，补充 Date 类型列（非 Datetime）的处理 |
| `round_weight` | 补充 digits=0, digits=4 边界 |

## Rust Integration Tests (`tests/`)

### `tests/test_backtest.rs`

构造 3 symbol × 20 天的小型 DataFrame，测试完整流程：
- `WeightBacktest::new()` → `backtest()` → report 非空
- `dailys_df()` 返回正确列和行数
- `pairs_df()` 返回 Some/None
- `alpha_df()` 返回正确列
- TS vs CS 模式产生不同结果

## Python Tests (`python/tests/`)

### `test_df_convert.py`

| 函数 | 测试点 |
|-----|--------|
| `pandas_to_arrow_bytes` + `arrow_bytes_to_pd_df` | 往返一致性（int/float/str/datetime 列）|
| 空 DataFrame | 往返不报错 |
| Series 输入 | `pandas_to_arrow_bytes` 接受 Series |

### `test_daily_performance.py`

| 函数 | 测试点 |
|-----|--------|
| `daily_performance()` | 正常 numpy array→返回 dict 含 17 个 key |
| | 全零数组→不报错 |
| | yearly_days 参数影响年化计算 |
| | 空数组→返回默认值 |

### `test_backtest.py`

构造 2 symbol × 15 天内联 DataFrame，测试 `WeightBacktest` 所有 property 和 method：

| 属性/方法 | 测试点 |
|----------|--------|
| `__init__` | 正常构造, 参数传递正确 |
| `stats` | 返回 dict, 含 29 个 key, 值为 float/str |
| `symbol_dict` | 返回 list[str], 长度=symbol 数 |
| `daily_return` | DataFrame, 列含 date + symbols + total |
| `dailys` | DataFrame, 15 列 |
| `alpha` | DataFrame, 列 = [date, 超额, 策略, 基准] |
| `pairs` | DataFrame, 11 列 |
| `alpha_stats` / `bench_stats` | dict, 17 key |
| `long_daily_return` / `short_daily_return` | DataFrame 结构正确 |
| `long_stats` / `short_stats` | dict, 17 key |
| `get_top_symbols(n, kind)` | 返回 list, 长度≤n, kind=profit/loss |
| `get_symbol_daily(symbol)` | 返回该 symbol 的 DataFrame |
| `get_symbol_pairs(symbol)` | 返回该 symbol 的 pairs |

### `test_backtest_edge.py`

| 场景 | 测试点 |
|-----|--------|
| 缺失列 | 报错（缺 weight/price/dt/symbol）|
| 空 DataFrame | 合理报错 |
| 单 symbol | 正常运行 |
| weight 全零 | 正常运行, pairs 为空 |
| `weight_type="cs"` vs `"ts"` | 两者结果不同 |

## Execution Order

1. utils.rs → 2. trade_dir.rs → 3. errors.rs → 4. daily_performance.rs → 5. evaluate_pairs.rs → 6. native_engine.rs → 7. mod.rs → 8. tests/test_backtest.rs → 9. Python test_df_convert.py → 10. Python test_daily_performance.py → 11. Python test_backtest.py → 12. Python test_backtest_edge.py

每步完成后运行 `cargo test` / `pytest` 验证通过，最终用 `cargo tarpaulin` 和 `pytest --cov` 确认覆盖率 ≥ 90%。
