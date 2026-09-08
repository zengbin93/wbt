# API 表面积与数据转换成本（SKZ-726）

## 接口边界

审计基线为 `ee8d2832479c06a1ce12cede40be6a138c754e5d`，检查日期为 2026-09-08。

| 项目 | 可见性与调用证据 | 本次处理 |
| --- | --- | --- |
| `pearson_corr_inline` | `pub(crate)`；全仓搜索只有定义及自身 4 个单测；无生产调用 | 删除实现、`allow(dead_code)` 和这 4 个单测 |
| `Quantile` | `wbt::core::utils::Quantile` 可从 crate 外访问；仓内仅自身测试 | 保留签名、实现和测试，不添加弃用警告 |
| `TradeAction::first_create` | `wbt::core::trade_dir::TradeAction::first_create` 可从 crate 外访问；仓内仅自身测试 | 保留签名、实现和测试，不添加弃用警告 |

仓内证据可用 `rg -n 'pearson_corr_inline|Quantile|first_create' src tests python` 在基线复核。
`src/lib.rs` 的 `pub mod core`、`src/core/mod.rs` 的 `pub mod utils` / `pub mod trade_dir`
使后两项成为公开 Rust API；没有 Python 导出不能推导为私有接口。

下游证据：公开仓库 axile 的固定版本
[`bcb255261a5e784f5e31749037c292d94f119efe`](https://github.com/zengbin93/axile/tree/bcb255261a5e784f5e31749037c292d94f119efe)
在 [`pyproject.toml`](https://github.com/zengbin93/axile/blob/bcb255261a5e784f5e31749037c292d94f119efe/pyproject.toml#L27)
固定 `wbt==0.8.2`，在
[`performance.py`](https://github.com/zengbin93/axile/blob/bcb255261a5e784f5e31749037c292d94f119efe/axile/server/performance.py#L177)
实际构建 `WeightBacktest` 并读取 `daily_return`。
这证明 Python 回测与结果接口有生产下游，**不能**证明两个 Rust 工具接口无人使用。

以下 GitHub Code Search 在当日、当前可见范围内各返回 0 项（每项 `--limit 100`）：

```bash
gh search code '"Quantile" user:zengbin93 language:Rust' --limit 100 --json repository,path
gh search code '"first_create" user:zengbin93 language:Rust' --limit 100 --json repository,path
gh search code '"Quantile" org:sheng-ke-zhi language:Rust' --limit 100 --json repository,path
gh search code '"first_create" org:sheng-ke-zhi language:Rust' --limit 100 --json repository,path
gh search code '"wbt" filename:Cargo.toml user:zengbin93' --limit 100 --json repository,path
gh search code '"wbt" filename:Cargo.toml org:sheng-ke-zhi' --limit 100 --json repository,path
```

搜索索引甚至未返回本仓已知定义，存在索引缺失/延迟；上述阴性结果不构成删除依据，
也不覆盖离线、未授权仓库及其他组织。当前无足够证据认定公开项可删除。
将来若收缩公开面，应先确认 Rust 下游及替代路径，再单独提出弃用版本、迁移说明和删除版本；
遵守本仓 0.x 破坏性变更需 MINOR 升版的约定，不把“仓内无调用”作为删除批准。

## 成本归因与测量方法

本次只清理私有死代码并建立诊断基线，不改变以下运行路径，也不宣称提速：

- pandas 输入：`backtest.py` 选列并 `.copy()`，`_df_convert.py` 遇日期列再次复制和转毫秒，
  最后选列、`.astype()` 后保留 `dfw`。实际拷贝量受 pandas 版本、dtype 和 Copy-on-Write 影响。
- 结果读取：Rust 的 `daily_return_cache` 缓存 DataFrame；`src/python.rs` 每次仍写 IPC，
  Python `daily_return` 每次经 `arrow_bytes_to_pd_df` 解码并分配新的 pandas 表。
  直接缓存可变 pandas 对象会改变“修改一次读取不污染下次读取”的行为，需独立设计和回归。
- 宽表：`build_daily_return_df` 先创建 `S×D` 个 `Option<f64>` 槽位，再生成 Polars 列，
  即使长表稀疏也按完整日期轴分配。当前目标平台 `Option<f64>` 为 16 字节，临时槽位预算为
  `16×S×D`，另有输出列、有效位图、IPC、pandas 和引擎持有数据；这个预算不是峰值 RSS。
  `stats` 直接使用已算报告，是无需宽表的对照路径。

`python/scripts/benchmark_api_costs.py` 使用固定公式生成数据：每个品种每隔 stride 个日历日
参与一天，品种错开参与日，每天 09:30/15:00 两根 bar；权重每 20 天在 ±0.5 切换，
价格为正值的线性趋势加正弦扰动。数据无随机性、无空值，未模拟真实停牌、交易日历或持仓分布。
`digits=2, fee_rate=0.0002, n_jobs=1, weight_type=ts, yearly_days=252`。

每个 case 的每个样本都在新子进程执行一次；导入、造数和必要的回测/缓存准备不计入耗时。
首次读取测首次物化；重复读取仅在同一对象预读一次后计时；IPC 编码 case 预热 Rust 缓存；
IPC 编码还包含 FFI 返回 bytes 的复制；IPC 解码 case 使用提前生成的 bytes。`pandas_ipc` 包含日期转换、Arrow 表构建和编码。
`arrow_init` 使用提前编码的数据，包含 Rust 解码、校验、回测；与 `pandas_init` 的差值
还包含 Python 保留 `dfw` 等工作，不能直接等同于 `.copy()` 的成本。

JSON 保存每次耗时、测量前后 `ru_maxrss`、输出 shape/bytes 和中位数。
首次/重复读取的差异还包含编解码器初始化，不能当作 Rust 宽表构建的纯耗时；各 case 中位数不能简单相减归因。
RSS 为**整个子进程生命周期的峰值**，包含导入、造数、输入、预热等，不能当作单步骤分配量；
测量前后高水位差也不是完整分配量。各 case 不共用进程，避免前一个 case 的峰值污染后一个。
耗时仅包围调用，结果保持存活到内存采样之后。当前仅支持 Linux/macOS。

## 复现

在待测版本的 `python/` 目录中运行；旧版本可复制同一脚本后执行。必须重新构建扩展，
`--label` 只是记录版本，不会检查或切换版本。相同机器、线程配置和依赖下比较原始 JSON。

```bash
uv venv --python 3.13
uv pip install numpy==2.5.3 pandas==3.0.5 pyarrow==25.0.1 polars==1.44.1 plotly==7.0.0 loguru==0.7.3 maturin==1.15.0
uv run --no-sync maturin develop --release --skip-install
PYTHONPATH=. POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 uv run --no-sync python scripts/benchmark_api_costs.py --symbols 100 --days 252 --repeat 3 --label ee8d283 > small.json
PYTHONPATH=. POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 uv run --no-sync python scripts/benchmark_api_costs.py --symbols 1000 --days 1000 --repeat 3 --label ee8d283 > dense.json
PYTHONPATH=. POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 uv run --no-sync python scripts/benchmark_api_costs.py --symbols 1000 --days 1000 --stride 10 --repeat 3 --label ee8d283 > sparse.json
```

## 测量结果

硬件：Apple M4 / 16 GiB；macOS 26.6.2 arm64；Python 3.13.12。

测的是清理前 `ee8d283` 构建的 release 扩展。每组 3 次，表内为
**耗时中位数 ms / 整进程峰值 RSS 中位数 MiB**。

| 操作 | 100×252，50,400 行 | 1000×1000，2,000,000 行 | 1000×1000，stride=10，200,000 行 |
| --- | ---: | ---: | ---: |
| `pandas_ipc` | 6.798 / 149.8 | 59.944 / 648.9 | 5.176 / 186.9 |
| `pandas_init` | 15.828 / 160.1 | 241.568 / 912.7 | 25.837 / 227.6 |
| `arrow_init` | 5.423 / 159.0 | 182.457 / 874.9 | 37.658 / 228.6 |
| `stats` | 0.016 / 161.0 | 0.016 / 914.3 | 0.011 / 227.6 |
| `daily_return_first` | 7.286 / 163.6 | 22.227 / 948.2 | 12.758 / 269.8 |
| `daily_return_repeat` | 3.341 / 164.7 | 5.658 / 954.9 | 6.103 / 278.0 |
| `ipc_encode_cached` | 0.133 / 163.0 | 2.091 / 953.2 | 2.313 / 274.5 |
| `ipc_decode` | 2.919 / 163.5 | 9.266 / 949.0 | 4.074 / 269.7 |

三组宽表 shape 分别为 `(252, 102)`、`(1000, 1002)`、`(1000, 1002)`，脚本逐次校验。

可支持的结论：

- 密集组 pandas 初始化 241.568 ms、预编码 Arrow 初始化 182.457 ms；但稀疏组后者反而更慢
  （37.658 vs 25.837 ms）。不能据此推算删除复制的收益。小组 `pandas_ipc` 原始样本为
  93.63 / 4.28 / 6.80 ms，进一步说明冷启动与系统负载的影响。
- 密集组重复读取仍需 5.658 ms，稀疏组为 6.103 ms；两组缓存后的 IPC 编码也仍需约 2 ms。
  Rust 缓存没有消除每次跨语言结果转换的成本。
- 行数降为 1/10 后，宽表仍有一百万个品种日期槽位，临时 `Option<f64>` 槽位预算同为
  15.26 MiB。稀疏组首次宽表读取的整进程峰值为 269.8 MiB，`stats` 对照为 227.6 MiB；
  此差值包含多种分配，不能全部算给临时矩阵。
- 优先在真实调用频次与数据上复测结果重复读取，再决定是否引入保持返回对象隔离语义的缓存
  或更窄的查询接口。本 PR 不改变公开 API、结果缓存或数据转换行为。

本次锁文件依赖在线下载停滞，使用本机缓存安装上述精确版本，并非锁文件环境。
Rust 为 `rustc 1.98.1`，Polars crate 为 `0.53.0`，release 使用仓库默认 fat LTO、
`codegen-units=1, opt-level=3`；原始 JSON 与本次 Cargo.lock 随任务附件交付。

共享机器未隔离其他系统负载，3 次样本不足以判断微小差异的统计显著性；不设置 CI 性能阈值。

真实数据局限：仓库既有脚本指定的真实 Feather 文件在本机不存在，本次没有可用的真实权重数据。
这些数字只描述合成负载，不能外推真实策略总体耗时或宣布线上瓶颈。
