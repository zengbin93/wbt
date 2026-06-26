# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

wbt 是面向量化策略的**持仓权重回测引擎**：Rust 实现核心计算（crate 在仓库根目录），通过 PyO3 + maturin 暴露成 Python 扩展模块 `wbt._wbt`，Python 子包在 `python/`。两边共用同一份 `Cargo.toml`（Python 端 `[tool.maturin] manifest-path = "../Cargo.toml"`）。

- Rust crate：`wbt`（cdylib + lib），版本号在 `Cargo.toml`
- Python 包：`wbt`，`pyproject.toml` 使用 `dynamic = ["version"]` 由 maturin 从 Cargo.toml 同步
- Python 端最低 3.10；Rust pyo3 走 `abi3-py310`，一份 wheel 覆盖 3.10+
- 关键依赖版本绑定：`pyo3 0.28` 与 `numpy 0.28` 必须配套；polars `0.53`

## 常用命令

### Rust 侧（仓库根目录执行）

```bash
cargo test --lib                                            # 跑全部 Rust 单测（CI 等价）
cargo test --lib core::backtest                             # 跑某个模块下所有测试
cargo test --lib new_valid_dataframe                        # 跑单个测试
cargo fmt --all -- --check                                  # 格式检查（与 pre-commit、CI 等价）
cargo clippy --all -- -D warnings -A non_snake_case         # lint（CI 等价；允许非 snake_case 用于中文字段映射）
cargo build --release
cargo publish -p wbt --dry-run                              # 发版前清单/license/依赖验证
```

`.pre-commit-config.yaml` 镜像了 CI 的 Rust Lint job（fmt + clippy）。本地启用一次：`pre-commit install`。

### Python 侧（在 `python/` 目录执行）

```bash
uv sync --extra dev                                                   # 同步依赖（只在需要更新包时跑）
uv run --no-sync maturin develop --release                            # 把 Rust 扩展编译进当前 venv（改动 Rust 后必须重跑）
uv run --no-sync pytest -v                                            # 跑测试
uv run --no-sync pytest -v tests/test_backtest.py                     # 单文件
uv run --no-sync pytest -v tests/test_backtest.py::test_xxx -s        # 单用例
uv run --no-sync ruff format --check .                                # 格式（CI 等价）
uv run --no-sync ruff check . --no-fix                                # lint（CI 等价）
uv run --no-sync basedpyright                                         # 类型检查（CI 等价）
```

- **所有 `uv run` 默认带 `--no-sync`**，跳过每次自动同步，提速；只有真要更新包时才跑 `uv sync --extra dev`。
- 改了 Rust 代码后，**必须重跑 `uv run --no-sync maturin develop --release`** Python 才能看到变化。
- `tests/manual/` 被 pytest、ruff、basedpyright 一致排除（见 `pyproject.toml` 中 `norecursedirs` / `exclude`）；放性能/对比脚本，不在 CI 跑。
- 文件名后缀 `_script.py` 的 test 文件是与外部库（czsc 等）对比的基准脚本，按需手跑。

## 架构要点

### 边界约定：所有 DataFrame 跨 Rust↔Python 走 Arrow IPC

- Python → Rust：`wbt._df_convert.pandas_to_arrow_bytes` / `polars_to_arrow_bytes` 编码为 IPC 字节，传给 `PyWeightBacktest.from_arrow` 等接口。
- Rust → Python：`src/lib.rs::df_to_pyarrow` 把 polars DataFrame 写成 IPC 字节，Python 端再 `arrow_bytes_to_pd_df` 转回 pandas。
- **新增任何 Rust 函数若返回表格数据，应延续这套模式**：参数收 `PyBytes`，返回 `PyBytes`。直接走 numpy 数组只用于纯数组（如 `daily_performance`）。
- pandas→Arrow 时 `datetime64[*]` 列会被强转为 `datetime64[ms]`，因为 Rust polars 端不支持微秒精度。修改这块要同步看 `convert_datetime`（`src/core/mod.rs`）的接收类型分支。

### Rust 核心结构（`src/core/`）

- `mod.rs`：`WeightBacktest` 主体——`new` 做 dt 类型归一化、weight 4 位四舍五入、按 symbol 的 **O(N) counting sort**（替代 polars 通用排序）；之后 `backtest()` 在固定线程池里跑。
- `backtest.rs` / `native_engine.rs`：核心计算，结果以 SoA（Struct-of-Arrays）形式存放（`DailysSoA` / `PairsSoA`），DataFrame 是**按需物化 + 缓存**（`daily_return_cache` / `dailys_cache` / `pairs_cache`）。
- `daily_performance.rs` / `top_drawdowns.rs` / `rolling_daily_performance.rs` / `cal_yearly_days.rs`：可单独通过 `wbt.xxx` 调用的独立指标函数。
- `report.rs` / `evaluate_pairs.rs` / `period_win_rates.rs` / `yearly_return.rs`：报告与拆分指标。
- 并发：`rayon` 线程池在 `WeightBacktest::backtest` 里显式构建（64MB stack、可指定 `n_jobs`），不要直接用全局 `rayon` 池。
- 错误：`errors::WbtError`，通过 `anyhow::Context` 累积上下文，FFI 边界统一转成 `PyException`。

### Python 适配层（`python/wbt/`）

- `__init__.py` 收口公共 API；改动公共面要同步 `_wbt.pyi`、根目录 `README.md`（中文，GitHub 主页默认）/ `README_EN.md`（英文镜像）、`python/scripts/quick_start.ipynb`（发版清单第 5 节明确要求四方一致）。
- `backtest.py::WeightBacktest`：调度 pandas / polars / 文件路径三条输入路径；维护 `STATS_FIELD_ORDER` 强制按发版约定的中文字段顺序输出（**修改 stats 字段时必须同步这个列表**）。
- `utils/`：每个迁移自 czsc 的函数独占一个文件；Rust 核心 + Python adapter 的拆分（`cal_yearly_days`、`rolling_daily_performance`）vs. 纯 Python（`weights_simple_ensemble`、`cal_trade_price`、`log_strategy_info`）。
- `plotting/` 是单图（plotly），`report/` 是组合图 + `HtmlReportBuilder` + `generate_backtest_report`。

### 日志桥接

Rust 端用 `log::warn!`（例如 `cal_yearly_days` 跨度不足时回退 252 的 warning）；`_wbt` 模块初始化时调用 `pyo3_log::try_init()`，warning 自动进入 Python `logging`。loguru 用户配一次 `InterceptHandler` 即可统一接管，不要在 Rust 端 println。

## 字段顺序与中文命名（**硬约束**）

所有 `stats` 类输出（`stats`、`long_stats`、`short_stats`、`segment_stats`、`long_alpha_stats`）的**字段名是中文、顺序固定**（见 `docs/desgin.md` 20260403 节与 `backtest.py::STATS_FIELD_ORDER`）。`src/lib.rs::PyWeightBacktest::stats` 里逐项 `set_item` 的顺序也要与之一致。增删字段：

1. 改 Rust 端 `src/lib.rs` 与对应 `report.rs` / 指标实现；
2. 改 Python 端 `STATS_FIELD_ORDER` 与 `_wbt.pyi`；
3. 更新 README 与 docstring 示例；
4. 在 0.x 阶段属于 BREAKING，需 MINOR 升版并在 release notes 显式标注。

## 测试与质量门

CI（`.github/workflows/ci.yml`）跑 5 个 job 必须全绿：

- `Rust Tests`：ubuntu / macos / windows × stable
- `Rust Lint`：fmt + clippy
- `Python Lint`：ruff format / ruff check
- `Python Type Check`：先 `maturin develop --release` 再 basedpyright（**类型检查依赖编译出的 `_wbt.abi3.so`**）
- `Python Test`：py3.10–3.13 × ubuntu，3.13 × macos / windows

本地复现失败时按上面"常用命令"逐条跑。

## 发版

详尽清单见 `docs/release_check.md`，硬性环节：

- §1 版本号严格 SemVer，`Cargo.toml` 与 Python 端必须完全一致；同版本号不可重发；
- §4 发版前必须做一次大模型全仓 review；
- §5 文档与代码不一致**一律改文档**（除非代码是 bug）。

发布顺序固定：**先 crate 后 Python**。

- `crate-vX.Y.Z` tag → `.github/workflows/release-crate.yml` → crates.io
- `vX.Y.Z` tag → `.github/workflows/release.yml` → PyPI（wheel 矩阵 + sdist）

## 关联生态

- [czsc](https://github.com/waditu/czsc)：5 个工具函数（`cal_yearly_days` / `rolling_daily_performance` / `weights_simple_ensemble` / `cal_trade_price` / `log_strategy_info`）从 czsc 迁移过来，数值口径与 czsc 对齐（见 `python/tests/test_compare_with_czsc_script.py`，手跑）。
- [wmr](https://github.com/zengbin93/wmr)：权重数据持久层；wbt 是其下游的回测计算层。
