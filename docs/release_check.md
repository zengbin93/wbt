# 发布检查清单（Release Checklist）

> 目标：在发版前以「机械式」的步骤排查掉常见回归，把版本质量从「凭手感」收敛为「可控」。
> 适用范围：`wbt` Rust crate（`crate-v*` tag → crates.io）与 `wbt` Python 包（`v*` tag → PyPI），二者通常同步发布。

发版规则速查：

- `v0.x.y` tag → 触发 `.github/workflows/release.yml`，构建 wheel + sdist 并上传到 PyPI。
- `crate-v0.x.y` tag → 触发 `.github/workflows/release-crate.yml`，发布到 crates.io。
- 版本号源：`Cargo.toml` 的 `package.version`（Python 包通过 `dynamic = ["version"]` 由 maturin 同步读取）。
- 默认两者版本号保持一致，先 crate 后 Python。

> **本清单中有三个不可跳过的硬性环节，分别对应本次发版稳定性的核心保障**：
> 1. **§1 版本号必须严格遵循 SemVer（语义化版本）**；
> 2. **§4 大模型对全仓代码做一次快速 Review**，捕捉单元测试覆盖不到的逻辑陷阱与隐藏 bug；
> 3. **§5 文档与代码一致性校验**——发现不一致一律以代码为准、修改文档（除非代码本身是 Bug）。

---

## 0. 前置确认（开工前 60 秒）

- [ ] 当前在 `main` 分支，且 `git status` 干净（无未提交改动、无未追踪敏感文件）。
- [ ] 已 `git pull --rebase`，本地与 `origin/main` 一致。
- [ ] 最近一次 `main` 上的 CI 全绿（GitHub Actions「CI」工作流）。
- [ ] 当前没有未合并的、被认为应进入本版本的 PR。

## 1. 版本号必须语义化（SemVer，硬性门禁）

> **强制要求**：版本号必须严格遵循 [Semantic Versioning 2.0.0](https://semver.org/)。这是发版的硬性门禁，违反任意一条即拒绝发版。

- [ ] **版本号格式校验**：`MAJOR.MINOR.PATCH`，三段均为非负整数。
  - 不允许 `0.1`、`1.0.0.0` 等位数不符；
  - 不允许 `0.1.8-dev`、`0.1.8.post1`、`v1.0-beta` 等非标准后缀（如需预发布，使用 `0.2.0-rc.1` 格式并在 RFC 中评审）；
  - `Cargo.toml`、`python/pyproject.toml`（若有显式 version）、`__init__.py`（若暴露 `__version__`）三处必须**完全一致**。
- [ ] **同版本号不可重用**：即便已 `cargo yank` 或在 PyPI 标记 deletion，同一 `X.Y.Z` 也不允许重新发布；新版本必须严格递增。
- [ ] **用决策表自查本次升级档位**（依据「与上个 tag 之间的变更」）：

  ```
  git log --oneline $(git describe --tags --abbrev=0)..HEAD
  git diff $(git describe --tags --abbrev=0)..HEAD -- python/wbt python/wbt/__init__.py src
  ```

  | 本次变更内容                                       | 0.x 阶段        | 1.0+ 阶段 |
  |----------------------------------------------------|------------------|-----------|
  | 删除/重命名公共 API、改签名、改返回类型/语义       | MINOR + 标注 BREAKING | MAJOR |
  | 改默认参数、改异常类型、改输出列名/列顺序           | MINOR + 标注 BREAKING | MAJOR |
  | 新增公共 API、新增可选参数（带默认值）             | MINOR            | MINOR     |
  | 修 Bug、行为等价的内部重构、性能优化               | PATCH            | PATCH     |
  | 仅文档/CI/测试改动                                  | 不发版           | 不发版    |

- [ ] **破坏性变更的标注**：0.x 阶段允许在 MINOR 引入 break，但必须在 Release notes 顶部用 `**BREAKING CHANGE**` 显式标注，并提供旧→新用法对照。
- [ ] **公共 API 表面已审阅**：`python/wbt/__init__.py` 的 `__all__`、`python/wbt/_wbt.pyi` 的导出符号，与上个版本逐项 diff，每一处差异都已映射到决策表的某一行。
- [ ] `Cargo.lock` 已随 `cargo build` 同步刷新并提交；`python/pyproject.toml` 的 `requires-python`、依赖最低版本未被意外修改。

## 2. 代码静态检查（与 CI 等价的本地跑一遍）

> 这一步的意义是在 push tag 前发现 CI 会失败的问题，避免「tag 推上去才发现红 → 删 tag → 重发」造成的混乱与脏 PyPI/crates.io 状态。

Rust 侧：

- [ ] `cargo fmt --all -- --check` 通过。
- [ ] `cargo clippy --all -- -D warnings -A non_snake_case` 无 warning。
- [ ] `cargo test --lib` 全绿（与 `release-crate.yml` 一致）。
- [ ] `cargo build --release` 成功。
- [ ] `cargo publish -p wbt --dry-run` 通过（验证 manifest、license、readme 字段、依赖发布合规性）。

Python 侧（`cd python`）：

- [ ] `uv sync --extra dev` 同步成功，`uv.lock` 未出现非预期改动。
- [ ] `uv run ruff format --check .` 通过。
- [ ] `uv run ruff check . --no-fix` 通过。
- [ ] `uv run maturin develop --release` 成功，本地导入 `import wbt` 不抛错。
- [ ] `uv run basedpyright` 无新增错误。

## 3. 测试与功能验证

- [ ] `cd python && uv run pytest -v --tb=short` 全部通过。
- [ ] 关键回归用例已抽查（至少手跑或确认存在对应自动化测试）：
  - `WeightBacktest` 在含 NaN/缺失 dt 的边界输入下行为正确（参考 `test_backtest_edge.py`、`test_edge_cases.py`）；
  - `daily_performance` / `rolling_daily_performance` 与 czsc 历史口径对齐（`test_compare_with_czsc_script.py`、`test_rolling_daily_performance.py`）；
  - `cal_trade_price` 的 TWAP/VWAP 数值与尾段填充行为（`test_cal_trade_price.py`）；
  - `weights_simple_ensemble` 的 mean/vote/sum_clip 三种模式（`test_weights_simple_ensemble.py`）；
  - `generate_backtest_report` / `top_drawdowns` / `plotting` 不报错（`test_generate_backtest_report.py`、`test_plotting.py`）；
  - `test_imports.py` 覆盖到所有 `__all__` 中的符号。
- [ ] 新增/修改的公共 API 至少有一条最小测试覆盖（输入正常 + 至少一个边界）。
- [ ] 若本版本涉及性能变更：`scripts/perf_compare_real_data.py` 抽样跑一次，确认无明显回退。
- [ ] 若本版本涉及绘图/报告：`quick_start.ipynb` 至少在本地完整运行一次，肉眼检查输出。

## 4. 大模型全仓 Review（发版前强制）

> **目的**：在 push tag 前，用大模型对全仓代码做一次「冷读」式审查，捕捉单元测试覆盖不到的逻辑陷阱、隐藏边界条件与 PyO3/FFI 层的微妙误用。
> **这是发版的强制环节，不可跳过**——即便测试全绿，也必须完成本节。

- [ ] **划定 review 范围**：
  - 必查：自上个 tag 到 HEAD 的全部变更（`git diff $(git describe --tags --abbrev=0)..HEAD`）；
  - 抽查：随机挑 1–2 个本版本未改动的模块做「冷读」对照（防止旧代码因依赖升级/外部行为改变而隐性失效）。
- [ ] **向大模型提供的最少输入**：
  - 完整 diff（按文件粒度）；
  - 受影响模块的完整源码（Rust 侧 `src/core/*.rs`，Python 侧 `python/wbt/*.py`）；
  - 对应的测试文件；
  - 公共 API 表面：`python/wbt/__init__.py`、`python/wbt/_wbt.pyi`；
  - 上个版本的 Release notes（用于对比预期行为）。
- [ ] **Review 维度清单**（要求模型逐项核查、不允许「整体看起来没问题」式回答）：
  - **逻辑陷阱**：off-by-one、累加器初值错误、循环边界、状态机分支遗漏、提前 return 跳过清理；
  - **数值与边界**：NaN / Inf / None / 空序列 / 单元素 / 全相同值 / 全零 / 负数 / 极大值；
  - **时间序列特有**：未排序输入、重复时间戳、缺失 dt、时区 / 时间精度（ns vs ms）混用、跨日/跨年边界；
  - **PyO3/FFI**：GIL 释放区域内的可变状态、Py 对象生命周期、numpy / polars / pandas 之间的零拷贝与所有权陷阱、Rust 异常向 Python 的透传是否丢失上下文；
  - **并发**：`rayon` 并行段内的共享可变、累加顺序是否影响结果（浮点求和）、并行收尾的有序性；
  - **资源管理**：未关闭文件 / 上下文、循环中重复构造 DataFrame、潜在的内存膨胀；
  - **API 一致性**：新增/修改参数的默认值是否与旧版本行为兼容、错误类型是否保持稳定、列名/列顺序是否未变；
  - **可疑回归**：被删除的 `if` / 校验是否真的不需要、被改写的循环是否等价、被「化简」的表达式是否在边界情形等价。
- [ ] **每条「高 / 中」级别发现的处置**（不允许静默忽略）：
  - 修复 → 重跑 §2 lint + §3 测试 → 把修复 commit 进本次发版；
  - 或在 Release notes 中明确标注「已知问题」并提供规避方案；
  - 或论证为「误报」并记录论证理由。
- [ ] **Review 结论留痕**：把范围、所用模型、发现条目、处置结果写入 `docs/release_notes/vX.Y.Z.md`（若无可放 GitHub Release 草稿正文），便于后续追溯。

## 5. 文档与代码一致性校验（不一致以代码为准，改文档）

> **原则**：代码是事实，文档是描述。
> **任何不一致一律以代码为准，修改文档**——除非代码本身是 Bug，那应回到 §3/§4 修代码、补测试，再重跑 review。

- [ ] **公共 API 表面四方对齐**——以下四处提到的函数名、参数名、参数顺序、默认值、返回类型必须**完全一致**：
  - `python/wbt/__init__.py` 的 `__all__`；
  - `python/wbt/_wbt.pyi` 的类型签名；
  - `README.md`（中文，GitHub 主页默认）/ `README_EN.md`（英文镜像）的 API 列表与示例；
  - `python/scripts/quick_start.ipynb` 中实际调用的 API。
- [ ] **行为描述一致**：README、docstring、`_wbt.pyi` 注释中对函数行为的描述（NaN/None/异常处理、列名、单位、时间精度、错误类型）与实际实现一致。
- [ ] **示例可运行性**：README 中每个代码块都能在干净环境中复制粘贴跑通（导入路径、参数名、依赖版本匹配当前代码）。
- [ ] **版本号占位更新**：README、文档、CHANGELOG（若有）中所有「自 v0.x 起」「需要 wbt ≥ X」「Python ≥ 3.10」等占位已校对并更新到目标版本。
- [ ] **双语镜像同步**：`README.md`（中文）与 `README_EN.md`（英文）的 API 列表、示例段、minimum-version 段保持镜像一致；若仅改了一边，发版前补齐另一边。
- [ ] **破坏性变更迁移指引**：若本版本有破坏性变更，README 顶部或显著位置已写明迁移说明（旧用法 → 新用法的代码对照），并在 Release notes 中重复一次。
- [ ] **发现不一致时的标准处理流程**：
  1. 默认假设是**文档错**（代码已经通过测试 + LLM review）；
  2. 文档错 → 直接改文档 → 回到 §2 重跑 lint；
  3. 代码错 → 回到 §3 / §4 修代码 + 补测试 + 重跑 review；
  4. **绝对不允许**为了「先发版」而保留任何已知的文档 / 代码不一致；如果一致性问题在最后一刻冒出，宁可推迟发版也不容忍带病发布。

## 6. CI 全绿确认

- [ ] 在 GitHub Actions 上确认 `main` 分支最新 commit 的「CI」工作流全部 job 通过：
  - `Rust Tests`（ubuntu / macos / windows）
  - `Rust Lint`（fmt + clippy）
  - `Python Lint`（ruff）
  - `Python Type Check`（basedpyright）
  - `Python Test`（py3.10–3.13 × ubuntu / 3.13 × macos / 3.13 × windows）
- [ ] 没有任何 job 处于「pending / cancelled」状态。

## 7. 打 tag 与发布

> 顺序：先 crate，后 Python。crate 先发是因为如果 Cargo manifest 有问题（license、readme、依赖版本约束），先在 crates.io 失败比在 PyPI 失败更容易回滚（PyPI 不允许覆盖同名版本）。

- [ ] 写好本次发布的变更摘要（用于 GitHub Release notes / tag 注释）：
  - 1–3 条核心变化（feature / fix / perf / break）；
  - 破坏性变更与迁移指引（必须显式标注 `BREAKING CHANGE`）；
  - 新增/移除的公共 API；
  - §4 LLM review 的结论与「已知问题」（若有）。
- [ ] 发布 Rust crate：
  - `git tag -a crate-vX.Y.Z -m "release: crate vX.Y.Z"`
  - `git push origin crate-vX.Y.Z`
  - 观察 `Release Crate` workflow 全绿，在 [crates.io/crates/wbt](https://crates.io/crates/wbt) 确认新版本可见。
- [ ] 发布 Python 包：
  - `git tag -a vX.Y.Z -m "release: vX.Y.Z"`
  - `git push origin vX.Y.Z`
  - 观察 `Release` workflow 中 `build-wheels`（5 个矩阵）+ `build-sdist` + `publish` 均通过；
  - 在 [pypi.org/project/wbt](https://pypi.org/project/wbt/) 确认新版本与全部 wheel（linux x86_64/aarch64, macos x86_64/aarch64, windows x86_64）齐全。
- [ ] 在 GitHub Releases 页面基于 `vX.Y.Z` tag 创建 Release，贴入变更摘要。

## 8. 发布后冒烟（10 分钟内完成）

- [ ] 在一个干净的虚拟环境中：
  - `pip install -U wbt==X.Y.Z`
  - `python -c "import wbt; print(wbt.__all__)"` 输出符合预期。
  - 至少跑一次 `wbt.backtest(...)` 或 `WeightBacktest(...).backtest()` 的最小用例，确认没有 ABI / 动态库加载错误。
- [ ] 在 macOS（Apple Silicon）/ Linux / Windows 中至少 2 个平台做上述冒烟（若条件不允许，至少 1 个平台 + CI 矩阵作为佐证）。
- [ ] 若提供下游 czsc 兼容路径，触发一次下游联调脚本（`compare_with_czsc_real_data.py` 或等价）确认未回归。

## 9. 失败回滚预案

- crates.io 发布失败：
  - 同版本号不可重发。若仅 tag 失败而 crate 未发布，删除 tag 后修正再发：`git push origin :refs/tags/crate-vX.Y.Z`；
  - 若 crate 已发布但严重缺陷，直接发 `X.Y.(Z+1)` 修复版，并在 README 提示跳过坏版本。**不要**对已发布版本做覆盖式操作，必要时使用 `cargo yank`。
- PyPI 发布失败：
  - 同版本号不可重发。修复后发 `X.Y.(Z+1)`；
  - 若仅部分平台 wheel 缺失，可通过 `workflow_dispatch` 手动触发 `Release` 工作流补齐（前提是该 tag 对应的 commit 状态可重现）。
- 严重缺陷（如崩溃、数据错误）：
  - 立刻发 patch 版本修复；
  - 在 GitHub Release notes 中标注坏版本，README 顶部加临时说明。

---

## 附：日常维护节奏建议（非每次发版必查）

- 每 1–2 个版本至少跑一次 `cargo update`、`uv lock --upgrade`，关注 polars / pyo3 / numpy / pandas 的兼容性矩阵。
- 关注 `pyo3` 与 `numpy` 主版本号必须配套升级（当前均为 0.28）。
- 关注 `manylinux_2_28` 在目标用户系统上的可用性（特别是 CentOS 7 / 老 glibc 的用户）。
