# 原生 HTML 回测报告 · 设计与验收

本次按用户更正后的要求重设计整页，不沿用早先 Plotly Table 补丁的验收结论。实际安装并使用 Anthropic `frontend-design` 技能，以策略研究报告为方向：蓝灰底色、清晰的标题与数字层级、统一留白、轻量分隔、红涨绿跌的收益语义及中性的风险指标。布局适配明暗主题与窄屏。

## 实现边界

- 报告生成器只用语义化 HTML 表格：年度审核、近期窗口、完整绩效、回撤、分段与多空比较，以及可展开判定字段。`caption`、列/行标题与 `scope` 提供清晰结构；原生 HTML 表格是默认路径，不是 Plotly Table 的替代展示层。
- 所有绩效字段（包括开始/结束日期）和年度完整性标记保留，判定理由显示为可换行正文。补齐历史超额回撤/夏普及两项达标条件的中文标签，未注册的判定字段也保留显示；年度嵌套数据由年度表完整展示，不重复塞入字段表。数据与回测计算不变。
- 曲线、分布及热力图保留 Plotly，脚本仅内联一次；不依赖 Bootstrap、外部字体或图标 CDN，生成文件可离线查看。
- 章节导航支持左右箭头、Home、End，按钮可用 Enter/Space；主题状态通过 `aria-pressed` 表达，标签页通过 `aria-selected`、关联 ID 与 roving tabindex 表达；支持跳至正文、原生 details 与可聚焦滚动表格。
- 窄屏表格在自身容器横向滚动，日期/数值不被强制压缩；策略说明可自然换行，不进入 SVG。不会引入整页横向溢出。
- 公共 `wbt.plotting` 的独立 Table 接口仍保留兼容性，但不用于回测报告。已撤回 PR 早先针对 Plotly Table 的共享行高/画布修改。

## 复现

在 `python/` 的开发环境中执行：

```bash
uv pip install playwright
uv run --no-sync playwright install chromium
WBT_BROWSER_ARTIFACTS=./report-artifacts uv run --no-sync pytest -q tests/test_report_tables_browser.py
uv run --no-sync pytest -q tests/test_generate_backtest_report.py tests/test_table_metric_consistency.py tests/test_plotting.py
```

可用 `WBT_BROWSER_EXECUTABLE` 指定已有 Chromium。没有安装 Playwright 时浏览器用例会跳过，验收时必须实际执行；已安装但没有浏览器时直接失败，不静默跳过。

固定种子的四年 mock 数据不依赖数据库或行情接口。浏览器阻断 HTTPS 请求，核实不产生外部网络请求；1440/390/320px × 明暗主题 × 五个章节保存整页截图。精确逐格核对所有八张原生表格的表头、行数、列数和显示值（直接对照数据源，而非另一个 HTML 渲染器），包括展开的两张完整判定表。检查真实文字/单元格边界、末行、实际边框、横向滚动到末列的可达性、说明完整与换行、键盘导航/焦点、主题持久化及 JavaScript 错误。收益曲线渲染的完整数据点与数据源比较；报告前后完整 JSON 快照必须一致，且 Plotly 中不允许出现 Table trace。

## 限制

实测浏览器为 Chromium，不代表 Safari/Firefox 独立验证通过。窄屏数据列需要横向滚动，不承诺所有列同屏。截图反映 mock 数据，不代表生产数据；极端长标题/自定义字段应另行验证。打印样式展开章节，但 PDF/打印分页未验收。Rust 扩展未因展示层修改而重建，使用同版本 0.9.1 wheel 验证；远端 CI 与合并门禁以远端状态为准，不从本地结果推断。
# 五层分析升级（2026-10-07）

本轮使用已安装的 frontend-design 技能，将报告改为研究工作台：桌面侧栏、手机横向章节导航；
摘要从 14 项收敛为 4 项，累计收益轨迹成为概览主图，完整字段放入策略诊断。
章节为概览、收益表现、风险回撤、交易与持仓、策略诊断。

## 数据来源与口径

| 内容 | BacktestResult 来源 | 口径 / 限制 |
| --- | --- | --- |
| 摘要 | stats | 缺失显示 —，不默认零 |
| 样本 | dates、symbol_count、weight_type、yearly_days | 观测日数量，不推断缺失交易日 |
| 收益轨迹 | curves.daily/cum | 算术累加，不是复利净值 |
| 月度收益 | monthly.z/text + dates | 原始月度和；未观测月份图表留空、表格显示 — |
| 年度表现 | yearly_returns、verdict.yearly_metrics | 保留引擎绝对/超额收益及完整年度标识 |
| 品种收益 | symbol_returns | 品种收益和，不是加权组合贡献 |
| 风险 | curves.drawdown、drawdowns、return_dist、rolling | 滚动窗口 252 日，最少 100 日；无样本自然降级 |
| 交易 | pairs_dist、key_trades | 聚合成交对；盈亏分布已为百分数，关键交易 pnl 为小数 |
| 持仓 | pairs_dist.holds、key_trades.hold_bars | K 线长度；快照没有逐日仓位，不展示虚构当前持仓 |
| 诊断 | verdict/recent、segment_comparison、stats、curves_voladj | 归一曲线使用快照数据，不改引擎计算 |

新增月度、品种和关键交易原生表格，报告共 11 张语义化表格。月度热力图在窄屏可键盘滚动，
避免强行压缩 12 个月文字。不存在完整交易流水，不将关键交易选样称作全部交易。

## 本轮验证范围

固定 seed 905 的四年 mock，Playwright 以 file URL 打开并拦截外网请求。
1440、390、320 像素 × 明暗主题 × 五章节，逐表核对源字段、列/行数及格式，
检查文字与单元格真实边界、末行、实际主题边框、横向键盘滚动、说明文字边界。
核对策略/基准实际绘制曲线、热力图、成交盈亏/持仓分布；保留主题持久化、
Tab/箭头/Home/End、详情展开与跳转正文检查。生成前后结果快照一致。
另覆盖短样本零仓位、缺失月份和空图的自然降级。

限制：浏览器自动化为 Chromium；尚未在 Safari/Firefox、真实触屏及屏幕阅读器验收。
视觉审美由用户验收；独立审核不以此前版本结论替代。
