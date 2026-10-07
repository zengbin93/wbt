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
