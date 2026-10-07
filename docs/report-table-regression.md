# HTML 报告表格回归检查

报告生成器使用原生 HTML 表格；公开的 `plot_*_table`、指标对比和策略判定接口仍返回 Plotly Table。两条路径都需要验证，不能仅检查 HTML 文件生成成功。

截图中的 Plotly 表格使用默认约 20px 的单元格行高，但画布按 28–30px/行估算，并保留普通曲线图的大边距，造成文字拥挤和大量留白。共享布局现在统一表头 32px、单元格 30px、字体 13px，并按实际行数计算画布高度；判定表额外计入说明区高度。报告主题切换同时更新表格字体与边框，避免深色下出现默认亮色网格线。原生表格保持既有实现，不重复重写。

在 `python/` 目录、已有开发环境中执行：

```bash
uv pip install playwright
uv run --no-sync playwright install chromium
WBT_BROWSER_ARTIFACTS=./table-artifacts uv run --no-sync pytest -q tests/test_report_tables_browser.py
uv run --no-sync pytest -q tests/test_generate_backtest_report.py tests/test_table_metric_consistency.py tests/test_plotting.py
```

浏览器依赖是可选的；没有安装 Playwright 时，仅浏览器用例跳过，五类 Plotly 表格布局单测仍执行。已安装 Playwright 但没有浏览器时测试失败，不会静默跳过。可以用 `WBT_BROWSER_EXECUTABLE` 指定已有 Chromium 可执行文件。

固定随机种子的四年 mock 数据覆盖正负收益、history/recent 判定、年度明细、Top 10 回撤、完整绩效指标、分段比较及多空比较。Playwright 检查实际标签切换、明暗主题、表格可见性和行高、收益着色、移动端及 JavaScript 错误，并保存截图和 HTML。浏览器检查需要能够加载报告原有的 Bootstrap CDN 资源；不依赖行情接口或数据库。
