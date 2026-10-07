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

## 移动端与实际渲染边界

报告内的 Plotly 表格保留至少 1000px 的可读宽度，并在自身容器内横向滚动；普通曲线图不受影响。表格初始化后及主题切换时重新测量宽度，隐藏标签页留待显示时 resize，避免隐藏元素的 resize 错误。无需改变数据或把数值缩小到不可读。

浏览器用例覆盖 1440px、390px、320px × 明暗两种主题，对五类 Plotly 表格逐一检查真实 SVG：每个表头/单元格的文本包围盒必须落在单元格矩形内，矩形不能超出 SVG；渲染行列数必须完整，最后一行必须落在表格裁切区域内；策略说明必须落在 SVG 内且不能与表头重叠；表头与正文矩形的实际 stroke 必须符合当前主题。移动端滚动到左右两端，验证首尾列实际可见，并分别保存截图。

原生 HTML 表格在相同视口/主题组合下检查文本与单元格边界、行列完整性、末行、实际边框颜色以及右端内容可滚动查看。另有隐藏标签页表格用例，验证显示后的真实边界及 JavaScript 错误。

限制：窄屏需要横向滚动，不承诺所有列同屏显示；这些检查针对内置五类表格及固定 mock 内容，不承诺任意超长自定义字段或无空格说明均可自动换行。独立调用 Plotly `Figure.to_html()` 而不经过报告构建器不会获得报告的滚动 CSS。这里只实际执行 Chromium，不将结果外推至其他浏览器；CSS 使用现代浏览器的 `:has()`。
