# 报告中的 Plotly 表格

公共绘图函数仍返回 Plotly `Figure`，其数据、布局与回测计算不变。
`HtmlReportBuilder.add_section`、`add_chart_tab`、`add_chart_grid_tab` 接收 HTML，
最终由 `render()` 组合报告。在这个公共输出边界，标准 `Figure.to_html` 生成的
单一 Table 面板被转换为原生 HTML 表格，复用 `_html_tables._fin_table`。
标准 `generate_backtest_report` 已使用原生表格，未改造该入口。

```python
builder.add_chart_tab(
    "指标", figure.to_html(full_html=False, include_plotlyjs=False), active=True
).add_charts_section()
html = builder.render()
```

转换保留已有表头、单元格字符串、列/行顺序、标题和说明，使用文本转义及有限的
富文本标签白名单。只对转换后的面板添加样式和可键盘聚焦的局部横向滚动区域。
转换后的面板外层固定高度局部改为自适应，避免内容增长后被网格卡片裁切；
使用现代浏览器的 CSS `:has()`，不改变其他图表容器。
非表格图表的 HTML 保持原样；原生表格与标准报告的布局不变。

仅转换单 Table 中已预格式化的字符串表头/单元格；不自行实现 Plotly/D3 格式化。
header/cells 显式设置 `format`、`prefix`、`suffix`（包括空设置）时保留原片段；
原始数值、非默认 Table/annotation 可见性、显式列顺序/定位 domain、
模板内的格式或隐藏设置、模板说明及 annotation 模板引用也保留原片段。
说明的非默认透明度、点击显示设置以及交互菜单/滑块同样回退，避免强制显示内容
或丢弃原交互。
这是一种保守回退：不支持的表格继续由 Plotly 渲染，原有百分比、金额、前后缀和
隐藏语义保持不变，但这些回退表格不会获得原生表格的窄屏可读性修复。

支持标准 JSON 参数的单 Table `Plotly.newPlot` 片段。混合图、回调/`post_script`、
不支持的编码或不规则列形状保持原样，避免丢失行为。绕过 builder 的外部自定义
HTML 生成器不受此边界处理；本修复不会追溯修改已有报告文件。

## 回归验证

在 `python` 目录使用安装了测试依赖和 Playwright 的 Python：

```sh
python tests/manual/prepare_report_browser_assets.py /path/to/assets
WBT_BROWSER_ASSETS=/path/to/assets \
WBT_BROWSER_ARTIFACTS=/path/to/artifacts \
WBT_BROWSER_EXECUTABLE=/path/to/chromium \
python -m pytest -q tests/test_report_plotly_boundary.py
```

浏览器测试使用固定模拟数据重建五个表格、十一张其他图表，验证 1440/390/320
宽度与明暗主题下的数据映射、文字边界、说明、末行、键盘滚动和主题边框，
并对照未经转换的报告检查其他图表的数据、布局和位置。
兼容性测试额外逐字对比 Plotly 的实际 SVG 显示，断言百分比、金额、表头/单元格
前后缀不变，隐藏 Table 与说明不出现，回退输入字节不变；生成独立的
`mock-compatibility.html` 和兼容性检查记录。内置五类表格仍由前述原生表格测试覆盖。
仅测试产物内联原报告依赖以提供可离线打开的 `mock-table-fix.html`；
产品报告的依赖加载方式未改变。测试阻断所有外部 HTTP(S) 请求。
未设置浏览器资源目录时浏览器测试明确跳过，不能据此声称渲染验证通过。
