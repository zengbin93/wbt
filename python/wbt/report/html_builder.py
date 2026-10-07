"""
HTML 报告构建器

提供灵活的 HTML 报告生成功能，支持链式调用和按需添加内容元素。

"""

from __future__ import annotations

import html
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from ._styles import REPORT_CSS


class HtmlReportBuilder:
    """HTML 报告构建器

    支持链式调用，按需添加各种 HTML 元素，生成美观的 HTML 报告。

    示例用法：
        builder = HtmlReportBuilder(title="我的报告")
        builder.add_header({"日期": "2024-01-01", "版本": "v1.0"}) \\
               .add_metrics([{"label": "收益率", "value": "15.3%", "is_positive": True}]) \\
               .add_section("简介", "<p>这是报告内容</p>") \\
               .save("report.html")
    """

    def __init__(self, title: str = "HTML 报告", theme: str = "light"):
        """初始化 HTML 报告构建器

        :param title: 报告标题
        :param theme: 主题，可选 'light' 或 'dark'
        """
        self.title = title
        self.theme = theme
        self.sections: list[tuple[str, Any]] = []  # 存储所有内容区域
        self.custom_css: list[str] = []  # 自定义 CSS
        self.custom_scripts: list[str] = []  # 自定义脚本
        self.chart_count = 0  # 图表计数器，用于生成唯一ID
        self._init_default_styles()

    def _init_default_styles(self) -> None:
        self.base_css = REPORT_CSS

    def add_custom_css(self, css: str) -> HtmlReportBuilder:
        """添加自定义 CSS 样式

        :param css: CSS 字符串
        :return: self，支持链式调用
        """
        self.custom_css.append(css)
        return self

    def add_custom_script(self, script: str) -> HtmlReportBuilder:
        """添加自定义 JavaScript 脚本

        :param script: JavaScript 字符串
        :return: self，支持链式调用
        """
        self.custom_scripts.append(script)
        return self

    def add_header(self, params: dict[str, str], subtitle: str | None = None) -> HtmlReportBuilder:
        """添加头部区域

        :param params: 参数字典，如 {"日期": "2024-01-01", "版本": "v1.0"}
        :param subtitle: 副标题
        :return: self，支持链式调用
        """
        badges_html = ""
        for key, value in params.items():
            badges_html += f'                        <span class="param-badge">{html.escape(key)} <b>{html.escape(value)}</b></span>\n'

        subtitle_html = f'<p class="header-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
        header_html = f"""    <!-- 头部区域 -->
    <header class="header-section">
        <div class="container">
            <div class="header-bar">
                <div>
                    <h1 class="header-title">{html.escape(self.title)}</h1>
                    {subtitle_html}
                </div>
                <div class="theme-switch" role="group" aria-label="主题切换">
                    <button type="button" data-theme="light"><i class="bi bi-sun"></i> 浅色</button>
                    <button type="button" data-theme="dark"><i class="bi bi-moon-stars"></i> 深色</button>
                </div>
            </div>
            <div class="param-badges">
{badges_html}            </div>
        </div>
    </header>
"""

        self.sections.append(("header", header_html))
        return self

    def add_metrics(self, metrics: list[dict[str, Any]], title: str = "核心绩效指标") -> HtmlReportBuilder:
        """添加绩效指标卡片

        :param metrics: 指标列表，每个元素为 {"label": str, "value": str, "is_positive": bool}；
            可选 "neutral": bool —— True 时用中性蓝（适合占比/持仓等无涨跌语义的结构指标）
        :param title: 区域标题
        :return: self，支持链式调用
        """
        tiles_html = ""
        for m in metrics:
            if m.get("neutral"):
                value_class = "metric-neutral"
            else:
                value_class = "metric-positive" if m.get("is_positive", False) else "metric-negative"
            tiles_html += f"""                <div class="stat-tile">
                    <span class="stat-label">{m["label"]}</span>
                    <span class="stat-value {value_class}">{m["value"]}</span>
                </div>\n"""

        # 选择能整除指标数的列数（优先大列数），让每行填满、末行无空位（14→7×2）
        n = len(metrics)
        cols = next((c for c in (7, 6, 5, 4) if n and n % c == 0), 5)
        section_html = f"""    <!-- {title} -->
    <section>
        <div class="section-header">
            <i class="bi bi-speedometer2 section-icon"></i>
            <h2 class="section-title">{title}</h2>
        </div>

        <div class="stat-grid" style="grid-template-columns: repeat({cols}, 1fr);">
{tiles_html}        </div>
    </section>
"""

        self.sections.append(("metrics", section_html))
        return self

    def add_chart_tab(
        self, name: str, chart_html: str, icon: str = "bi-graph-up", active: bool = False
    ) -> HtmlReportBuilder:
        """添加单个图表标签页

        :param name: 标签页名称
        :param chart_html: 图表 HTML 内容
        :param icon: 图标类名（Bootstrap Icons）
        :param active: 是否为默认激活的标签页
        :return: self，支持链式调用
        """
        self.chart_count += 1
        tab_id = f"chart-tab-{self.chart_count}"

        tab_button = f"""                        <li class="nav-item">
                            <button class="nav-link {"active" if active else ""}"
                                    data-bs-toggle="tab" data-bs-target="#{tab_id}"
                                    type="button" role="tab" id="{tab_id}-button"
                                    aria-controls="{tab_id}" aria-selected="{str(active).lower()}" tabindex="{0 if active else -1}">
                                <i class="bi {icon}"></i> {name}
                            </button>
                        </li>"""

        tab_content = f"""                        <div class="tab-pane fade {"show active" if active else ""}"
                                          id="{tab_id}" role="tabpanel" aria-labelledby="{tab_id}-button" tabindex="0">
                            <div class="chart-body">
                                {chart_html}
                            </div>
                        </div>"""

        self.sections.append(("chart_tab", {"button": tab_button, "content": tab_content}))
        return self

    def add_chart_grid_tab(
        self,
        name: str,
        charts: Sequence[str | tuple[str, str] | tuple[str, str, bool]],
        cols: int = 2,
        icon: str = "bi-graph-up",
        active: bool = False,
    ) -> HtmlReportBuilder:
        """添加一个内部以 CSS 网格排布多张图表的标签页。

        :param name: 标签页名称
        :param charts: 图表列表；元素可为：图表 HTML 字符串、``(小标题, 图表 HTML)``
            二元组，或 ``(小标题, 图表 HTML, 是否整行跨列)`` 三元组（最后一项为 True 时该图占满整行）
        :param cols: 网格列数（移动端自动退化为单列）
        :param icon: 图标类名（Bootstrap Icons）
        :param active: 是否为默认激活的标签页
        :return: self，支持链式调用
        """
        self.chart_count += 1
        tab_id = f"chart-tab-{self.chart_count}"

        items_html = ""
        for chart in charts:
            if isinstance(chart, str):
                sub_title, chart_html, full_width = "", chart, False
            else:
                sub_title, chart_html = chart[0], chart[1]
                full_width = chart[2] if len(chart) == 3 else False
            title_html = f'<h3 class="chart-grid-title">{html.escape(sub_title)}</h3>' if sub_title else ""
            item_class = "chart-grid-item full-width" if full_width else "chart-grid-item"
            items_html += f'                                <div class="{item_class}">{title_html}{chart_html}</div>\n'

        tab_button = f"""                        <li class="nav-item">
                            <button class="nav-link {"active" if active else ""}"
                                    data-bs-toggle="tab" data-bs-target="#{tab_id}"
                                    type="button" role="tab" id="{tab_id}-button"
                                    aria-controls="{tab_id}" aria-selected="{str(active).lower()}" tabindex="{0 if active else -1}">
                                <i class="bi {icon}"></i> {name}
                            </button>
                        </li>"""

        tab_content = f"""                        <div class="tab-pane fade {"show active" if active else ""}"
                                          id="{tab_id}" role="tabpanel" aria-labelledby="{tab_id}-button" tabindex="0">
                            <div class="chart-grid" style="grid-template-columns: repeat({cols}, 1fr);">
{items_html}                            </div>
                        </div>"""

        self.sections.append(("chart_tab", {"button": tab_button, "content": tab_content}))
        return self

    def add_charts_section(self, title: str = "可视化分析") -> HtmlReportBuilder:
        """添加图表展示区域（收集所有之前添加的图表标签页）

        :param title: 区域标题
        :return: self，支持链式调用
        """
        chart_tabs = [section for section in self.sections if section[0] == "chart_tab"]

        if not chart_tabs:
            return self

        tabs_html = (
            '                <div class="chart-card">\n                    <ul class="nav nav-tabs" role="tablist">\n'
        )
        tabs_html += "\n".join([tab[1]["button"] for tab in chart_tabs])
        tabs_html += "\n                    </ul>\n"

        content_html = '                    <div class="tab-content">\n'
        content_html += "\n".join([tab[1]["content"] for tab in chart_tabs])
        content_html += "\n                    </div>\n                </div>"

        section_html = f"""    <!-- {title} -->
    <section class="mb-4">
        <div class="section-header">
            <i class="bi bi-bar-chart-line section-icon"></i>
            <h2 class="section-title">{title}</h2>
        </div>

{tabs_html}
{content_html}
    </section>
"""

        self.sections = [s for s in self.sections if s[0] != "chart_tab"]
        self.sections.append(("charts_section", section_html))

        return self

    def add_table(
        self,
        df: pd.DataFrame,
        title: str = "数据表",
        max_rows: int | None = None,
        style: str = "Table Grid",
    ) -> HtmlReportBuilder:
        """添加数据表格

        :param df: pandas DataFrame
        :param title: 表格标题
        :param max_rows: 最大显示行数，None 表示全部显示
        :param style: 表格样式（保留参数以兼容 czsc 接口）
        :return: self，支持链式调用
        """
        del style  # 预留以兼容原 czsc 签名
        if df.empty:
            return self

        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)

        table_html = df.to_html(classes="table table-striped table-hover", index=False, border=0, justify="center")

        section_html = f"""    <!-- {title} -->
    <section class="mb-4">
        <div class="section-header">
            <i class="bi bi-table section-icon"></i>
            <h2 class="section-title">{title}</h2>
        </div>

        <div class="data-table">
            {table_html}
        </div>
    </section>
"""

        self.sections.append(("table", section_html))
        return self

    def add_section(self, title: str, content: str, icon: str = "bi-file-text") -> HtmlReportBuilder:
        """添加自定义章节

        :param title: 章节标题
        :param content: 章节内容（HTML字符串）
        :param icon: 图标类名
        :return: self，支持链式调用
        """
        section_html = f"""    <!-- {title} -->
    <section class="mb-4">
        <div class="section-header">
            <i class="bi {icon} section-icon"></i>
            <h2 class="section-title">{title}</h2>
        </div>

        <div class="section-content">
            {content}
        </div>
    </section>
"""

        self.sections.append(("custom", section_html))
        return self

    def add_footer(self, text: str | None = None) -> HtmlReportBuilder:
        """添加页脚

        :param text: 页脚文本，None 则使用默认文本
        :return: self，支持链式调用
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if text is None:
            text = (
                '<i class="bi bi-code-square"></i> '
                "由 wbt 权重回测引擎生成 | "
                f'<i class="bi bi-clock-history"></i> 生成时间: {current_time}'
            )

        footer_html = f"""    <!-- 页脚 -->
    <footer class="footer">
        <div class="container">
            <p class="mb-0">
                {text}
            </p>
        </div>
    </footer>
"""

        self.sections.append(("footer", footer_html))
        return self

    def render(self) -> str:
        """渲染完整的 HTML 报告

        :return: HTML 字符串
        """
        # 兜底：若调用方添加了图表标签页却忘了 add_charts_section()，自动收口，
        # 否则这些 chart_tab（以 dict 暂存）会被下方静默跳过，生成无图报告。
        if any(s[0] == "chart_tab" for s in self.sections):
            self.add_charts_section()

        all_css = self.base_css + "\n" + "\n".join(self.custom_css)

        header_html = ""
        footer_html = ""
        main_body_html = ""

        for section_type, section_content in self.sections:
            if isinstance(section_content, dict):
                continue  # 跳过未处理的图表标签页

            if section_type == "header":
                header_html += section_content + "\n"
            elif section_type == "footer":
                footer_html += section_content + "\n"
            else:
                main_body_html += section_content + "\n"

        custom_scripts_str = "\n".join(self.custom_scripts)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.title)}</title>

    <!-- 主题初始化（首屏前执行，避免闪烁）：localStorage 优先，默认深色 -->
    <script>
        (function () {{
            var t = '{self.theme if self.theme in ("light", "dark") else "light"}';
            try {{ t = localStorage.getItem('wbt-theme') || t; }} catch (e) {{}}
            if (t !== 'light' && t !== 'dark') t = 'light';
            document.documentElement.setAttribute('data-theme', t);
        }})();
    </script>

    <style>
{all_css}
    </style>
</head>
<body>
    <a class="skip-link" href="#report-content">跳至报告正文</a>
{header_html}
    <main class="container main-content" id="report-content" tabindex="-1">
{main_body_html}
    </main>
{footer_html}

    <script>
        // ---- Plotly 主题同步：图表底色/网格/字体/表格跟随明暗主题 ----
        function wbtPlotlyColors(theme) {{
            return theme === 'dark'
                ? {{ font: '#a6b7cb', grid: '#314359', zero: '#566c86',
                     line: '#566c86', bar: '#a6b7cb', active: '#94c9ef' }}
                : {{ font: '#52647a', grid: '#dde4ec', zero: '#b9c6d5',
                     line: '#b9c6d5', bar: '#52647a', active: '#174d75' }};
        }}
        function wbtApplyPlotlyTheme(theme) {{
            if (typeof Plotly === 'undefined') return;
            var c = wbtPlotlyColors(theme);
            document.querySelectorAll('.plotly-graph-div').forEach(function (d) {{
                if (!d || !d.layout) return;
                var up = {{ 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
                            'font.color': c.font, 'legend.font.color': c.font,
                            'modebar.bgcolor': 'rgba(0,0,0,0)', 'modebar.color': c.bar, 'modebar.activecolor': c.active }};
                Object.keys(d.layout).forEach(function (k) {{
                    if (/^(xaxis|yaxis)/.test(k)) {{
                        up[k + '.gridcolor'] = c.grid;
                        up[k + '.zerolinecolor'] = c.zero;
                        up[k + '.linecolor'] = c.line;
                        up[k + '.tickfont.color'] = c.font;
                    }}
                }});
                try {{ Plotly.relayout(d, up); }} catch (e) {{}}
            }});
        }}

        function wbtSetTheme(t) {{
            document.documentElement.setAttribute('data-theme', t);
            try {{ localStorage.setItem('wbt-theme', t); }} catch (e) {{}}
            document.querySelectorAll('.theme-switch button').forEach(function (b) {{
                b.classList.toggle('active', b.getAttribute('data-theme') === t);
                b.setAttribute('aria-pressed', String(b.getAttribute('data-theme') === t));
            }});
            wbtApplyPlotlyTheme(t);
        }}

        document.addEventListener('DOMContentLoaded', function () {{
            var theme = document.documentElement.getAttribute('data-theme') || 'dark';
            document.querySelectorAll('.theme-switch button').forEach(function (b) {{
                b.classList.toggle('active', b.getAttribute('data-theme') === theme);
                b.setAttribute('aria-pressed', String(b.getAttribute('data-theme') === theme));
                b.addEventListener('click', function () {{ wbtSetTheme(b.getAttribute('data-theme')); }});
            }});

            function resizePane(pane) {{
                if (!pane || typeof Plotly === 'undefined') return;
                pane.querySelectorAll('.plotly-graph-div').forEach(function (d) {{ Plotly.Plots.resize(d); }});
            }}
            function updateTableHints() {{
                document.querySelectorAll('.fin-wrap').forEach(function (region) {{
                    if (region.clientWidth) region.dataset.overflow = String(region.scrollWidth > region.clientWidth);
                }});
            }}
            function resizeActivePanes() {{
                document.querySelectorAll('.tab-pane.active').forEach(resizePane);
                updateTableHints();
            }}
            document.querySelectorAll('details').forEach(function (details) {{
                details.addEventListener('toggle', updateTableHints);
            }});
            document.querySelectorAll('[role="tablist"]').forEach(function (list) {{
                var tabs = Array.from(list.querySelectorAll('[role="tab"]'));
                function activate(target) {{
                    tabs.forEach(function (tab) {{
                        var selected = tab === target;
                        tab.classList.toggle('active', selected);
                        tab.setAttribute('aria-selected', String(selected));
                        tab.tabIndex = selected ? 0 : -1;
                        var pane = document.getElementById(tab.getAttribute('aria-controls'));
                        pane.classList.toggle('active', selected);
                        pane.classList.toggle('show', selected);
                        pane.hidden = !selected;
                    }});
                    target.scrollIntoView({{ block: 'nearest', inline: 'nearest' }});
                    target.dispatchEvent(new Event('shown.bs.tab'));
                    updateTableHints();
                }}
                tabs.forEach(function (tab, index) {{
                    tab.addEventListener('click', function () {{ activate(tab); }});
                    tab.addEventListener('keydown', function (event) {{
                        var next;
                        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
                        else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
                        else if (event.key === 'Home') next = 0;
                        else if (event.key === 'End') next = tabs.length - 1;
                        else return;
                        event.preventDefault();
                        activate(tabs[next]);
                        tabs[next].focus({{ preventScroll: true }});
                    }});
                }});
            }});
            document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(function (el) {{
                el.addEventListener('shown.bs.tab', function (ev) {{
                    resizePane(document.querySelector(ev.target.getAttribute('data-bs-target')));
                }});
            }});
            window.addEventListener('resize', resizeActivePanes);

            // 每个图表区的首个标签页均不会触发 shown.bs.tab，载入后重算所有可见容器。
            resizeActivePanes();

            // 初次着色图表以匹配当前主题（plotly 渲染稍晚，window.load 再补一次）
            wbtApplyPlotlyTheme(theme);
            window.addEventListener('load', function () {{
                resizeActivePanes();
                wbtApplyPlotlyTheme(theme);
            }});
        }});

        // 用户自定义脚本
        {custom_scripts_str}
    </script>
</body>
</html>
"""

    def save(self, file_path: str) -> str:
        """保存 HTML 报告到文件

        :param file_path: 输出文件路径
        :return: 文件路径
        """
        html_content = self.render()

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return file_path
