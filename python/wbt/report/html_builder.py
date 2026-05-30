"""
HTML 报告构建器

提供灵活的 HTML 报告生成功能，支持链式调用和按需添加内容元素。

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd


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
        """初始化默认样式（Quant Terminal 设计系统，双主题）。"""
        self.base_css = """
        /* ============ 主题变量：机构研报(light) / 交易终端(dark) ============ */
        [data-theme="light"] {
            --bg: #f6f7f9;
            --panel: #ffffff;
            --panel-2: #eef1f5;
            --ink: #0e1116;
            --muted: #5b6573;
            --border: #e4e8ee;
            --border-strong: #cfd6e0;
            --accent: #2f5fef;
            --up: #d6233b;      /* 红涨：正收益/盈利 */
            --down: #0b9d6f;    /* 绿跌：负收益/亏损 */
            --shadow: 0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.05);
            --dot: rgba(14,17,22,0.035);
        }
        [data-theme="dark"] {
            --bg: #0b0e15;
            --panel: #131722;
            --panel-2: #1b2130;
            --ink: #e6e9ef;
            --muted: #868fa3;
            --border: #232b3b;
            --border-strong: #313b50;
            --accent: #5b8cff;
            --up: #f6465d;
            --down: #1fc995;
            --shadow: 0 1px 2px rgba(0,0,0,.45);
            --dot: rgba(230,233,239,0.045);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { width: 100%; overflow-x: hidden; }
        html { scroll-behavior: smooth; }

        body {
            background-color: var(--bg);
            background-image: radial-gradient(var(--dot) 1px, transparent 1px);
            background-size: 22px 22px;
            color: var(--ink);
            font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            -webkit-font-smoothing: antialiased;
            transition: background-color .25s ease, color .25s ease;
        }

        .container { max-width: 1440px; width: 94%; padding: 0 8px; margin: 0 auto; }
        .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }

        /* ============ Header ============ */
        .header-section {
            border-bottom: 1px solid var(--border);
            padding: 1.6rem 0 1.3rem;
            margin-bottom: .4rem;
        }
        .header-bar { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
        .header-title {
            display: flex; align-items: center; gap: .6rem;
            font-size: 1.5rem; font-weight: 600; letter-spacing: -0.015em; color: var(--ink);
        }
        .brand-mark {
            width: 12px; height: 22px; border-radius: 2px;
            background: linear-gradient(180deg, var(--up) 0 50%, var(--down) 50% 100%);
            display: inline-block; flex: none;
        }
        .header-subtitle { color: var(--muted); font-size: .85rem; margin-top: .35rem; }
        .param-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: 1rem; }
        .param-badge {
            font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: var(--muted);
            background: var(--panel-2); border: 1px solid var(--border);
            border-radius: 5px; padding: .3rem .6rem; white-space: nowrap;
        }
        .param-badge b { color: var(--ink); font-weight: 500; }

        /* ============ Theme switch ============ */
        .theme-switch {
            display: inline-flex; flex: none; border: 1px solid var(--border);
            border-radius: 7px; overflow: hidden; background: var(--panel-2);
        }
        .theme-switch button {
            background: transparent; border: 0; color: var(--muted); cursor: pointer;
            padding: .42rem .7rem; font-size: .78rem; font-family: inherit;
            display: inline-flex; align-items: center; gap: .35rem; transition: all .15s;
        }
        .theme-switch button:hover { color: var(--ink); }
        .theme-switch button.active { background: var(--accent); color: #fff; }

        .main-content { flex: 1; padding-bottom: 2.5rem; }

        /* ============ Section header ============ */
        .section-header { display: flex; align-items: center; gap: .55rem; margin: 1.5rem 0 .5rem; }
        .section-header .section-icon { color: var(--accent); font-size: .95rem; }
        .section-title {
            font-size: .82rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
            color: var(--muted); margin: 0;
        }
        .section-header::after { content: ""; flex: 1; height: 1px; background: var(--border); margin-left: .3rem; }

        /* ============ Stat tiles ============ */
        .stat-grid {
            display: grid; gap: 1px; background: var(--border); border: 1px solid var(--border);
            border-radius: 9px; overflow: hidden;
            /* 列数由 add_metrics 按指标数取整除值内联设置，保证每行填满、无空位 */
        }
        .stat-tile {
            background: var(--panel); padding: .8rem .95rem; display: flex; flex-direction: column;
            gap: .35rem; position: relative; transition: background .15s;
        }
        .stat-tile:hover { background: var(--panel-2); }
        .stat-label { font-size: .67rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
        .stat-value {
            font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
            font-weight: 500; font-size: 1.3rem; letter-spacing: -0.01em; line-height: 1.1;
        }
        .stat-value.metric-positive { color: var(--up); }
        .stat-value.metric-negative { color: var(--down); }
        .stat-value.metric-neutral  { color: var(--ink); }

        /* ============ Tabs (override bootstrap) ============ */
        .chart-card { background: transparent; border: 0; box-shadow: none; }
        .nav-tabs {
            position: sticky; top: 0; z-index: 40; display: flex; gap: .15rem;
            border: 0; border-bottom: 1px solid var(--border);
            background: color-mix(in srgb, var(--bg) 88%, transparent);
            backdrop-filter: blur(8px); padding-top: .3rem;
        }
        .nav-tabs .nav-link {
            border: 0 !important; border-bottom: 2px solid transparent !important; border-radius: 0;
            background: transparent; color: var(--muted); font-size: .85rem; font-weight: 500;
            padding: .65rem .95rem; transition: color .15s, border-color .15s;
        }
        .nav-tabs .nav-link:hover { color: var(--ink); background: transparent; }
        .nav-tabs .nav-link.active {
            color: var(--accent) !important; background: transparent;
            border-bottom-color: var(--accent) !important;
        }
        .tab-content { background: transparent; }

        /* ============ Chart grid panels ============ */
        .chart-grid { display: grid; gap: 14px; padding: 16px 0; }
        .chart-grid-item {
            background: var(--panel); border: 1px solid var(--border); border-radius: 9px;
            overflow: hidden; box-shadow: var(--shadow); min-width: 0;
        }
        .chart-grid-item.full-width { grid-column: 1 / -1; }
        .chart-grid-item .plotly-graph-div { width: 100% !important; }
        .chart-grid-title {
            font-size: .8rem; font-weight: 600; letter-spacing: .01em; color: var(--ink);
            padding: .65rem .9rem; border-bottom: 1px solid var(--border); background: var(--panel-2);
        }

        /* ============ add_table fallback ============ */
        .data-table { background: var(--panel); border: 1px solid var(--border); border-radius: 9px; overflow: hidden; }
        .table { color: var(--ink); margin-bottom: 0; font-size: .88rem; }
        .table thead th {
            background: var(--panel-2); border-bottom: 1px solid var(--border); color: var(--muted);
            font-weight: 600; padding: .7rem; text-transform: uppercase; font-size: .72rem; letter-spacing: .04em;
        }
        .table tbody tr { border-bottom: 1px solid var(--border); }
        .table tbody tr:hover { background: var(--panel-2); }
        .table tbody td { padding: .65rem .7rem; vertical-align: middle; font-family: 'IBM Plex Mono', monospace; }

        /* ============ Footer ============ */
        .footer {
            margin-top: auto; border-top: 1px solid var(--border); padding: 1.1rem 0; text-align: center;
            color: var(--muted); font-size: .76rem; font-family: 'IBM Plex Mono', monospace;
        }

        @media (max-width: 768px) {
            .header-title { font-size: 1.25rem; }
            .header-bar { flex-direction: column; }
            .stat-value { font-size: 1.12rem; }
            .nav-tabs .nav-link { padding: .55rem .7rem; font-size: .8rem; }
            .chart-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 1024px) { .stat-grid { grid-template-columns: repeat(2, 1fr) !important; } }
        @media (max-width: 520px) { .stat-grid { grid-template-columns: 1fr !important; } }
        """

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
            badges_html += f'                        <span class="param-badge">{key} <b>{value}</b></span>\n'

        subtitle_html = f'<p class="header-subtitle">{subtitle}</p>' if subtitle else ""
        header_html = f"""    <!-- 头部区域 -->
    <div class="header-section">
        <div class="container">
            <div class="header-bar">
                <div>
                    <h1 class="header-title"><span class="brand-mark"></span>{self.title}</h1>
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
    </div>
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
                                    type="button" role="tab">
                                <i class="bi {icon}"></i> {name}
                            </button>
                        </li>"""

        tab_content = f"""                        <div class="tab-pane fade {"show active" if active else ""}"
                                          id="{tab_id}" role="tabpanel">
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
            title_html = f'<div class="chart-grid-title">{sub_title}</div>' if sub_title else ""
            item_class = "chart-grid-item full-width" if full_width else "chart-grid-item"
            items_html += f'                                <div class="{item_class}">{title_html}{chart_html}</div>\n'

        tab_button = f"""                        <li class="nav-item">
                            <button class="nav-link {"active" if active else ""}"
                                    data-bs-toggle="tab" data-bs-target="#{tab_id}"
                                    type="button" role="tab">
                                <i class="bi {icon}"></i> {name}
                            </button>
                        </li>"""

        tab_content = f"""                        <div class="tab-pane fade {"show active" if active else ""}"
                                          id="{tab_id}" role="tabpanel">
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
    <title>{self.title}</title>

    <!-- 主题初始化（首屏前执行，避免闪烁）：localStorage 优先，默认深色 -->
    <script>
        (function () {{
            var t = 'dark';
            try {{ t = localStorage.getItem('wbt-theme') || 'dark'; }} catch (e) {{}}
            document.documentElement.setAttribute('data-theme', t);
        }})();
    </script>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">

    <style>
{all_css}
    </style>
</head>
<body>
{header_html}
    <div class="container main-content">
{main_body_html}
    </div>
{footer_html}

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // ---- Plotly 主题同步：图表底色/网格/字体/表格跟随明暗主题 ----
        function wbtPlotlyColors(theme) {{
            return theme === 'dark'
                ? {{ font: '#aab2c5', grid: 'rgba(230,233,239,0.07)', zero: 'rgba(230,233,239,0.16)',
                     line: '#2b3346', cell: '#cfd5e2', bar: 'rgba(170,178,197,0.7)', active: '#5b8cff' }}
                : {{ font: '#46505f', grid: 'rgba(14,17,22,0.07)', zero: 'rgba(14,17,22,0.16)',
                     line: '#cfd6e0', cell: '#1a1f29', bar: 'rgba(70,80,95,0.55)', active: '#2f5fef' }};
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
                try {{
                    (d.data || []).forEach(function (tr, i) {{
                        if (tr.type === 'table') Plotly.restyle(d, {{ 'cells.font.color': c.cell }}, [i]);
                    }});
                }} catch (e) {{}}
            }});
        }}

        function wbtSetTheme(t) {{
            document.documentElement.setAttribute('data-theme', t);
            try {{ localStorage.setItem('wbt-theme', t); }} catch (e) {{}}
            document.querySelectorAll('.theme-switch button').forEach(function (b) {{
                b.classList.toggle('active', b.getAttribute('data-theme') === t);
            }});
            wbtApplyPlotlyTheme(t);
        }}

        document.addEventListener('DOMContentLoaded', function () {{
            var theme = document.documentElement.getAttribute('data-theme') || 'dark';
            document.querySelectorAll('.theme-switch button').forEach(function (b) {{
                b.classList.toggle('active', b.getAttribute('data-theme') === theme);
                b.addEventListener('click', function () {{ wbtSetTheme(b.getAttribute('data-theme')); }});
            }});

            function resizePane(pane) {{
                if (!pane) return;
                pane.querySelectorAll('.plotly-graph-div').forEach(function (d) {{ Plotly.Plots.resize(d); }});
            }}
            document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(function (el) {{
                el.addEventListener('shown.bs.tab', function (ev) {{
                    resizePane(document.querySelector(ev.target.getAttribute('data-bs-target')));
                }});
            }});
            window.addEventListener('resize', function () {{ resizePane(document.querySelector('.tab-pane.active')); }});

            // 初次着色图表以匹配当前主题（plotly 渲染稍晚，window.load 再补一次）
            wbtApplyPlotlyTheme(theme);
            window.addEventListener('load', function () {{ wbtApplyPlotlyTheme(theme); }});
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
