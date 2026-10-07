from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest
from wbt.plotting import (
    plot_colored_table,
    plot_drawdowns_table,
    plot_segment_comparison,
    plot_stats_comparison,
    plot_verdict,
)
from wbt.report._generator import _prepare_config, _render_backtest_report
from wbt.report.html_builder import HtmlReportBuilder


@pytest.fixture(scope="module")
def table_result():
    rng = np.random.default_rng(905)
    dates = pd.bdate_range("2022-01-03", "2025-12-31")
    frame = pd.DataFrame(
        {
            "dt": dates,
            "symbol": "MOCK",
            "price": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates)))),
            "weight": np.where(np.arange(len(dates)) % 20 < 10, 0.8, -0.8),
        }
    )
    return WeightBacktest(frame, n_jobs=1).to_result()


TABLE_PLOTS = [plot_stats_comparison, plot_segment_comparison, plot_drawdowns_table, plot_colored_table, plot_verdict]


def _assert_rendered_plotly_table(graph, rows, columns, theme):
    result = graph.evaluate(
        """(graph, expected) => {
            const failures = [];
            const bounds = graph.querySelector('.main-svg').getBoundingClientRect();
            const inside = (inner, outer) => inner.left >= outer.left - 1 &&
                inner.right <= outer.right + 1 && inner.top >= outer.top - 1 &&
                inner.bottom <= outer.bottom + 1;
            const cells = [...graph.querySelectorAll('.column-cell')].filter(
                cell => cell.querySelector('.cell-text'));
            const headers = cells.filter(cell => cell.closest('.column-block').id === 'header');
            if (cells.length !== expected.columns * (expected.rows + 1)) failures.push('missing cells');
            if (headers.length !== expected.columns) failures.push('missing headers');
            const border = expected.theme === 'dark' ? 'rgb(43, 51, 70)' : 'rgb(207, 214, 224)';
            for (const cell of cells) {
                const text = cell.querySelector('.cell-text');
                const rect = cell.querySelector('.cell-rect');
                if (!inside(text.getBoundingClientRect(), rect.getBoundingClientRect())) {
                    failures.push('clipped cell: ' + text.textContent);
                }
                if (!inside(rect.getBoundingClientRect(), bounds)) failures.push('cell outside SVG');
                const style = getComputedStyle(rect);
                if (style.stroke !== border || parseFloat(style.strokeWidth) <= 0 ||
                    parseFloat(style.strokeOpacity) <= 0) failures.push('wrong rendered border');
            }
            for (const column of graph.querySelectorAll('.y-column')) {
                const body = [...column.querySelectorAll('.column-block:not([id="header"]) .column-cell')]
                    .filter(cell => cell.querySelector('.cell-text'));
                if (body.length !== expected.rows) failures.push('missing last row');
                const last = body.at(-1).querySelector('.cell-rect').getBoundingClientRect();
                const clip = graph.querySelector('.scroll-background').getBoundingClientRect();
                if (!inside(last, clip)) failures.push('last row clipped by scroll area');
            }
            for (const annotation of graph.querySelectorAll('.annotation-text')) {
                const annotationBounds = annotation.getBoundingClientRect();
                if (!inside(annotationBounds, bounds)) failures.push('clipped annotation');
                if (headers.length && annotationBounds.bottom > Math.min(
                    ...headers.map(header => header.getBoundingClientRect().top)) - 1) {
                    failures.push('annotation overlaps table');
                }
            }
            return failures;
        }""",
        {"rows": rows, "columns": columns, "theme": theme},
    )
    assert result == []


def _assert_rendered_native_table(table):
    failures = table.evaluate(
        """table => {
            const failures = [];
            const bounds = table.getBoundingClientRect();
            const inside = (inner, outer) => inner.left >= outer.left - 1 &&
                inner.right <= outer.right + 1 && inner.top >= outer.top - 1 &&
                inner.bottom <= outer.bottom + 1;
            const rootStyle = getComputedStyle(document.documentElement);
            const probe = document.createElement('span');
            document.body.append(probe);
            const borderColor = variable => {
                probe.style.color = rootStyle.getPropertyValue(variable);
                return getComputedStyle(probe).color;
            };
            const headerBorder = borderColor('--border-strong');
            const bodyBorder = borderColor('--border');
            const columns = table.querySelectorAll('thead th').length;
            for (const row of table.querySelectorAll('tr')) {
                if (row.children.length !== columns) failures.push('missing columns');
                for (const cell of row.children) {
                    const range = document.createRange();
                    range.selectNodeContents(cell);
                    if (!inside(range.getBoundingClientRect(), cell.getBoundingClientRect())) {
                        failures.push('clipped native text: ' + cell.textContent);
                    }
                    if (!inside(cell.getBoundingClientRect(), bounds)) failures.push('cell outside table');
                    const style = getComputedStyle(cell);
                    if (style.borderBottomWidth !== '0px' && style.borderBottomColor !==
                        (cell.tagName === 'TH' ? headerBorder : bodyBorder)) failures.push('wrong native border');
                }
            }
            if (!inside(table.querySelector('tbody tr:last-child').getBoundingClientRect(), bounds)) {
                failures.push('last native row clipped');
            }
            probe.remove();
            return failures;
        }"""
    )
    assert failures == []


@pytest.mark.parametrize("plot", TABLE_PLOTS)
def test_plotly_tables_have_matching_row_and_canvas_heights(table_result, plot):
    figure = plot(table_result, title="")
    table = figure.data[0]
    rows = max(len(column) for column in table.cells.values)
    assert table.cells.height == 30
    assert table.header.height == 32
    assert table.cells.font.size >= 13
    assert figure.layout.height == figure.layout.margin.t + 32 + rows * 30 + figure.layout.margin.b


def test_report_tables_in_browser(table_result, tmp_path: Path):
    playwright = pytest.importorskip("playwright.sync_api")
    artifacts = Path(os.environ.get("WBT_BROWSER_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    report = artifacts / "native-report.html"
    _render_backtest_report(table_result, _prepare_config({}), str(report), "Mock 回测报告")
    builder = HtmlReportBuilder(title="Plotly 表格回归验证")
    for index, plot in enumerate(TABLE_PLOTS):
        figure = plot(table_result, title="")
        builder.add_section(plot.__name__, figure.to_html(full_html=False, include_plotlyjs=index == 0))
    legacy = artifacts / "plotly-report.html"
    builder.save(str(legacy))
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("WBT_BROWSER_EXECUTABLE"))
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for path in (report, legacy):
            page.goto(path.resolve().as_uri(), wait_until="networkidle")
            for width, theme in ((width, theme) for width in (1440, 390, 320) for theme in ("dark", "light")):
                page.set_viewport_size({"width": width, "height": 1000})
                page.evaluate("theme => wbtSetTheme(theme)", theme)
                page.wait_for_timeout(350)
                if path == report:
                    for label in ("策略审核", "稳健性分析", "多空对比"):
                        page.locator(".nav-link").filter(has_text=label).click()
                        page.wait_for_timeout(350)
                        tables = page.locator(".tab-pane.active .fin-table")
                        assert tables.count() >= 1
                        for table_index in range(tables.count()):
                            table = tables.nth(table_index)
                            assert table.is_visible()
                            assert table.locator("tbody tr").count() > 0
                            assert (
                                table.locator("td").first.evaluate("cell => cell.getBoundingClientRect().height") >= 25
                            )
                            _assert_rendered_native_table(table)
                            wrapper = table.locator("..")
                            wrapper.evaluate("wrapper => wrapper.scrollLeft = wrapper.scrollWidth")
                            assert table.locator("tbody tr:last-child td:last-child").evaluate(
                                "cell => cell.getBoundingClientRect().right <= cell.closest('.fin-wrap').getBoundingClientRect().right + 1"
                            )
                            wrapper.evaluate("wrapper => wrapper.scrollLeft = 0")
                        assert page.locator(".tab-pane.active").get_by_text("生成失败", exact=False).count() == 0
                        if label == "策略审核":
                            assert page.locator(".tab-pane.active .kv-grid .kv").count() >= len(table_result.stats)
                            assert page.locator(".tab-pane.active .t-up").count() > 0
                            assert page.locator(".tab-pane.active .t-down").count() > 0
                            assert page.locator(".tab-pane.active .kv:visible").evaluate_all(
                                """items => items.every(item => {
                                    const key = item.querySelector('.kv-k').getBoundingClientRect();
                                    const value = item.querySelector('.kv-v').getBoundingClientRect();
                                    const bounds = item.getBoundingClientRect();
                                    return key.left >= bounds.left && value.right <= bounds.right &&
                                        key.right <= value.left;
                                })"""
                            )
                        page.screenshot(path=str(artifacts / f"native-{label}-{width}-{theme}.png"), full_page=True)
                else:
                    page.wait_for_function(
                        """() => [...document.querySelectorAll('.plotly-graph-div')].every(
                            graph => graph._fullData[0].cells.height === 30 &&
                            graph._fullData[0].cells.line.color === wbtPlotlyColors(
                                document.documentElement.dataset.theme).line)"""
                    )
                    assert page.locator(".plotly-graph-div").count() == len(TABLE_PLOTS)
                    for index, plot in enumerate(TABLE_PLOTS):
                        graph = page.locator(".plotly-graph-div").nth(index)
                        figure = plot(table_result, title="")
                        columns = len(figure.data[0].header.values)
                        rows = max(len(column) for column in figure.data[0].cells.values)
                        _assert_rendered_plotly_table(graph, rows, columns, theme)
                        if width < 1000:
                            graph.scroll_into_view_if_needed()
                            scroll = graph.locator("..")
                            assert scroll.evaluate("wrapper => wrapper.scrollWidth > wrapper.clientWidth")
                            scroll.evaluate("wrapper => wrapper.scrollLeft = 0")
                            assert graph.locator(".y-column").first.evaluate(
                                "column => column.getBoundingClientRect().left >= column.closest('.plotly-graph-div').parentElement.getBoundingClientRect().left"
                            )
                            scroll.screenshot(path=str(artifacts / f"{plot.__name__}-{width}-{theme}-left.png"))
                            scroll.evaluate("wrapper => wrapper.scrollLeft = wrapper.scrollWidth")
                            assert graph.locator(".y-column").last.evaluate(
                                "column => column.getBoundingClientRect().right <= column.closest('.plotly-graph-div').parentElement.getBoundingClientRect().right + 1"
                            )
                            _assert_rendered_plotly_table(graph, rows, columns, theme)
                            scroll.screenshot(path=str(artifacts / f"{plot.__name__}-{width}-{theme}-right.png"))
                            scroll.evaluate("wrapper => wrapper.scrollLeft = 0")
                    page.screenshot(path=str(artifacts / f"plotly-{width}-{theme}.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.set_viewport_size({"width": 390, "height": 844})
            if path == report:
                page.locator(".nav-link").filter(has_text="策略审核").click()
                page.wait_for_timeout(350)
                assert page.locator(".tab-pane.active .fin-table").first.is_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(artifacts / "native-mobile.png"), full_page=True)
            page.set_viewport_size({"width": 1440, "height": 1000})
        assert errors == []
        browser.close()


def test_plotly_table_in_hidden_tab(table_result, tmp_path: Path):
    playwright = pytest.importorskip("playwright.sync_api")
    figure = plot_stats_comparison(table_result, title="")
    report = tmp_path / "hidden-table.html"
    builder = HtmlReportBuilder(title="隐藏标签页表格")
    builder.add_chart_tab("初始页", "<p>初始页</p>", active=True)
    builder.add_chart_tab("表格页", figure.to_html(full_html=False, include_plotlyjs=True))
    builder.save(str(report))
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("WBT_BROWSER_EXECUTABLE"))
        page = browser.new_page(viewport={"width": 390, "height": 844})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("wbtSetTheme('light')")
        page.locator(".nav-link").filter(has_text="表格页").click()
        page.wait_for_timeout(350)
        rows = max(len(column) for column in figure.data[0].cells.values)
        _assert_rendered_plotly_table(
            page.locator(".plotly-graph-div"), rows, len(figure.data[0].header.values), "light"
        )
        assert errors == []
        browser.close()
