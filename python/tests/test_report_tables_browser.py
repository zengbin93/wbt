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
            for theme in ("dark", "light"):
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
                        assert page.locator(".tab-pane.active").get_by_text("生成失败", exact=False).count() == 0
                        if label == "策略审核":
                            assert page.locator(".tab-pane.active .kv-grid .kv").count() >= len(table_result.stats)
                            assert page.locator(".tab-pane.active .t-up").count() > 0
                            assert page.locator(".tab-pane.active .t-down").count() > 0
                        page.screenshot(path=str(artifacts / f"native-{label}-{theme}.png"), full_page=True)
                else:
                    page.wait_for_function(
                        """() => [...document.querySelectorAll('.plotly-graph-div')].every(
                            graph => graph._fullData[0].cells.height === 30 &&
                            graph._fullData[0].cells.line.color === wbtPlotlyColors(
                                document.documentElement.dataset.theme).line)"""
                    )
                    assert page.locator(".plotly-graph-div").count() == len(TABLE_PLOTS)
                    page.screenshot(path=str(artifacts / f"plotly-{theme}.png"), full_page=True)
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
