from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wbt import WeightBacktest
from wbt.metrics import COMPARE_METRICS, COMPARE_SIDES, DETAIL_LABELS, lookup_metric
from wbt.plotting._common import fmt_value
from wbt.report._generator import _prepare_config, _render_backtest_report
from wbt.report._html_tables import drawdowns_table_html, history_verdict_card_html


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


def _expected_tables(result):
    history, recent = result.verdict, result.verdict_recent
    years = sorted(history["yearly_metrics"], key=lambda year: year["year"])
    tables = {
        "年度审核明细": (
            ["年份", "绝对收益", "超额收益", "超额回撤", "交易日数", "完整年度", "达标"],
            [
                [
                    str(int(year["year"])),
                    fmt_value("绝对收益", year.get("abs_return")),
                    fmt_value("超额收益", year.get("alpha_return")),
                    fmt_value("超额回撤", year.get("alpha_max_drawdown")),
                    fmt_value("交易日数", year.get("days")),
                    fmt_value("完整年度", year.get("is_complete_year")),
                    "达标" if year.get("year_passed") else "未达标",
                ]
                for year in years
            ],
        ),
        "近期窗口指标": (
            ["近期指标", "数值"],
            [
                [label, fmt_value(label, recent.get(key))]
                for label, key in (
                    ("近期绝对收益", "recent_abs_return"),
                    ("近期超额收益", "recent_alpha_return"),
                    ("近期超额回撤", "recent_alpha_max_drawdown"),
                    ("历史超额回撤(剔除近期)", "history_alpha_max_drawdown_excl_recent"),
                )
            ],
        ),
        "完整绩效指标 · 含样本起止日期": (
            ["指标", "数值"],
            [[key, fmt_value(key, value)] for key, value in result.stats.items()],
        ),
    }
    for mode, verdict in (("history", history), ("recent", recent)):
        keys = [key for key in DETAIL_LABELS if key in verdict]
        keys.extend(key for key in verdict if key not in DETAIL_LABELS and key != "yearly_metrics")
        assert len(keys) == len(verdict) - ("yearly_metrics" in verdict)
        tables[f"{mode} · 完整判定字段"] = (
            ["判定字段", "结果"],
            [[DETAIL_LABELS.get(key, key), fmt_value(key, verdict[key])] for key in keys],
        )
    headers = list(result.drawdowns[0])
    tables["最大回撤明细 · Top 10"] = (
        headers,
        [[fmt_value(key, row.get(key)) for key in headers] for row in result.drawdowns],
    )
    for caption, stats, order in (
        ("多空与基准 · 关键指标对比", {"多空": result.stats, **result.stats_by_side}, COMPARE_SIDES),
        ("全样本与近一年 · 分段比较", result.segment_comparison, ("全样本", "近1年")),
    ):
        sides = [side for side in order if side in stats]
        tables[caption] = (
            ["指标", *sides],
            [
                [metric, *[fmt_value(metric, lookup_metric(stats[side], metric)) for side in sides]]
                for metric in COMPARE_METRICS
            ],
        )
    return tables


def _assert_native_table(table, expected):
    headers, rows = expected
    assert table.locator("thead th").all_text_contents() == headers
    assert table.locator("thead th[scope=col]").count() == len(headers)
    assert table.locator("tbody tr").count() == len(rows)
    assert table.locator("tbody th[scope=row]").count() == len(rows)
    actual = table.locator("tbody tr").evaluate_all(
        "rows => rows.map(row => [...row.children].map(cell => cell.textContent))"
    )
    assert actual == rows
    failures = table.evaluate(
        """table => {
            const failures = [];
            const bounds = table.getBoundingClientRect();
            const inside = (inner, outer) => inner.left >= outer.left - 1 &&
                inner.right <= outer.right + 1 && inner.top >= outer.top - 1 && inner.bottom <= outer.bottom + 1;
            const probe = document.createElement('span');
            document.body.append(probe);
            const root = getComputedStyle(document.documentElement);
            const color = variable => {
                probe.style.color = root.getPropertyValue(variable);
                return getComputedStyle(probe).color;
            };
            const headerBorder = color('--border-strong'), bodyBorder = color('--border');
            for (const cell of table.querySelectorAll('th, td')) {
                const range = document.createRange();
                range.selectNodeContents(cell);
                if (!inside(range.getBoundingClientRect(), cell.getBoundingClientRect())) {
                    failures.push('clipped text: ' + cell.textContent);
                }
                if (!inside(cell.getBoundingClientRect(), bounds)) failures.push('cell outside table');
                const style = getComputedStyle(cell);
                if (style.borderBottomWidth !== '0px' && style.borderBottomColor !==
                    (cell.closest('thead') ? headerBorder : bodyBorder)) failures.push('wrong rendered border');
            }
            if (!inside(table.querySelector('tbody tr:last-child').getBoundingClientRect(), bounds)) failures.push('last row clipped');
            probe.remove();
            return failures;
        }"""
    )
    assert failures == []
    wrapper = table.locator("..")
    assert wrapper.get_attribute("tabindex") == "0"
    wrapper.focus()
    if wrapper.evaluate("wrapper => wrapper.scrollWidth > wrapper.clientWidth"):
        wrapper.evaluate("wrapper => wrapper.scrollLeft = 0")
        wrapper.press("ArrowRight")
        wrapper.page.wait_for_timeout(150)
        assert wrapper.evaluate("wrapper => wrapper.scrollLeft") > 0
        assert "左右滚动" in table.locator("caption").evaluate(
            "caption => getComputedStyle(caption, '::after').content"
        )
    wrapper.evaluate("wrapper => wrapper.scrollLeft = wrapper.scrollWidth")
    assert table.locator("tbody tr:last-child > :last-child").evaluate(
        "cell => cell.getBoundingClientRect().right <= cell.closest('.fin-wrap').getBoundingClientRect().right + 1"
    )
    wrapper.evaluate("wrapper => wrapper.scrollLeft = 0")


def test_native_tables_escape_content_and_handle_empty_drawdowns():
    assert "暂无回撤记录" in drawdowns_table_html(SimpleNamespace(drawdowns=[]))
    markup = history_verdict_card_html({"reason": '<script>alert("x")</script>', "yearly_metrics": []})
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def test_redesigned_report_in_browser(table_result, tmp_path: Path):
    playwright = pytest.importorskip("playwright.sync_api")
    artifacts = Path(os.environ.get("WBT_BROWSER_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    report = artifacts / "redesigned-report.html"
    snapshot = table_result.to_json()
    _render_backtest_report(table_result, _prepare_config({}), str(report), "多空轮动策略 · 回测报告")
    assert table_result.to_json() == snapshot
    expected = _expected_tables(table_result)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("WBT_BROWSER_EXECUTABLE"))
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.route("https://**", lambda route: route.abort())
        page.goto(report.resolve().as_uri(), wait_until="networkidle")
        assert page.locator("h1").count() == 1
        assert page.locator("main").count() == 1
        assert page.locator(".stat-tile").count() == 14
        assert page.evaluate("""() => [...document.querySelectorAll('.plotly-graph-div')].every(
            graph => graph.data.every(trace => trace.type !== 'table'))""")
        assert page.locator(".plotly-graph-div .table").count() == 0
        assert page.locator(".fin-table").count() == len(expected)
        tabs = page.get_by_role("tab")
        tabs.first.focus()
        tabs.first.press("ArrowRight")
        assert tabs.nth(1).get_attribute("aria-selected") == "true"
        tabs.nth(1).press("End")
        assert tabs.last.get_attribute("aria-selected") == "true"
        tabs.last.press("Home")
        assert tabs.first.get_attribute("aria-selected") == "true"
        tabs.first.press("ArrowLeft")
        assert tabs.last.get_attribute("aria-selected") == "true"
        tabs.last.press("Home")
        assert tabs.first.evaluate("tab => getComputedStyle(tab).outlineWidth") == "3px"
        assert tabs.evaluate_all("tabs => tabs.filter(tab => tab.tabIndex === 0).length") == 1
        for width in (1440, 390, 320):
            page.set_viewport_size({"width": width, "height": 1000})
            for theme in ("light", "dark"):
                checked = set()
                toggle = page.locator(f'.theme-switch button[data-theme="{theme}"]')
                toggle.focus()
                toggle.press("Enter")
                assert toggle.get_attribute("aria-pressed") == "true"
                assert page.locator("html").get_attribute("data-theme") == theme
                for label in ("回测概览", "策略审核", "稳健性分析", "多空对比", "交易分析"):
                    tab = tabs.filter(has_text=label)
                    tab.click()
                    page.wait_for_timeout(120)
                    assert tab.get_attribute("aria-selected") == "true"
                    assert page.get_by_role("tabpanel").count() == 1
                    assert page.locator(".tab-pane.active").get_by_text("生成失败", exact=False).count() == 0
                    if label == "策略审核":
                        summary = page.locator(".verdict-details summary")
                        summary.focus()
                        assert page.locator(".verdict-details").get_attribute("open") is None
                        summary.press("Enter")
                        assert page.locator(".verdict-details").get_attribute("open") == ""
                        reasons = page.locator(".verdict-reason")
                        assert reasons.all_text_contents() == [
                            str(table_result.verdict.get("reason") or ""),
                            str(table_result.verdict_recent.get("reason") or ""),
                        ]
                        assert reasons.evaluate_all("""items => items.every(item => {
                            const range = document.createRange(); range.selectNodeContents(item);
                            const text = range.getBoundingClientRect(), bounds = item.getBoundingClientRect();
                            return text.left >= bounds.left - 1 && text.right <= bounds.right + 1 &&
                                text.bottom <= bounds.bottom + 1;
                        })""")
                    if label == "多空对比":
                        graph = (
                            page.locator(".chart-grid-item")
                            .filter(has_text="累计收益（原始）")
                            .locator(".plotly-graph-div")
                        )
                        actual_curve = graph.evaluate("graph => Array.from(graph._fullData[0].y)")
                        np.testing.assert_allclose(actual_curve, table_result.curves["多空"].cum)
                        assert graph.evaluate("graph => graph._fullData[0].x.length") == len(table_result.dates)
                    for table in page.locator(".tab-pane.active .fin-table:visible").all():
                        caption = table.locator("caption").inner_text()
                        _assert_native_table(table, expected[caption])
                        checked.add(caption)
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    if label == "策略审核":
                        page.locator(".verdict-details summary").click()
                        assert page.locator(".verdict-details").get_attribute("open") is None
                    page.evaluate("document.activeElement.blur(); window.scrollTo(0, 0)")
                    page.mouse.move(0, 0)
                    page.screenshot(path=str(artifacts / f"{label}-{width}-{theme}.png"), full_page=True)
                page.locator('.theme-switch button[data-theme="dark"]').click()
                assert checked == set(expected)
        page.reload(wait_until="networkidle")
        assert page.locator("html").get_attribute("data-theme") == "dark"
        page.keyboard.press("Tab")
        assert page.get_by_role("link", name="跳至报告正文").evaluate("link => link === document.activeElement")
        page.keyboard.press("Enter")
        assert page.locator("main").evaluate("main => main === document.activeElement")
        assert errors == []
        assert not any(url.startswith("http") for url in requests)
        browser.close()
