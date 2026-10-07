from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from wbt import WeightBacktest
from wbt.plotting import (
    plot_colored_table,
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_drawdowns_table,
    plot_key_trades,
    plot_monthly_heatmap,
    plot_pairs_hold_dist,
    plot_pairs_pnl_dist,
    plot_rolling_metrics,
    plot_segment_comparison,
    plot_stats_comparison,
    plot_symbol_returns,
    plot_verdict,
    plot_yearly_returns,
)
from wbt.report import HtmlReportBuilder, get_performance_metrics_cards
from wbt.report._plotly_html import replace_plotly_tables


@pytest.mark.parametrize("entry", ["section", "tab", "grid"])
def test_all_public_html_composition_paths_replace_tables(entry):
    figure = go.Figure(go.Table(header={"values": ["字段", "数值"]}, cells={"values": [["日期"], ["2026-01-01"]]}))
    snapshot = figure.to_json()
    fragment = figure.to_html(full_html=False, include_plotlyjs=False)
    builder = HtmlReportBuilder()
    if entry == "section":
        builder.add_section("数据", fragment)
    elif entry == "tab":
        builder.add_chart_tab("数据", fragment, active=True).add_charts_section()
    else:
        builder.add_chart_grid_tab("数据", [("明细", fragment, True)], active=True).add_charts_section()
    markup = builder.render()
    assert 'class="plotly-table-panel"' in markup
    assert 'scope="col"' in markup and 'scope="row"' in markup
    assert "2026-01-01" in markup and "Plotly.newPlot" not in markup
    assert figure.to_json() == snapshot and figure.data[0].type == "table"
    assert builder.render() == markup


def test_boundary_escapes_cells_titles_annotations_and_css():
    payload = '<img src=x onerror="window.leaked=true">'
    figure = go.Figure(go.Table(header={"values": [payload]}, cells={"values": [[payload]]}))
    figure.update_layout(title=payload, annotations=[{"text": f"<b>说明</b><br>{payload}"}])
    fragment = figure.to_html(full_html=False, include_plotlyjs=False)
    markup = replace_plotly_tables(fragment)
    assert "<img" not in markup
    assert "&lt;img" in markup
    assert "<b>说明</b><br>" in markup
    assert "Plotly.newPlot" not in markup
    assert replace_plotly_tables(markup) == markup


@pytest.mark.parametrize("kind", ["curve", "mixed", "callback", "malformed", "already_html"])
def test_unsupported_or_non_table_content_is_byte_identical(kind):
    figure = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    if kind in ("mixed", "callback", "malformed"):
        figure = go.Figure(go.Table(header={"values": ["字段"]}, cells={"values": [["值"]]}))
    if kind == "mixed":
        figure.add_trace(go.Scatter(x=[1], y=[2]))
    fragment = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        post_script="console.log('custom hook')" if kind == "callback" else None,
    )
    if kind == "malformed":
        fragment = fragment.replace('"type":"table"', '"type":not_json')
    if kind == "already_html":
        fragment = '<div class="fin-wrap"><table><tr><th>原生</th><td>值</td></tr></table></div>'
    assert replace_plotly_tables(fragment) == fragment


def test_conversion_retains_cell_background_and_empty_table():
    figure = go.Figure(
        go.Table(
            header={"values": ["字段", "数值"]},
            cells={"values": [["收益"], ["12.34%"]], "fill_color": [["rgba(0,0,0,0)"], ["rgba(231,76,60,0.12)"]]},
        )
    )
    markup = replace_plotly_tables(figure.to_html(full_html=False, include_plotlyjs=False))
    assert 'style="background:rgba(231,76,60,0.12)">12.34%' in markup
    figure.data[0].cells.values = [[], []]
    assert "<tbody></tbody>" in replace_plotly_tables(figure.to_html(full_html=False, include_plotlyjs=False))


@pytest.fixture
def mock_report(monkeypatch):
    rng = np.random.default_rng(905)
    dates = pd.bdate_range("2022-01-03", "2025-12-31")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "dt": dates,
                    "symbol": symbol,
                    "price": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates)))),
                    "weight": np.where((np.arange(len(dates)) + offset) % 20 < 10, 0.8, -0.8),
                }
            )
            for offset, symbol in enumerate(("MOCK-A", "MOCK-B", "MOCK-C"))
        ],
        ignore_index=True,
    )
    result = WeightBacktest(frame, n_jobs=1).to_result()
    snapshot = result.to_json()
    specs = [
        (
            "回测概览",
            [
                ("回撤分析", plot_drawdown, True),
                ("日收益分布", plot_daily_return_dist, True),
                ("月度收益热力图", plot_monthly_heatmap, True),
                ("品种收益分布", plot_symbol_returns, True),
            ],
        ),
        (
            "策略审核",
            [
                ("策略判定（history）", plot_verdict, True),
                ("回撤明细（Top 10）", plot_drawdowns_table, True),
                ("完整绩效指标", plot_colored_table, True),
            ],
        ),
        (
            "稳健性分析",
            [
                ("年度收益（绝对 vs 超额）", plot_yearly_returns, True),
                ("滚动指标（252日）", plot_rolling_metrics, True),
                ("分段对比（近1年 vs 全样本）", plot_segment_comparison, True),
            ],
        ),
        (
            "多空对比",
            [
                (
                    "累计收益（原始）",
                    lambda value, **kwargs: plot_cumulative_returns(
                        value, keys=["多空", "多头", "空头", "基准"], **kwargs
                    ),
                    False,
                ),
                (
                    "波动率归一累计收益",
                    lambda value, **kwargs: plot_cumulative_returns(
                        value, keys=["多空", "多头", "空头", "基准", "多头超额", "空头超额"], voladj=True, **kwargs
                    ),
                    False,
                ),
                ("关键指标对比", plot_stats_comparison, True),
            ],
        ),
        (
            "交易分析",
            [
                ("盈亏比例分布", plot_pairs_pnl_dist, False),
                ("持仓K线数分布", plot_pairs_hold_dist, False),
                ("关键交易", plot_key_trades, True),
            ],
        ),
    ]
    builder = HtmlReportBuilder(title="固定模拟数据 · 原格式表格修复报告")
    builder.add_header(
        {"日期范围": f"{result.start_date} ~ {result.end_date}", "标的数": "3"}, subtitle="仅用于回归验证，数据为模拟"
    )
    builder.add_metrics(get_performance_metrics_cards(result.stats))
    expected, curves = {}, {}
    index = 0
    for tab_name, panels in specs:
        items = []
        for panel_name, make, full_width in panels:
            figure = make(result, title="")
            figure.update_layout(autosize=True)
            fragment = figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"responsive": True, "displayModeBar": "hover", "displaylogo": False, "scrollZoom": True},
            )
            plot_id = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', fragment)[1]
            if figure.data[0].type == "table":
                trace = figure.to_plotly_json()["data"][0]
                expected[plot_id] = trace
            else:
                curves[plot_id] = figure.to_json()
            items.append((panel_name, fragment, full_width))
            index += 1
        builder.add_chart_grid_tab(tab_name, items, active=tab_name == "回测概览")
    builder.add_charts_section().add_footer("固定 seed 905；全部为模拟数据")
    with monkeypatch.context() as baseline:
        baseline.setattr("wbt.report.html_builder.replace_plotly_tables", lambda content: content)
        before = builder.render()
    after = builder.render()
    assert result.to_json() == snapshot
    return before, after, expected, curves


def _inline_dependencies(markup, assets):
    for tag in re.findall(r"<link[^>]*>", markup):
        if "preconnect" in tag:
            markup = markup.replace(tag, "")
        else:
            name = (
                "fonts.css"
                if "fonts.googleapis" in tag
                else ("icons.css" if "bootstrap-icons" in tag else "bootstrap.css")
            )
            markup = markup.replace(tag, "<style>" + (assets / name).read_text() + "</style>")
    return markup.replace(
        '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>',
        "<script>" + (assets / "bootstrap.js").read_text() + "</script>",
    )


def test_mock_report_browser_boundary(mock_report, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    if not os.environ.get("WBT_BROWSER_ASSETS"):
        pytest.skip("Set WBT_BROWSER_ASSETS to locally cached official report dependencies")
    assets = Path(os.environ["WBT_BROWSER_ASSETS"])
    artifacts = Path(os.environ.get("WBT_BROWSER_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    before, after, expected, _curves = mock_report
    for name, content in (("before", before), ("mock-table-fix", after)):
        (artifacts / f"{name}.html").write_text(_inline_dependencies(content, assets))
    records = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("WBT_BROWSER_EXECUTABLE"))
        source, target = browser.new_page(), browser.new_page()
        errors, external = [], []
        for name, page in (("before", source), ("mock-table-fix", target)):
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route(re.compile(r"^https?://"), lambda route: (external.append("blocked"), route.abort()))
            page.goto((artifacts / f"{name}.html").resolve().as_uri(), wait_until="networkidle")
        assert target.locator(".fin-table").count() == 5
        assert target.locator(".plotly-graph-div").count() == 11
        for width in (1440, 390, 320):
            for theme in ("light", "dark"):
                for page in (source, target):
                    page.set_viewport_size({"width": width, "height": 1000})
                    page.locator(f'.theme-switch button[data-theme="{theme}"]').click()
                for tab in range(5):
                    snapshots = []
                    for page in (source, target):
                        page.locator(".nav-link").nth(tab).click()
                        page.wait_for_timeout(400)
                        snapshots.append(
                            page.evaluate("""() => {
                            const box = element => {const rect = element.getBoundingClientRect(); return [rect.x + scrollX, rect.y + scrollY, rect.width, rect.height];};
                            return {header: ['.header-section','.stat-grid','.nav-tabs'].map(selector => box(document.querySelector(selector))),
                                charts: [...document.querySelectorAll('.tab-pane.active .plotly-graph-div')].filter(graph => graph.data.every(trace => trace.type !== 'table'))
                                    .map(graph => ({id: graph.id, box: box(graph), data: JSON.stringify(graph.data), layout: JSON.stringify(graph.layout)}))};
                        }""")
                        )
                    assert snapshots[0] == snapshots[1]
                    for panel in target.locator(".tab-pane.active .plotly-table-panel").all():
                        trace = expected[panel.get_attribute("id")]
                        table = panel.locator("table")
                        assert table.locator("thead th").all_text_contents() == trace["header"]["values"]
                        rows = table.locator("tbody tr").evaluate_all(
                            "rows => rows.map(row => [...row.children].map(cell => cell.textContent))"
                        )
                        assert rows == [list(row) for row in zip(*trace["cells"]["values"], strict=True)]
                        assert table.locator("thead th[scope=col]").count() == len(trace["header"]["values"])
                        assert table.locator("tbody th[scope=row]").count() == len(rows)
                        notes = source.locator(f'[id="{panel.get_attribute("id")}"]').evaluate("""graph =>
                            (graph.layout.annotations || []).map(annotation => {
                                const element = document.createElement('div'); element.innerHTML = annotation.text;
                                return element.textContent;
                            })""")
                        assert panel.locator(".plotly-table-notes").all_text_contents() == notes
                        assert table.evaluate("""table => {
                            const inside = (inner, outer) => inner.left >= outer.left - 1 && inner.right <= outer.right + 1 && inner.top >= outer.top - 1 && inner.bottom <= outer.bottom + 1;
                            const bounds = table.getBoundingClientRect(), cells = [...table.querySelectorAll('th,td')];
                            const probe = document.createElement('span'); document.body.append(probe);
                            probe.style.color = getComputedStyle(document.documentElement).getPropertyValue('--border-strong');
                            const border = getComputedStyle(probe).color; probe.remove();
                            return cells.every(cell => {
                                const range = document.createRange(); range.selectNodeContents(cell);
                                const style = getComputedStyle(cell);
                                return inside(range.getBoundingClientRect(), cell.getBoundingClientRect()) && inside(cell.getBoundingClientRect(), bounds) &&
                                    style.borderBottomColor === border && parseFloat(style.borderBottomWidth) > 0;
                            }) && inside(table.querySelector('tbody tr:last-child').getBoundingClientRect(), bounds);
                        }""")
                        assert panel.locator(".plotly-table-notes").evaluate_all("""notes => notes.every(note => {
                            const range = document.createRange(); range.selectNodeContents(note);
                            const text = range.getBoundingClientRect(), box = note.getBoundingClientRect(), table = note.parentElement.querySelector('table').getBoundingClientRect();
                            return text.left >= box.left - 1 && text.right <= box.right + 1 && text.bottom <= box.bottom + 1 && text.bottom < table.top;
                        })""")
                        wrapper = table.locator("..")
                        wrapper.focus()
                        if wrapper.evaluate("node => node.scrollWidth > node.clientWidth"):
                            wrapper.evaluate("node => node.scrollLeft = 0")
                            wrapper.press("ArrowRight")
                            target.wait_for_timeout(150)
                            assert wrapper.evaluate("node => node.scrollLeft > 0")
                        wrapper.evaluate("node => node.scrollLeft = node.scrollWidth")
                        assert table.locator("tbody tr:last-child > :last-child").evaluate("""cell => {
                            const box = cell.getBoundingClientRect(), region = cell.closest('.fin-wrap').getBoundingClientRect();
                            const range = document.createRange(); range.selectNodeContents(cell);
                            const text = range.getBoundingClientRect();
                            let ancestor = cell.parentElement;
                            while (ancestor && ancestor !== document.body) {
                                const style = getComputedStyle(ancestor), bounds = ancestor.getBoundingClientRect();
                                if (['hidden','clip','auto','scroll'].includes(style.overflowY) && box.bottom > bounds.bottom + 1) return false;
                                ancestor = ancestor.parentElement;
                            }
                            return box.right <= region.right + 1 && box.bottom <= region.bottom + 1 && text.bottom <= region.bottom + 1;
                        }""")
                        records.append(
                            {
                                "width": width,
                                "theme": theme,
                                "rows": len(rows),
                                "columns": len(trace["header"]["values"]),
                            }
                        )
                    target.screenshot(path=str(artifacts / f"mock-{width}-{theme}-{tab}.png"), full_page=True)
        assert len(records) == 30 and errors == [] and external == []
        browser.close()
    (artifacts / "browser-results.json").write_text(
        json.dumps(
            {
                "checked_tables": records,
                "non_table_layout_unchanged": True,
                "errors": errors,
                "external_requests": external,
            },
            indent=2,
        )
    )
