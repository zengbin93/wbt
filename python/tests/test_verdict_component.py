import html
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import plotly.graph_objects as go
import pytest
import test_report_plotly_boundary as mock_tests

from wbt.report._plotly_html import _verdict_notes
from wbt.report.html_builder import HtmlReportBuilder

SAMPLE = (
    "<b>history（逐年）：❌ 不可用</b><br>原历史理由<br>原第二行<br>"
    "<b>recent（近期窗口）：✅ 可用</b><br>窗口开始：2025-01-14<br>"
    "近期绝对收益：-20.40%<br>历史窗口不足：否<br>原近期理由"
)

THRESHOLD_CASES = [
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000", True),
    ("(abs_return=-0.2, alpha_max_drawdown=0.397700 ≥ threshold 0.200000); other reason", True),
    ("other_alpha_max_drawdown=0.397700 ≥ threshold 0.200000", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000e-2", False),
    ("alpha_max_drawdown=0.397700e0 ≥ threshold 0.200000", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000abc", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000.5", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000_2", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000/2", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000,5", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000; alpha_max_drawdown = 0.397700 ≥ threshold 0.300000", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000; alpha_max_drawdown=0.397700 ≥ threshold 0.300000", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000; alpha_max_drawdown=0.397700 ≥ threshold 0.300000e-2", False),
    ("alpha_max_drawdown=0.397700 ≥ threshold 0.200000; alpha_max_drawdown=0.397700 ≥ threshold 0.200000", False),
]


def _threshold_text(reason):
    return SAMPLE.replace("原近期理由", reason).replace("历史窗口不足：否", "近期超额回撤：39.77%<br>历史窗口不足：否")


@pytest.mark.parametrize("reason,supported", THRESHOLD_CASES)
def test_threshold_tokens_and_original_details(reason, supported):
    text = _threshold_text(reason)
    rendered = _verdict_notes(text)
    assert ('class="metric-threshold"' in rendered) is supported
    assert ("原文阈值" in rendered) is supported
    assert f'<div class="review-original">{text}</div>' in rendered


def test_threshold_tokens_in_browser(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    if not os.environ.get("WBT_BROWSER_ASSETS"):
        pytest.skip("Set WBT_BROWSER_ASSETS to locally cached report dependencies")
    builder = HtmlReportBuilder("阈值识别回归（模拟数据）")
    builder.add_header({}, subtitle="固定模拟数据；严格阈值回归")
    for index, (reason, _) in enumerate(THRESHOLD_CASES):
        figure = go.Figure(go.Table(header={"values": ["字段"]}, cells={"values": [["模拟值"]]}))
        figure.update_layout(annotations=[{"text": _threshold_text(reason)}])
        builder.add_section(str(index), figure.to_html(full_html=False, include_plotlyjs=False))
    path = tmp_path / "threshold-cases.html"
    path.write_text(mock_tests._inline_dependencies(builder.render(), Path(os.environ["WBT_BROWSER_ASSETS"])))
    errors, external, records = [], [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("WBT_BROWSER_EXECUTABLE"))
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(re.compile(r"^https?://"), lambda route: (external.append("blocked"), route.abort()))
        page.goto(path.resolve().as_uri(), wait_until="networkidle")
        for width in (1440, 390, 320):
            page.set_viewport_size({"width": width, "height": 1000})
            for theme in ("light", "dark"):
                page.locator(f'.theme-switch button[data-theme="{theme}"]').click()
                for index, (reason, supported) in enumerate(THRESHOLD_CASES):
                    component = page.locator(".verdict-review").nth(index)
                    assert component.locator(".metric-threshold").count() == int(supported)
                    assert ("原文阈值" in component.inner_text()) is supported
                    if supported:
                        marker = component.locator(".metric-threshold")
                        assert marker.get_attribute("data-threshold") == "0.2"
                        assert marker.get_attribute("d") == "M40 0V20"
                        assert marker.evaluate("""node => {
                            const svg = node.ownerSVGElement, box = svg.getBoundingClientRect();
                            const matrix = node.getScreenCTM();
                            const start = new DOMPoint(40, 0).matrixTransform(matrix);
                            const end = new DOMPoint(40, 20).matrixTransform(matrix);
                            const style = getComputedStyle(node);
                            return svg.checkVisibility() && style.stroke !== 'none' && parseFloat(style.strokeWidth) > 0 &&
                                start.x >= box.left && end.x <= box.right && start.y >= box.top - 1 && end.y <= box.bottom + 1;
                        }""")
                    summary = component.locator("summary")
                    summary.focus()
                    summary.press("Enter")
                    original = component.locator(".review-original")
                    assert original.is_visible()
                    assert original.text_content() == html.unescape(re.sub(r"<[^>]*>", "", _threshold_text(reason)))
                    summary.press("Space")
                    assert not original.is_visible()
                    records.append({"width": width, "theme": theme, "case": index, "threshold_visible": supported})
        assert errors == [] and external == []
        browser.close()
    artifacts = Path(os.environ.get("WBT_BROWSER_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "threshold-validation.json").write_text(json.dumps(records, indent=2))


@pytest.mark.parametrize(
    "text", [SAMPLE, SAMPLE.replace("-20.40%", "—"), SAMPLE.replace("原历史理由", "&lt;原理由&gt;")]
)
def test_component_preserves_all_original_text(text):
    rendered = _verdict_notes(text)
    assert rendered.count('<details class="review-section">') == 1
    original = re.search(r'<div class="review-original">(.*?)</div>', rendered, re.DOTALL)
    assert original is not None
    assert html.unescape(re.sub(r"<[^>]*>", "", original[1])) == html.unescape(re.sub(r"<[^>]*>", "", text))


@pytest.mark.parametrize("text", ["普通说明", "<b>history</b><br>其他结构", "<b>recent（近期窗口）：可用</b>"])
def test_unknown_notes_keep_original_paragraph(text):
    assert _verdict_notes(text).startswith('<p class="plotly-table-notes">')


def test_component_escapes_untrusted_content():
    rendered = _verdict_notes(SAMPLE.replace("原历史理由", '<img src=x onerror="alert(1)">'))
    assert "<img" not in rendered and "&lt;img" in rendered


def test_graph_coordinates_use_original_percentages():
    rendered = _verdict_notes(SAMPLE)
    assert 'cx="79.6"' in rendered
    assert "刻度 −100% · 0 · +100%" in rendered
    for value in ("—", "120%", "NaN%"):
        assert 'class="review-metric"' not in _verdict_notes(SAMPLE.replace("-20.40%", value))
    assert _verdict_notes(SAMPLE.replace("✅ 可用", "未知")).startswith('<p class="plotly-table-notes">')


def test_threshold_requires_explicit_matching_original_metric():
    text = SAMPLE.replace("原近期理由", "alpha_max_drawdown=0.397706 ≥ threshold 0.200000")
    text = text.replace("历史窗口不足：否", "近期超额回撤：39.77%<br>历史窗口不足：否")
    assert 'data-threshold="0.2" d="M40 0V20"' in _verdict_notes(text)
    assert "data-threshold=" not in _verdict_notes(text.replace("39.77%", "40.00%"))
    assert "data-threshold=" not in _verdict_notes(text.replace("0.200000", "1.200000"))


def _load_baseline(root, name):
    spec = importlib.util.spec_from_file_location(f"wbt.report._component_baseline_{name}", root / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_browser_against_main(tmp_path, monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")
    if not os.environ.get("WBT_BROWSER_ASSETS") or not os.environ.get("WBT_BASELINE_SOURCE"):
        pytest.skip("Set WBT_BROWSER_ASSETS and WBT_BASELINE_SOURCE to a clean ff38e79 checkout")
    baseline_source = Path(os.environ["WBT_BASELINE_SOURCE"])
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline_source, text=True).strip() == (
        "ff38e79f86bc4ab6ca2a4f7d80ce2c3ef516e39a"
    )
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=baseline_source, text=True) == ""
    root = baseline_source / "python/wbt/report"
    baseline_builder = _load_baseline(root, "html_builder")
    baseline_tables = _load_baseline(root, "_plotly_html")
    baseline_builder.replace_plotly_tables = baseline_tables.replace_plotly_tables
    with monkeypatch.context() as patch:
        patch.setattr(mock_tests, "HtmlReportBuilder", baseline_builder.HtmlReportBuilder)
        _, before, _, _ = mock_tests.mock_report.__wrapped__(patch)
    _, after, _, _ = mock_tests.mock_report.__wrapped__(monkeypatch)
    artifacts = Path(os.environ.get("WBT_BROWSER_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    assets = Path(os.environ["WBT_BROWSER_ASSETS"])
    for name, content in (("component-before", before), ("verdict-component", after)):
        (artifacts / f"{name}.html").write_text(mock_tests._inline_dependencies(content, assets))
    errors, external, records = [], [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("WBT_BROWSER_EXECUTABLE"))
        source, target = browser.new_page(), browser.new_page()
        for name, page in (("component-before", source), ("verdict-component", target)):
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route(re.compile(r"^https?://"), lambda route: (external.append("blocked"), route.abort()))
            page.goto((artifacts / f"{name}.html").resolve().as_uri(), wait_until="networkidle")
        for width in (1440, 390, 320):
            for theme in ("light", "dark"):
                for page in (source, target):
                    page.set_viewport_size({"width": width, "height": 1000})
                    page.locator(f'.theme-switch button[data-theme="{theme}"]').click()
                for tab in range(5):
                    snapshots = []
                    for page in (source, target):
                        page.locator(".nav-link").nth(tab).click()
                        page.wait_for_timeout(300)
                        snapshots.append(
                            page.evaluate("""() => {
                            const box = node => {const rect = node.getBoundingClientRect(); return [rect.x + scrollX, rect.y + scrollY, rect.width, rect.height];};
                            const style = node => {const css = getComputedStyle(node);return [css.fontSize, css.fontFamily, css.color, css.backgroundColor, css.padding, css.borderColor];};
                            return {
                                chrome: ['.header-section','.stat-grid','.nav-tabs'].map(selector => {const node = document.querySelector(selector);return {box:box(node),text:node.textContent,style:style(node)};}),
                                charts: [...document.querySelectorAll('.tab-pane.active .plotly-graph-div')].map(node => ({box:box(node),data:JSON.stringify(node.data),layout:JSON.stringify(node.layout)})),
                                tables: [...document.querySelectorAll('.tab-pane.active table')].map(node => ({text:node.textContent,box:box(node),style:style(node),cells:[...node.querySelectorAll('th,td')].map(cell => [cell.textContent,style(cell)])})),
                                notes: [...document.querySelectorAll('.tab-pane.active .plotly-table-notes')].map(node => (node.querySelector('.review-original') || node).textContent),
                                panels: [...document.querySelectorAll('.tab-pane.active .plotly-table-panel')].map(box)
                            };
                        }""")
                        )
                    old, new = snapshots
                    assert old["chrome"] == new["chrome"] and old["charts"] == new["charts"]
                    assert old["notes"] == new["notes"]
                    for index, (old_table, new_table) in enumerate(zip(old["tables"], new["tables"], strict=True)):
                        assert old_table["text"] == new_table["text"] and old_table["style"] == new_table["style"]
                        assert old_table["cells"] == new_table["cells"]
                        assert (
                            old_table["box"][0] == new_table["box"][0] and old_table["box"][2:] == new_table["box"][2:]
                        )
                        if tab != 1:
                            assert old_table["box"] == new_table["box"]
                        elif index > 0:
                            height_delta = new["panels"][0][3] - old["panels"][0][3]
                            assert abs(new_table["box"][1] - old_table["box"][1] - height_delta) < 1
                    if tab == 1:
                        component = target.locator(".verdict-review")
                        assert component.count() == 1
                        assert component.locator("details[open]").count() == 0
                        assert component.locator(".review-metric").count() == 4
                        assert component.locator(".metric-threshold").get_attribute("data-threshold") == "0.2"
                        assert component.locator(".metric-threshold").get_attribute("d") == "M40 0V20"
                        assert component.locator(".review-metric svg").evaluate_all("""nodes => nodes.every(node => {
                            const point = node.querySelector('circle');
                            const box = node.getBoundingClientRect(), region = node.closest('.verdict-review').getBoundingClientRect();
                            const glyph = point.getBoundingClientRect();
                            return Number(point.getAttribute('cx')) >= 0 && Number(point.getAttribute('cx')) <= 200 &&
                                box.left >= region.left && box.right <= region.right &&
                                glyph.left >= region.left && glyph.right <= region.right && glyph.top >= box.top && glyph.bottom <= box.bottom &&
                                getComputedStyle(node.querySelector('.metric-bar')).stroke !== 'none';
                        })""")
                        assert component.locator("summary,.review-metric span,.review-metric strong").evaluate_all("""nodes => nodes.every(node => {
                            const range = document.createRange(); range.selectNodeContents(node);
                            const text = range.getBoundingClientRect(), bounds = node.getBoundingClientRect();
                            return text.left >= bounds.left - 1 && text.right <= bounds.right + 1 && text.top >= bounds.top - 1 && text.bottom <= bounds.bottom + 1;
                        })""")
                        target.locator(".nav-link").nth(tab).focus()
                        target.keyboard.press("Tab")
                        for summary in component.locator("summary").all():
                            assert summary.evaluate("node => node === document.activeElement")
                            summary.press("Enter")
                            assert summary.locator("..").get_attribute("open") is not None
                            assert component.locator(".review-original").is_visible()
                            assert component.locator(".review-original").evaluate("""node => {
                                const range = document.createRange(); range.selectNodeContents(node);
                                const text = range.getBoundingClientRect(), box = node.getBoundingClientRect();
                                return text.left >= box.left - 1 && text.right <= box.right + 1 && text.bottom <= box.bottom + 1;
                            }""")
                            summary.press("Space")
                            assert summary.locator("..").get_attribute("open") is None
                            assert summary.evaluate("node => getComputedStyle(node).outlineWidth") == "2px"
                            target.keyboard.press("Tab")
                        assert target.evaluate("document.activeElement.classList.contains('fin-wrap')")
                        target.locator(".nav-link").nth(tab).click()
                        for name, page in (("before", source), ("after", target)):
                            page.locator(".tab-pane.active .plotly-table-panel").first.screenshot(
                                path=str(artifacts / f"component-{name}-{width}-{theme}.png")
                            )
                            page.screenshot(path=str(artifacts / f"report-{name}-{width}-{theme}.png"), full_page=True)
                    records.append({"width": width, "theme": theme, "tab": tab, "outside_region_unchanged": True})
        assert errors == [] and external == []
        browser.close()
    (artifacts / "component-validation.json").write_text(
        json.dumps(
            {
                "comparisons": records,
                "mock_only": True,
                "page_errors": errors,
                "external_requests": external,
                "body_flow": "Only authorized component height can shift following strategy tables; their content/styles/sizes remain identical",
            },
            indent=2,
        )
    )
