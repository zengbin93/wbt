"""Render single-Table Plotly HTML fragments at the report composition boundary."""

from __future__ import annotations

import html
import json
import math
import re
from html.parser import HTMLParser

from ._html_tables import _fin_table

_FRAGMENT = re.compile(
    r'<div(?=[^>]*\bid="(?P<id>[^"]+)")(?=[^>]*\bclass="plotly-graph-div")[^>]*>'
    r"\s*</div>\s*<script(?:\s[^>]*)?>(?P<script>.*?)</script>",
    re.DOTALL,
)
_PREFIX = re.compile(
    r"\s*window\.PLOTLYENV\s*=\s*window\.PLOTLYENV\s*\|\|\s*{}\s*;"
    r'\s*if\s*\(document\.getElementById\("[^"]+"\)\)\s*{\s*'
)


class _RichText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("b", "strong", "i", "em", "br", "sub", "sup"):
            self.parts.append(f"<{tag}>")
        else:
            self.parts.append(html.escape(self.get_starttag_text()))

    def handle_endtag(self, tag: str) -> None:
        if tag in ("b", "strong", "i", "em", "sub", "sup"):
            self.parts.append(f"</{tag}>")
        elif tag != "br":
            self.parts.append(html.escape(f"</{tag}>"))

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))


def _rich_text(value: object) -> str:
    parser = _RichText()
    parser.feed(str(value))
    return "".join(parser.parts)


def _verdict_notes(value: object) -> str:
    text = _rich_text(value)
    match = re.fullmatch(
        r"<b>(history（逐年）：[^<]*)</b><br>(.*?)<b>(recent（近期窗口）：[^<]*)</b><br>(.*)",
        text,
        re.DOTALL,
    )
    if match is None:
        return f'<p class="plotly-table-notes">{text}</p>'
    history, _history_reason, recent, recent_content = match.groups()
    lines = recent_content.split("<br>")
    fields = []
    while lines and "：" in lines[0] and "<" not in lines[0]:
        label, value = lines.pop(0).split("：", 1)
        fields.append((label, value))
    sections = []
    for title in (history, recent):
        label, status = title.split("：", 1)
        if status not in ("✅ 可用", "❌ 不可用"):
            return f'<p class="plotly-table-notes">{text}</p>'
        symbol = "M9 16l4 4 10-12" if status == "✅ 可用" else "M10 10l12 12M22 10L10 22"
        sections.append(
            '<div class="review-state">'
            f'<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="14"/>'
            f'<path d="{symbol}"/></svg><div><span>{label}</span><strong>{status}</strong></div></div>'
        )
    metrics = []
    for label, value in fields:
        if label not in ("近期绝对收益", "近期超额收益", "近期超额回撤", "历史超额回撤(剔除近期)"):
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?%", value) is None:
            continue
        number = float(value[:-1])
        signed = "收益" in label
        if not (-100 <= number <= 100 if signed else 0 <= number <= 100):
            continue
        position = 100 + number if signed else number * 2
        origin = 100 if signed else 0
        start, end = sorted((origin, position))
        axis = "−100% · 0 · +100%" if signed else "0 · 100%"
        threshold_graph = ""
        threshold_label = ""
        reason = " ".join(lines)
        threshold_match = re.search(
            r"(?:^|[(,;])\s*alpha_max_drawdown=([0-9]+\.[0-9]+) ≥ threshold ([0-9]+\.[0-9]+)"
            r"(?=\s*(?:$|[);]))",
            reason,
        )
        if len(re.findall(r"alpha_max_drawdown\s*=", reason)) != 1:
            threshold_match = None
        if label == "近期超额回撤" and threshold_match is not None:
            reported, threshold = map(float, threshold_match.groups())
            if 0 <= reported <= 1 and f"{reported * 100:.2f}%" == value and 0 <= threshold <= 1:
                threshold_graph = (
                    f'<path class="metric-threshold" data-threshold="{threshold:g}" d="M{threshold * 200:g} 0V20"/>'
                )
                threshold_label = f" · 原文阈值 {threshold * 100:g}%"
        metrics.append(
            f'<div class="review-metric"><div><span>{label}</span><strong>{value}</strong></div>'
            f'<svg viewBox="0 0 200 20" role="img" aria-label="{label}：{value}，刻度 {axis}">'
            '<path class="metric-track" d="M0 10H200"/>'
            f'<path class="metric-bar" d="M{start:g} 10H{end:g}"/>'
            f'<path class="metric-zero" d="M{origin} 2V18"/>'
            f'{threshold_graph}<circle class="metric-point" cx="{position:g}" cy="10" r="3"/></svg>'
            f"<small>{axis}{threshold_label}</small></div>"
        )
    return (
        '<div class="plotly-table-notes verdict-review">'
        f'<div class="review-states">{"".join(sections)}</div><div class="review-metrics">{"".join(metrics)}</div>'
        '<details class="review-section"><summary>完整判定与窗口详情</summary>'
        f'<div class="review-original">{text}</div></details></div>'
    )


def _background(value: object, column: int, row: int) -> str:
    for position in (column, row):
        if isinstance(value, list) and value:
            value = value[position % len(value)]
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{3,8}|rgba?\([0-9., %]+\)|[a-zA-Z]+", value):
        return value
    return "transparent"


def _unsupported_annotation(annotation: dict) -> bool:
    return (
        annotation.get("visible", True) is not True
        or annotation.get("opacity", 1) != 1
        or annotation.get("clicktoshow", False) is not False
        or "templateitemname" in annotation
    )


def _unsupported_display(trace: dict, layout: dict) -> bool:
    template = layout.get("template", {})
    if not isinstance(template, dict):
        return True
    template_data, template_layout = template.get("data", {}), template.get("layout", {})
    if not isinstance(template_data, dict) or not isinstance(template_layout, dict):
        return True
    tables = template_data.get("table", [])
    if not isinstance(tables, list):
        return True
    for table in [trace, *tables]:
        if not isinstance(table, dict) or table.get("visible", True) is not True:
            return True
        if any(key in table for key in ("columnorder", "domain")):
            return True
        for key in ("header", "cells"):
            section = table.get(key, {})
            if not isinstance(section, dict) or any(
                section.get(modifier) is not None for modifier in ("format", "prefix", "suffix")
            ):
                return True
    if template_layout.get("annotations"):
        return True
    for settings in (layout, template_layout):
        defaults = settings.get("annotationdefaults", {})
        if not isinstance(defaults, dict) or _unsupported_annotation(defaults):
            return True
        if settings.get("updatemenus") or settings.get("sliders"):
            return True
    return False


def _table_fragment(match: re.Match[str]) -> str:
    script = match["script"]
    call = re.search(r"Plotly\.newPlot\(", script)
    if call is None or _PREFIX.fullmatch(script[: call.start()]) is None:
        return match[0]
    decoder = json.JSONDecoder()
    position = call.end()
    arguments = []
    try:
        for index in range(4):
            while script[position].isspace():
                position += 1
            value, length = decoder.raw_decode(script[position:])
            arguments.append(value)
            position += length
            while position < len(script) and script[position].isspace():
                position += 1
            if index < 3:
                if script[position] != ",":
                    return match[0]
                position += 1
    except (ValueError, IndexError):
        return match[0]
    if re.fullmatch(r"\)\s*;?\s*}\s*;?\s*", script[position:]) is None:
        return match[0]
    plot_id, traces, layout, _config = arguments
    if plot_id != match["id"] or not isinstance(traces, list) or len(traces) != 1:
        return match[0]
    trace = traces[0]
    if not isinstance(trace, dict) or trace.get("type") != "table" or not isinstance(layout, dict):
        return match[0]
    if _unsupported_display(trace, layout):
        return match[0]
    header, cells = trace.get("header"), trace.get("cells")
    if not isinstance(header, dict) or not isinstance(cells, dict):
        return match[0]
    headers, columns = header.get("values"), cells.get("values")
    if not isinstance(headers, list) or not headers or not isinstance(columns, list) or len(headers) != len(columns):
        return match[0]
    if any(not isinstance(column, list) for column in columns) or len({len(column) for column in columns}) != 1:
        return match[0]
    if any(not isinstance(value, str) for value in headers) or any(
        not isinstance(value, str) for column in columns for value in column
    ):
        return match[0]
    title_config = layout.get("title", {})
    annotations = layout.get("annotations", [])
    fill_config = cells.get("fill", {})
    if not isinstance(title_config, dict) or not isinstance(fill_config, dict) or not isinstance(annotations, list):
        return match[0]
    if any(not isinstance(annotation, dict) for annotation in annotations):
        return match[0]
    if any(_unsupported_annotation(annotation) for annotation in annotations):
        return match[0]
    title = title_config.get("text", "")
    notes = "".join(_verdict_notes(annotation.get("text", "")) for annotation in annotations)
    rows = [[_rich_text(value) for value in row] for row in zip(*columns, strict=True)]
    fill = fill_config.get("color")
    backgrounds = [[_background(fill, column, row) for column in range(len(headers))] for row in range(len(rows))]
    table = _fin_table(
        [str(value) for value in headers], rows, label=str(title) or "报告数据表，可左右滚动", backgrounds=backgrounds
    )
    height = layout.get("height", 300)
    if not isinstance(height, (int, float)) or not math.isfinite(height) or height < 0:
        height = 300
    heading = f'<div class="plotly-table-title">{_rich_text(title)}</div>' if title else ""
    return (
        f'<div id="{html.escape(plot_id, quote=True)}" class="plotly-table-panel" style="min-height:{height}px">'
        f"{heading}{notes}{table}</div>"
    )


def replace_plotly_tables(content: str) -> str:
    """Replace standard to_html single-Table fragments; leave other/custom scripts intact."""
    return _FRAGMENT.sub(_table_fragment, content)
