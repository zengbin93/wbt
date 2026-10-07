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
    notes = "".join(
        f'<p class="plotly-table-notes">{_rich_text(annotation.get("text", ""))}</p>' for annotation in annotations
    )
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
