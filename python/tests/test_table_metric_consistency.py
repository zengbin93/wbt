"""The Plotly and native HTML renderers share labels and alias precedence."""

import html
import re
from types import SimpleNamespace

import pytest

from wbt.plotting.tables import plot_segment_comparison, plot_stats_comparison
from wbt.report._html_tables import segment_comparison_html, stats_comparison_html


@pytest.mark.parametrize(
    "stats",
    [
        {"年化收益": 0.12, "夏普比率": 1.2, "卡玛比率": 2.0},
        {"年化": 0.12, "夏普": 1.2, "卡玛": 2.0},
        {"annual_returns": 0.12, "sharpe_ratio": 1.2, "calmar_ratio": 2.0},
        {"annual_returns": 0.12, "年化收益": 0.99, "年化": 0.98, "夏普比率": 1.2, "卡玛比率": 2.0},
        {"年化收益": None, "年化": 0.99, "夏普": float("nan")},
        {},
    ],
)
@pytest.mark.parametrize("segment", [False, True])
def test_rendered_metric_cells_agree(stats, segment):
    result = SimpleNamespace(
        stats=stats, stats_by_side={"基准": stats}, segment_comparison={"全样本": stats, "近1年": stats}
    )
    plot = plot_segment_comparison(result) if segment else plot_stats_comparison(result)
    markup = segment_comparison_html(result) if segment else stats_comparison_html(result)
    html_cells = [html.unescape(re.sub(r"<[^>]+>", "", cell)) for cell in re.findall(r"<td>(.*?)</td>", markup)]
    columns = plot.data[0].cells.values
    plot_cells = [cell for row in zip(*columns, strict=True) for cell in row]
    assert html_cells == plot_cells
    assert list(columns[0]) == ["年化收益", "夏普比率", "卡玛比率", "最大回撤", "年化波动率", "日胜率"]
    if stats and stats.get("年化收益", 0) is not None:
        assert columns[1][0] == "12.00%"
    else:
        assert columns[1][0] == "—"
