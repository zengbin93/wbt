"""wbt.generate_backtest_report integration tests (迁移自 czsc.utils.backtest_report)"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from wbt import WeightBacktest, generate_backtest_report
from wbt.report import HtmlReportBuilder
from wbt.report._generator import (
    _normalize_stats_for_czsc_view,
    _validate_input_data,
)
from wbt.report._plot_backtest import get_performance_metrics_cards

# ============================================================
# _validate_input_data
# ============================================================


def test_validate_rejects_missing_columns(sample_dfw: pd.DataFrame) -> None:
    bad = sample_dfw.drop(columns=["price"])
    with pytest.raises(ValueError, match="缺少必需列"):
        _validate_input_data(bad)


def test_validate_rejects_empty_dataframe() -> None:
    empty = pd.DataFrame({"dt": [], "symbol": [], "weight": [], "price": []})
    with pytest.raises(ValueError, match="不能为空"):
        _validate_input_data(empty)


def test_validate_rejects_nan_in_weight_or_price(sample_dfw: pd.DataFrame) -> None:
    bad = sample_dfw.copy()
    bad.loc[0, "weight"] = float("nan")
    with pytest.raises(ValueError, match="不能包含空值"):
        _validate_input_data(bad)


# ============================================================
# _normalize_stats_for_czsc_view
# ============================================================


def test_normalize_adds_czsc_aliases() -> None:
    wbt_stats = {"年化收益": 0.15, "夏普比率": 1.2, "卡玛比率": 0.8, "其他": "x"}
    out = _normalize_stats_for_czsc_view(wbt_stats)
    assert out["年化"] == 0.15
    assert out["夏普"] == 1.2
    assert out["卡玛"] == 0.8
    assert out["其他"] == "x"
    # 原 key 保留
    assert out["年化收益"] == 0.15


def test_normalize_preserves_existing_short_keys() -> None:
    # 如果输入已经有短名（来自 czsc），不覆盖
    stats = {"年化": 0.05, "年化收益": 0.10}
    out = _normalize_stats_for_czsc_view(stats)
    assert out["年化"] == 0.05  # 不被 "年化收益" 覆盖


def test_normalize_does_not_mutate_input() -> None:
    stats = {"年化收益": 0.15}
    out = _normalize_stats_for_czsc_view(stats)
    assert "年化" not in stats  # 原 dict 不变
    assert "年化" in out


# ============================================================
# get_performance_metrics_cards
# ============================================================


def test_metrics_cards_on_real_wbt_stats(wb: WeightBacktest) -> None:
    normalized = _normalize_stats_for_czsc_view(wb.stats)
    cards = get_performance_metrics_cards(normalized)
    assert isinstance(cards, list)
    assert len(cards) == 11
    labels = {c["label"] for c in cards}
    # 11 个指标的 label 与 czsc 版本一致
    assert labels == {
        "年化收益率",
        "单笔收益(BP)",
        "交易胜率",
        "持仓K线数",
        "最大回撤",
        "年化",
        "夏普",
        "卡玛",
        "年化波动率",
        "多头占比",
        "空头占比",
    }
    # 所有 value 都是格式化后的字符串（非 0%/0.00 说明实际取到数值）
    for c in cards:
        assert isinstance(c["value"], str)
        assert isinstance(c["is_positive"], bool)


# ============================================================
# HtmlReportBuilder basic contract
# ============================================================


def test_html_builder_render_contains_title_and_basic_structure() -> None:
    html = (
        HtmlReportBuilder(title="My Test Report")
        .add_header({"Period": "2020-2024"}, subtitle="Hello")
        .add_metrics([{"label": "Return", "value": "15%", "is_positive": True}])
        .add_footer("End of report")
        .render()
    )
    assert "<!DOCTYPE html>" in html
    assert "<title>My Test Report</title>" in html
    assert "My Test Report" in html
    assert "Period: 2020-2024" in html
    assert "Hello" in html
    assert "Return" in html
    assert "15%" in html
    assert "End of report" in html


def test_html_builder_save_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "subdir" / "report.html"
    builder = HtmlReportBuilder(title="x")
    builder.add_header({"a": "b"})
    builder.add_footer()
    returned = builder.save(str(out))
    assert returned == str(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# ============================================================
# generate_backtest_report end-to-end
# ============================================================


def test_generate_backtest_report_writes_valid_html(sample_dfw: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "backtest_report.html"
    returned = generate_backtest_report(sample_dfw, output_path=str(out_path), title="测试报告", n_jobs=1)
    assert returned == str(out_path)
    assert out_path.exists()

    html = out_path.read_text(encoding="utf-8")
    # A 结构：基本骨架
    assert "<!DOCTYPE html>" in html
    assert "<title>测试报告</title>" in html
    assert "header-section" in html
    assert "section-title" in html
    assert 'class="nav nav-tabs"' in html
    assert 'class="chart-card"' in html
    # 有两个 tab：回测统计 / 多空对比
    assert "回测统计" in html
    assert "多空对比" in html
    # 至少包含一张 plotly 图
    assert "plotly-graph-div" in html
    # 有页脚 wbt 署名（我们改的默认文案）
    assert "wbt 权重回测引擎" in html


def test_generate_backtest_report_default_path_in_cwd(
    sample_dfw: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    returned = generate_backtest_report(sample_dfw, n_jobs=1)
    expected = tmp_path / "backtest_report.html"
    assert os.path.abspath(returned) == str(expected)
    assert expected.exists()


def test_generate_backtest_report_rejects_bad_input(tmp_path: Path) -> None:
    bad = pd.DataFrame({"dt": [], "symbol": [], "weight": []})  # 缺 price
    with pytest.raises(ValueError):
        generate_backtest_report(bad, output_path=str(tmp_path / "x.html"))


def test_generate_backtest_report_contains_all_11_metric_labels(sample_dfw: pd.DataFrame, tmp_path: Path) -> None:
    """B 数据：HTML 中包含 czsc 原版 11 个绩效指标的 label"""
    out = tmp_path / "r.html"
    generate_backtest_report(sample_dfw, output_path=str(out), n_jobs=1)
    html = out.read_text(encoding="utf-8")
    for label in [
        "年化收益率",
        "单笔收益(BP)",
        "交易胜率",
        "持仓K线数",
        "最大回撤",
        "年化波动率",
        "多头占比",
        "空头占比",
    ]:
        assert label in html, f"missing metric label: {label}"
