"""wbt.generate_backtest_report 集成测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from wbt import WeightBacktest, generate_backtest_report
from wbt.report import HtmlReportBuilder, get_performance_metrics_cards
from wbt.report._generator import _validate_input_data

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
# get_performance_metrics_cards（直接吃中文长名 stats）
# ============================================================


def test_metrics_cards_on_real_wbt_stats(wb: WeightBacktest) -> None:
    cards = get_performance_metrics_cards(wb.stats)
    assert isinstance(cards, list)
    labels = [c["label"] for c in cards]
    assert len(cards) == 14
    assert len(labels) == len(set(labels)), f"指标卡标签不应重复: {labels}"
    assert set(labels) == {
        "年化收益率",
        "绝对收益",
        "夏普",
        "卡玛",
        "最大回撤",
        "年化波动率",
        "下行波动率",
        "单笔收益(BP)",
        "单笔盈亏比",
        "交易胜率",
        "日胜率",
        "持仓K线数",
        "多头占比",
        "空头占比",
    }
    for c in cards:
        assert isinstance(c["value"], str)
        assert isinstance(c["is_positive"], bool)


def test_metrics_cards_from_result(wb: WeightBacktest) -> None:
    """result.stats 与 wb.stats 同构，指标卡一致。"""
    result = wb.to_result()
    assert get_performance_metrics_cards(result.stats) == get_performance_metrics_cards(wb.stats)


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
    assert "Period" in html and "2020-2024" in html  # 徽章：<key> <b><value></b>
    assert 'data-theme' in html and "theme-switch" in html  # 双主题切换已注入
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
    assert "<!DOCTYPE html>" in html
    assert "<title>测试报告</title>" in html
    assert "header-section" in html
    assert "section-title" in html
    assert 'class="nav nav-tabs"' in html
    assert 'class="chart-card"' in html
    assert "chart-grid" in html  # 单图以 CSS 网格排布（已拆开组合图）
    assert html.count("plotly-graph-div") >= 2  # 一个标签页内多张独立图
    assert "回测概览" in html
    assert "策略审核" in html
    assert "稳健性分析" in html
    assert "多空对比" in html
    assert "交易分析" in html
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


def test_generate_backtest_report_contains_metric_labels(sample_dfw: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    generate_backtest_report(sample_dfw, output_path=str(out), n_jobs=1)
    html = out.read_text(encoding="utf-8")
    for label in [
        "年化收益率",
        "绝对收益",
        "下行波动率",
        "单笔盈亏比",
        "日胜率",
        "持仓K线数",
        "多头占比",
        "空头占比",
    ]:
        assert label in html, f"missing metric label: {label}"
