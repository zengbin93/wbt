"""权重回测 HTML 报告生成器

基于 BacktestResult 标准数据 + wbt.plotting 绘图函数生成 HTML 报告。
所有数据预处理在 WeightBacktest.to_result() 一次完成，报告侧只做组合与布局。
"""

from __future__ import annotations

import html
import os
from typing import Any

import pandas as pd

from wbt.backtest import WeightBacktest
from wbt.plotting import (
    plot_cumulative_returns,
    plot_daily_return_dist,
    plot_drawdown,
    plot_key_trades,
    plot_monthly_heatmap,
    plot_pairs_hold_dist,
    plot_pairs_pnl_dist,
    plot_rolling_metrics,
    plot_symbol_returns,
    plot_yearly_returns,
)
from wbt.result import BacktestResult

from . import _html_tables as ht
from .html_builder import HtmlReportBuilder


def get_performance_metrics_cards(stats: dict[str, Any]) -> list[dict[str, Any]]:
    """从 stats（中文长名字段）提取核心绩效指标卡。

    :param stats: WeightBacktest.stats / BacktestResult.stats
    :return: 指标卡列表，元素为 {label, value, is_positive}
    """

    def g(key: str) -> float:
        v = stats.get(key, 0)
        return v if isinstance(v, (int, float)) else 0.0

    # 中式红涨绿跌：盈利/正向 → is_positive=True(红)，亏损/负向 → False(绿)；
    # 无涨跌语义的结构指标（波动率/占比/持仓）用 neutral=True(蓝)。
    return [
        {"label": "年化收益率", "value": f"{g('年化收益'):.2%}", "is_positive": g("年化收益") > 0},
        {"label": "绝对收益", "value": f"{g('绝对收益'):.2%}", "is_positive": g("绝对收益") > 0},
        {"label": "夏普", "value": f"{g('夏普比率'):.2f}", "neutral": True, "is_positive": g("夏普比率") > 0},
        {"label": "卡玛", "value": f"{g('卡玛比率'):.2f}", "neutral": True, "is_positive": g("卡玛比率") > 0},
        {"label": "最大回撤", "value": f"{g('最大回撤'):.2%}", "neutral": True, "is_positive": False},
        {"label": "年化波动率", "value": f"{g('年化波动率'):.2%}", "neutral": True, "is_positive": False},
        {"label": "下行波动率", "value": f"{g('下行波动率'):.2%}", "neutral": True, "is_positive": False},
        {"label": "单笔收益(BP)", "value": f"{g('单笔收益'):.2f}", "is_positive": g("单笔收益") > 0},
        {"label": "单笔盈亏比", "value": f"{g('单笔盈亏比'):.2f}", "neutral": True, "is_positive": g("单笔盈亏比") > 1},
        {"label": "交易胜率", "value": f"{g('交易胜率'):.2%}", "neutral": True, "is_positive": g("交易胜率") > 0.5},
        {"label": "日胜率", "value": f"{g('日胜率'):.2%}", "neutral": True, "is_positive": g("日胜率") > 0.5},
        {"label": "持仓K线数", "value": f"{g('持仓K线数'):.0f}", "neutral": True, "is_positive": True},
        {"label": "多头占比", "value": f"{g('多头占比'):.2%}", "neutral": True, "is_positive": True},
        {"label": "空头占比", "value": f"{g('空头占比'):.2%}", "neutral": True, "is_positive": True},
    ]


def _validate_input_data(df: pd.DataFrame) -> None:
    """验证输入数据格式。"""
    required_columns = ["dt", "symbol", "weight", "price"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必需列: {missing_columns}")
    if len(df) == 0:
        raise ValueError("输入数据不能为空")
    if df[["weight", "price"]].isna().any().any():
        raise ValueError("权重和价格列不能包含空值")


def _prepare_config(kwargs: dict) -> dict[str, Any]:
    default_config = {"fee_rate": 0.0002, "digits": 2, "weight_type": "ts", "yearly_days": 252, "n_jobs": 1}
    return {**default_config, **kwargs}


def _build_report_params(result: BacktestResult, config: dict[str, Any]) -> dict[str, str]:
    return {
        "日期范围": f"{result.start_date} ~ {result.end_date}",
        "手续费": f"{config['fee_rate'] * 10000:.2f} BP",
        "小数位": str(config["digits"]),
        "年交易日": str(config["yearly_days"]),
        "标的数": str(result.symbol_count),
    }


_PLOT_CONFIG = {"responsive": True, "displayModeBar": "hover", "displaylogo": False, "scrollZoom": True}


def _err_div(name: str, e: Exception) -> str:
    return f"<div style='padding:20px;text-align:center;color:#e0382c;'>{html.escape(f'{name}生成失败: {e}')}</div>"


def _safe_fig(name: str, build, *, include_plotlyjs: bool) -> str:
    """plotly 图表面板 → HTML；失败降级为错误 div，不拖垮整份报告。"""
    try:
        fig = build()
        if not fig.data:
            return '<p class="analysis-note">暂无可绘制样本；可能尚未达到该指标的观测要求。</p>'
        note = ""
        if name == "日收益分布":
            note = " · ".join(str(annotation.text) for annotation in fig.layout.annotations)
            fig.update_layout(annotations=[])
        elif name == "波动率归一累计收益":
            note = (
                "使用结果快照中的波动率归一曲线，不代表实际收益；多头超额为归一多头减基准，空头超额为归一空头加基准。"
            )
        elif name.startswith("年度收益"):
            note = "年度数据来自引擎快照；首尾年份可能不是完整年度，完整年度标识见策略诊断。"
        elif name in ("盈亏比例分布", "持仓K线数分布"):
            note = "基于聚合成交对的样本分布，不等同于原始成交笔数。"
        fig.update_layout(autosize=True, font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 12})
        markup = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False, config=_PLOT_CONFIG)
        if any(trace.type == "heatmap" for trace in fig.data):
            return f'<div class="heatmap-scroll" tabindex="0" role="region" aria-label="月度图表，可左右滚动">{markup}</div>'
        return markup + (f'<p class="analysis-note">{html.escape(note)}</p>' if note else "")
    except Exception as e:  # noqa: BLE001 — 面板级隔离
        return _err_div(name, e)


def _safe_html(name: str, build) -> str:
    """原生 HTML 表格/卡片面板；失败降级为错误 div。"""
    try:
        return build()
    except Exception as e:  # noqa: BLE001 — 面板级隔离
        return _err_div(name, e)


def _monthly_figure(result: BacktestResult):
    figure = plot_monthly_heatmap(result, title="")
    if figure.data:
        observed = {(str(date)[:4], str(date)[5:7]) for date in result.dates}
        figure.data[0].z = [
            [
                float(result.monthly.z[index][column]) if (str(year), f"{month:02d}") in observed else None
                for column, month in enumerate(result.monthly.months)
            ]
            for index, year in enumerate(result.monthly.years)
        ]
        figure.data[0].text = [
            [
                str(result.monthly.text[index][column]) if (str(year), f"{month:02d}") in observed else "—"
                for column, month in enumerate(result.monthly.months)
            ]
            for index, year in enumerate(result.monthly.years)
        ]
    return figure


# 面板定义：(标签名, [(小标题, make(include_plotlyjs)->html, 是否整行, 是否 plotly 图), ...])
def _tab_specs(result: BacktestResult):
    def fig(title, build, fw=True):
        return (title, lambda inc, b=build, t=title: _safe_fig(t, b, include_plotlyjs=inc), fw, True)

    def tbl(title, build, fw=True):
        return (title, lambda inc, b=build, t=title: _safe_html(t, b), fw, False)

    return [
        (
            "概览",
            [
                fig("累计收益（原始）", lambda: plot_cumulative_returns(result, keys=["多空", "基准"], title="")),
                tbl("样本与阅读指南", lambda: ht.sample_profile_html(result)),
            ],
        ),
        (
            "收益表现",
            [
                fig(
                    "收益来源 · 多空与基准",
                    lambda: plot_cumulative_returns(result, keys=["多空", "多头", "空头", "基准"], title=""),
                ),
                fig("月度收益热力图 · 未观测月份留空", lambda: _monthly_figure(result), fw=False),
                fig("年度收益（绝对 vs 超额）", lambda: plot_yearly_returns(result, title=""), fw=False),
                fig("品种收益分布 · 非组合贡献", lambda: plot_symbol_returns(result, title="")),
                tbl("收益数据明细", lambda: ht.return_details_html(result)),
                tbl("关键指标对比", lambda: ht.stats_comparison_html(result)),
            ],
        ),
        (
            "风险回撤",
            [
                fig("回撤路径", lambda: plot_drawdown(result, title="")),
                tbl("回撤明细（Top 10）", lambda: ht.drawdowns_table_html(result)),
                fig("日收益分布", lambda: plot_daily_return_dist(result, title="")),
                fig("滚动指标（252日窗口，最少100日）", lambda: plot_rolling_metrics(result, title="")),
            ],
        ),
        (
            "交易与持仓",
            [
                fig("盈亏比例分布", lambda: plot_pairs_pnl_dist(result, title=""), fw=False),
                fig("持仓K线数分布", lambda: plot_pairs_hold_dist(result, title=""), fw=False),
                fig("关键交易（每年最赚/最亏 · hover 查看开平与持仓详情）", lambda: plot_key_trades(result, title="")),
                tbl("关键交易明细", lambda: ht.key_trades_html(result)),
            ],
        ),
        (
            "策略诊断",
            [
                tbl("策略判定（history + recent）", lambda: ht.verdict_section_html(result)),
                tbl("分段对比（近1年 vs 全样本）", lambda: ht.segment_comparison_html(result)),
                fig(
                    "波动率归一累计收益",
                    lambda: plot_cumulative_returns(
                        result, keys=["多空", "多头", "空头", "基准", "多头超额", "空头超额"], voladj=True, title=""
                    ),
                    fw=True,
                ),
                tbl("完整绩效指标", lambda: ht.stats_kv_html(result)),
            ],
        ),
    ]


def _generate_chart_tabs(result: BacktestResult) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """生成各标签页面板 HTML。plotly.js 仅在全局第一个 plotly 面板内联一次。"""
    tabs: list[tuple[str, list[tuple[str, str, bool]]]] = []
    plotlyjs_emitted = False
    for tab_name, panels in _tab_specs(result):
        items: list[tuple[str, str, bool]] = []
        for sub_title, make, full_width, is_plotly in panels:
            include = is_plotly and not plotlyjs_emitted
            panel_html = make(include)
            if include and "Plotly" in panel_html:
                plotlyjs_emitted = True
            items.append((sub_title, panel_html, full_width))
        tabs.append((tab_name, items))
    return tabs


def _render_backtest_report(result: BacktestResult, config: dict[str, Any], output_path: str, title: str) -> str:
    """将已有回测结果组装并保存为 HTML 报告。"""
    metrics = [
        {"label": label, "value": ht.fmt_value(key, result.stats.get(key)), "neutral": True}
        for label, key in (
            ("绝对收益", "绝对收益"),
            ("年化收益率", "年化收益"),
            ("夏普比率", "夏普比率"),
            ("最大回撤", "最大回撤"),
        )
    ]
    tabs = _generate_chart_tabs(result)
    icons = ["bi-grid-1x2", "bi-clipboard-check", "bi-activity", "bi-arrows-collapse", "bi-star"]

    builder = HtmlReportBuilder(title=title)
    builder.add_header(_build_report_params(result, config), subtitle="从收益轨迹到交易证据 · 权重策略研究报告")
    builder.add_metrics(metrics, title="全样本摘要")
    for i, (tab_name, items) in enumerate(tabs):
        builder.add_chart_grid_tab(
            tab_name, items, cols=2, icon=icons[i] if i < len(icons) else "bi-graph-up", active=(i == 0)
        )
    builder.add_charts_section()
    builder.add_footer()
    builder.save(output_path)

    return output_path


def generate_backtest_report(
    df: pd.DataFrame, output_path: str | None = None, title: str = "权重回测报告", **kwargs
) -> str:
    """生成权重回测的 HTML 报告。

    :param df: 包含 dt, symbol, weight, price 列的权重数据
    :param output_path: HTML 文件输出路径，默认当前目录 backtest_report.html
    :param title: 报告标题
    :param kwargs: fee_rate / digits / weight_type / yearly_days / n_jobs / target_vol
    :return: HTML 文件路径
    """
    _validate_input_data(df)

    config = _prepare_config(kwargs)
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "backtest_report.html")

    wb = WeightBacktest(
        df,
        fee_rate=config["fee_rate"],
        digits=config["digits"],
        weight_type=config["weight_type"],
        yearly_days=config["yearly_days"],
        n_jobs=config["n_jobs"],
    )
    result = wb.to_result(target_vol=config.get("target_vol", 0.20))

    return _render_backtest_report(result, config, output_path, title)
