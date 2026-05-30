"""报告侧原生 HTML 表格与判定卡构建器。

go.Table 样式难控、随主题切换困难；报告内的表格改用原生 HTML，由报告 CSS
设计系统统一着色、随明暗主题自动切换、等宽数字对齐、红涨绿跌着色。
所有数据直接取自 BacktestResult，零额外计算。
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from wbt.plotting._common import fmt_value

if TYPE_CHECKING:
    from wbt.result import BacktestResult

# 关键指标对比所用指标 + 基准/超额（来自 daily_performance）键名别名
_CMP_METRICS = ["年化收益", "夏普比率", "卡玛比率", "最大回撤", "年化波动率", "日胜率"]
_ALIASES: dict[str, tuple[str, ...]] = {
    "年化收益": ("年化收益", "年化"),
    "夏普比率": ("夏普比率", "夏普"),
    "卡玛比率": ("卡玛比率", "卡玛"),
}


def _mget(d: dict, key: str) -> object:
    for alias in _ALIASES.get(key, (key,)):
        if alias in d:
            return d[alias]
    return None


def _is_signed(key: str) -> bool:
    """是否为带正负方向的收益类字段（用红涨绿跌着色）。回撤/波动率/胜率不在此列。"""
    return "收益" in key or key in ("abs_return", "alpha_return")


def _cell(key: str, v: object) -> str:
    """格式化为单元格 HTML：收益类按红涨绿跌着色，其余原样。"""
    text = html.escape(fmt_value(key, v))
    if _is_signed(key) and isinstance(v, (int, float)) and not isinstance(v, bool):
        cls = "t-up" if v > 0 else ("t-down" if v < 0 else "")
        if cls:
            return f'<span class="{cls}">{text}</span>'
    return text


def _fin_table(headers: list[str], rows: list[list[str]]) -> str:
    """生成 .fin-table 原生表格（首列左对齐标签，其余右对齐数值，单元格已是 HTML）。"""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="fin-wrap"><table class="fin-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def verdict_card_html(result: BacktestResult) -> str:
    """策略判定卡：状态徽章 + 结构化条件清单 + 年度明细表（取代英文 reason 堆叠）。"""
    v = result.verdict
    is_good = bool(v.get("is_good"))
    badge_cls, badge_txt = ("good", "✓ 可用") if is_good else ("bad", "✗ 不可用")

    ym = sorted(v.get("yearly_metrics") or [], key=lambda m: m["year"])
    dd = v.get("history_alpha_max_drawdown")
    dd_txt = f"{dd:.2%}" if isinstance(dd, (int, float)) else "—"
    ny = v.get("complete_year_count")
    ny_txt = str(ny) if ny is not None else "—"

    conds: list[tuple[str, str, str]] = []
    if "cond_yearly_passed" in v:
        ok = bool(v.get("cond_yearly_passed"))
        fails = [str(int(m["year"])) for m in ym if m.get("is_complete_year") and not m.get("year_passed")]
        detail = "全部完整年度达标" if ok else ("未达标年度：" + "、".join(fails) if fails else "存在未达标年度")
        conds.append(("ok" if ok else "no", "逐年达标", detail))
    if "cond_history_dd_passed" in v:
        ok = bool(v.get("cond_history_dd_passed"))
        detail = f"超额最大回撤 {dd_txt}，阈值 20%" + ("" if ok else "（已超阈值）")
        conds.append(("ok" if ok else "no", "超额回撤控制", detail))
    if v.get("alpha_degenerate"):
        conds.append(("no", "alpha 退化", "多头或基准波动率为 0，超额无法定义"))

    cond_html = "".join(
        f'<div class="verdict-cond {st}"><span class="ck">{"✓" if st == "ok" else "✗"}</span>'
        f'<span class="ct">{html.escape(t)}</span><span class="cd">{html.escape(d)}</span></div>'
        for st, t, d in conds
    )

    table_html = ""
    if ym:
        rows = []
        for m in ym:
            passed = bool(m.get("year_passed"))
            badge = (
                '<span class="badge badge-pass">达标</span>'
                if passed
                else '<span class="badge badge-fail">未达标</span>'
            )
            rows.append(
                [
                    str(int(m["year"])),
                    _cell("绝对收益", m.get("abs_return")),
                    _cell("超额收益", m.get("alpha_return")),
                    fmt_value("交易日数", m.get("days")),
                    badge,
                ]
            )
        table_html = _fin_table(["年份", "绝对收益", "超额收益", "交易日数", "达标"], rows)

    sub = f"{ny_txt} 个完整年度 · 超额最大回撤 {dd_txt}"
    return (
        '<div class="verdict">'
        f'<div class="verdict-head"><span class="verdict-badge {badge_cls}">{badge_txt}</span>'
        f'<span class="verdict-sub">{html.escape(sub)}</span></div>'
        f'<div class="verdict-conds">{cond_html}</div>'
        f"{table_html}</div>"
    )


def stats_kv_html(result: BacktestResult) -> str:
    """完整绩效指标：紧凑的 label/value 键值网格（取代高瘦的彩色表）。"""
    skip = {"开始日期", "结束日期"}
    items = ""
    for k, val in result.stats.items():
        if k in skip:
            continue
        items += (
            f'<div class="kv"><span class="kv-k">{html.escape(k)}</span><span class="kv-v">{_cell(k, val)}</span></div>'
        )
    return f'<div class="kv-grid">{items}</div>'


def stats_comparison_html(result: BacktestResult) -> str:
    """多空/多头/空头/基准/超额 关键指标对比表。"""
    sides = {"多空": result.stats, **result.stats_by_side}
    order = [s for s in ("多空", "多头", "空头", "基准", "超额") if s in sides]
    rows = [[m, *[_cell(m, _mget(sides[s], m)) for s in order]] for m in _CMP_METRICS]
    return _fin_table(["指标", *order], rows)


def segment_comparison_html(result: BacktestResult) -> str:
    """全样本 vs 近 1 年 关键指标对比表。"""
    seg = result.segment_comparison
    order = [s for s in ("全样本", "近1年") if s in seg]
    rows = [[m, *[_cell(m, _mget(seg[s], m)) for s in order]] for m in _CMP_METRICS]
    return _fin_table(["指标", *order], rows)


def drawdowns_table_html(result: BacktestResult) -> str:
    """Top 回撤明细表。"""
    rows_data = result.drawdowns
    if not rows_data:
        return '<div class="fin-wrap"><p class="verdict-sub" style="padding:1rem">暂无回撤记录</p></div>'
    headers = list(rows_data[0].keys())
    rows = [[_cell(h, r.get(h)) for h in headers] for r in rows_data]
    return _fin_table(headers, rows)
