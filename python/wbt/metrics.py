"""Stable machine names and Chinese display labels; legacy API keys remain supported."""

from collections.abc import Mapping

COMPARE_METRICS = ["年化收益", "夏普比率", "卡玛比率", "最大回撤", "年化波动率", "日胜率"]
COMPARE_SIDES = ["多空", "多头", "空头", "基准", "超额"]

METRIC_FIELDS = {
    "absolute_return": ("绝对收益",),
    "annual_returns": ("年化收益", "年化"),
    "sharpe_ratio": ("夏普比率", "夏普"),
    "calmar_ratio": ("卡玛比率", "卡玛"),
    "max_drawdown": ("最大回撤",),
    "annual_volatility": ("年化波动率",),
    "daily_win_rate": ("日胜率",),
    "downside_volatility": ("下行波动率",),
    "new_high_interval": ("新高间隔",),
    "new_high_ratio": ("新高占比",),
    "daily_profit_loss_ratio": ("日盈亏比",),
    "daily_win_probability": ("日赢面",),
    "non_zero_coverage": ("非零覆盖",),
    "break_even_point": ("盈亏平衡点",),
    "drawdown_risk": ("回撤风险",),
    "annual_lin_reg_cumsum_return": ("回归年度回报率",),
    "length_adjusted_average_max_drawdown": ("长度调整平均最大回撤",),
}
METRIC_LABELS = {key: names[0] for key, names in METRIC_FIELDS.items()}
_NAME_TO_KEY = {name: key for key, names in METRIC_FIELDS.items() for name in (key, *names)}


def lookup_metric(stats: Mapping, name: str) -> object:
    """Read a machine name or legacy label; machine key wins if both are present."""
    key = _NAME_TO_KEY.get(name, name)
    for alias in (key, *METRIC_FIELDS.get(key, ())):
        if alias in stats:
            return stats[alias]
    return None


def to_machine_metrics(stats: Mapping) -> dict:
    """Normalize known metric names, preserving other fields and their units."""
    out = {key: value for key, value in stats.items() if key not in _NAME_TO_KEY}
    for key, names in METRIC_FIELDS.items():
        if any(name in stats for name in (key, *names)):
            out[key] = lookup_metric(stats, key)
    return out


YEARLY_COLS: list[tuple[str, str]] = [
    ("year", "年份"),
    ("abs_return", "绝对收益"),
    ("alpha_return", "超额收益"),
    ("alpha_max_drawdown", "超额回撤"),
    ("days", "交易日数"),
    ("is_complete_year", "完整年"),
    ("year_passed", "达标"),
]


RECENT_KEYS: list[tuple[str, str]] = [
    ("recent_start_date", "窗口开始"),
    ("recent_end_date", "窗口结束"),
    ("recent_actual_days", "实际交易日"),
    ("recent_abs_return", "近期绝对收益"),
    ("recent_alpha_return", "近期超额收益"),
    ("recent_alpha_max_drawdown", "近期超额回撤"),
    ("history_alpha_max_drawdown_excl_recent", "历史超额回撤(剔除近期)"),
    ("history_window_empty", "历史窗口不足"),
]


DETAIL_LABELS: dict[str, str] = {
    "mode": "模式",
    "is_good": "可用",
    "complete_year_count": "完整年度数",
    "cond_yearly_passed": "逐年达标",
    "cond_recent_return_passed": "近期收益/回撤达标",
    "cond_recent_dd_passed": "近期回撤优于历史",
    "recent_start_date": "近期起始",
    "recent_end_date": "近期结束",
    "recent_actual_days": "近期实际天数",
    "recent_abs_return": "近期绝对收益",
    "recent_alpha_return": "近期超额收益",
    "recent_alpha_max_drawdown": "近期超额回撤",
    "history_alpha_max_drawdown_excl_recent": "历史超额回撤(剔除近期)",
    "history_window_empty": "历史窗口不足",
    "alpha_degenerate": "alpha 退化",
    "reason": "原始判定说明",
}
