"""BacktestResult：绘图与审核页面的统一输入数据模型。

设计见飞书文档 https://s0cqcxuy3p.feishu.cn/wiki/JwQjwdN4iiycLBkR9VCceHf4nih

核心约定：
- 字段即 plotly trace 的 x/y/z 直接入参，下游绘图零数据转换。
- 比例类数值一律原始小数（0.01 = 1%）；仅 ``*_pct`` 字段为百分比（已 ×100）。
- 轻量字段在 ``from_backtest`` 构造期算好；``curves_voladj`` / ``drawdowns`` /
  ``key_trades`` / ``verdict`` 为按需 ``cached_property``。
- ``to_dict()`` 产出 JSON 安全结构，供审核页面走 HTTP。
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from wbt.top_drawdowns import top_drawdowns

if TYPE_CHECKING:
    from wbt.backtest import WeightBacktest

logger = logging.getLogger(__name__)

# curves 键 → daily_return / alpha 的来源列
_CURVE_KEYS = ("多空", "多头", "空头", "基准", "超额")


@dataclass(frozen=True)
class Curve:
    """一条收益曲线（原始口径）。三序列与 BacktestResult.dates 等长、一一对应。"""

    daily: np.ndarray
    cum: np.ndarray
    drawdown: np.ndarray


@dataclass(frozen=True)
class ReturnDist:
    """日收益分布（单位：百分比）。"""

    values_pct: np.ndarray
    mean_pct: float
    std_pct: float


@dataclass(frozen=True)
class MonthlyHeatmap:
    """月度收益热力图（year × month）。"""

    years: list[int]
    months: list[int]
    z: np.ndarray
    text: np.ndarray
    month_win_rate: float
    year_win_rate: float


@dataclass(frozen=True)
class SymbolReturns:
    """品种累计收益（已按收益升序）。"""

    symbols: list[str]
    values: np.ndarray


@dataclass(frozen=True)
class PairsDist:
    """交易对分布，按方向分组。"""

    pnl_pct: dict[str, np.ndarray]
    holds: dict[str, np.ndarray]


@dataclass(frozen=True)
class KeyTrade:
    """一笔聚合后的关键开平记录。"""

    symbol: str
    open_dt: str
    close_dt: str
    direction: str
    pnl: float  # 盈亏比例（原始小数）
    hold_bars: int
    count: int  # 聚合的原始开平记录数量


@dataclass(frozen=True)
class KeyTrades:
    """每年最赚/最亏各 N 笔。"""

    best: dict[int, list[KeyTrade]]
    worst: dict[int, list[KeyTrade]]


@dataclass(frozen=True)
class YearlyReturns:
    """逐年绝对收益与超额收益（原始小数），三者按年份升序一一对应。"""

    years: list[int]
    abs_returns: np.ndarray
    alpha_returns: np.ndarray


@dataclass(frozen=True)
class RollingMetrics:
    """滚动窗口指标时间序列；以窗口结束日 ``edt`` 为 x 轴，各序列与之等长。"""

    window: int
    edt: np.ndarray
    sharpe: np.ndarray
    annual_return: np.ndarray  # 原始小数
    annual_vol: np.ndarray  # 原始小数


def _build_curve(daily: np.ndarray) -> Curve:
    cum = np.cumsum(daily)
    drawdown = cum - np.maximum.accumulate(cum)
    return Curve(daily=daily, cum=cum, drawdown=drawdown)


def _aligned(df: pd.DataFrame, date_col: str, val_col: str, index: pd.DatetimeIndex) -> np.ndarray:
    """把 df[val_col] 对齐到统一时间轴 index（缺失补 0），返回 float 数组。"""
    ser = pd.Series(df[val_col].to_numpy(), index=pd.to_datetime(df[date_col]))
    ser = ser[~ser.index.duplicated(keep="first")]
    return ser.reindex(index).fillna(0.0).to_numpy(dtype=float)


def _json_safe(obj: Any) -> Any:
    """递归转 JSON 安全类型：ndarray→list、np 标量→python、日期/时间→ISO 字符串。"""
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, _dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {(_json_safe(k) if not isinstance(k, str) else k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, Curve):
        return {"daily": _json_safe(obj.daily), "cum": _json_safe(obj.cum), "drawdown": _json_safe(obj.drawdown)}
    if isinstance(obj, KeyTrade):
        return {
            "symbol": obj.symbol,
            "open_dt": obj.open_dt,
            "close_dt": obj.close_dt,
            "direction": obj.direction,
            "pnl": obj.pnl,
            "hold_bars": obj.hold_bars,
            "count": obj.count,
        }
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


class BacktestResult:
    """绘图与审核页面的统一输入。轻量字段构造期算好，重字段按需。"""

    def __init__(
        self,
        *,
        wb: WeightBacktest,
        target_vol: float,
        start_date: str,
        end_date: str,
        symbol_count: int,
        weight_type: str,
        yearly_days: int,
        dates: np.ndarray,
        year_starts: np.ndarray,
        curves: dict[str, Curve],
        return_dist: ReturnDist,
        monthly: MonthlyHeatmap,
        symbol_returns: SymbolReturns,
        pairs_dist: PairsDist,
        stats: dict,
        stats_by_side: dict[str, dict],
    ) -> None:
        self._wb = wb  # 私有源引用，供 cached_property 懒算；不进入 to_dict
        self._target_vol = target_vol
        self.start_date = start_date
        self.end_date = end_date
        self.symbol_count = symbol_count
        self.weight_type = weight_type
        self.yearly_days = yearly_days
        self.dates = dates
        self.year_starts = year_starts
        self.curves = curves
        self.return_dist = return_dist
        self.monthly = monthly
        self.symbol_returns = symbol_returns
        self.pairs_dist = pairs_dist
        self.stats = stats
        self.stats_by_side = stats_by_side

    # ------------------------------------------------------------------ build
    @classmethod
    def from_backtest(cls, wb: WeightBacktest, target_vol: float = 0.20) -> BacktestResult:
        dr = wb.daily_return.copy()
        dr["date"] = pd.to_datetime(dr["date"])
        dr = dr.sort_values("date").reset_index(drop=True)
        index = pd.DatetimeIndex(dr["date"])
        dates = index.to_numpy()

        # 各源各解码一次后复用：dailys（多头/空头/品种收益）与 alpha（基准/超额曲线 + 拆分 stats）
        alpha = wb.alpha
        dailys = wb.dailys

        def _side_total(col: str) -> np.ndarray:
            """从单次解码的 dailys 复算某侧 total（口径同 _pivot_daily_return）。"""
            piv = pd.pivot_table(dailys, index="date", columns="symbol", values=col)
            tot = piv.mean(axis=1) if wb.weight_type == "ts" else piv.sum(axis=1)
            ser = pd.Series(tot.to_numpy(), index=pd.to_datetime(tot.index))
            return ser.reindex(index).fillna(0.0).to_numpy(dtype=float)

        curves = {
            "多空": _build_curve(dr["total"].fillna(0.0).to_numpy(dtype=float)),
            "多头": _build_curve(_side_total("long_return")),
            "空头": _build_curve(_side_total("short_return")),
            "基准": _build_curve(_aligned(alpha, "date", "基准", index)),
            "超额": _build_curve(_aligned(alpha, "date", "超额", index)),
        }

        # year_starts：每年最早日期
        ys = dr.groupby(index.year)["date"].min().to_numpy()

        # return_dist（百分比）
        total = dr["total"].to_numpy(dtype=float)
        total = total[~np.isnan(total)] * 100.0
        return_dist = ReturnDist(
            values_pct=total,
            mean_pct=float(np.mean(total)) if total.size else 0.0,
            std_pct=float(np.std(total, ddof=1)) if total.size > 1 else 0.0,
        )

        monthly = cls._build_monthly(dr)
        symbol_returns = cls._build_symbol_returns(dailys)
        pairs_dist = cls._build_pairs_dist(wb)

        stats = wb.stats
        # 基准/超额 stats 复用已解码的 alpha，避免再触发两次 alpha 解码
        stats_by_side = {
            "多头": wb.long_stats,
            "空头": wb.short_stats,
            "基准": wb._compute_stats(alpha, "基准"),
            "超额": wb._compute_stats(alpha, "超额"),
        }

        return cls(
            wb=wb,
            target_vol=target_vol,
            start_date=str(stats.get("开始日期", "")),
            end_date=str(stats.get("结束日期", "")),
            symbol_count=int(stats.get("品种数量", len(wb.symbols))),
            weight_type=wb.weight_type,
            yearly_days=wb.yearly_days,
            dates=dates,
            year_starts=ys,
            curves=curves,
            return_dist=return_dist,
            monthly=monthly,
            symbol_returns=symbol_returns,
            pairs_dist=pairs_dist,
            stats=stats,
            stats_by_side=stats_by_side,
        )

    @staticmethod
    def _build_monthly(dr: pd.DataFrame) -> MonthlyHeatmap:
        d = dr[["date", "total"]].copy()
        d["year"] = d["date"].dt.year
        d["month"] = d["date"].dt.month
        g = d.groupby(["year", "month"])["total"].sum()
        years = sorted(d["year"].unique().tolist())
        months = list(range(1, 13))
        z = np.zeros((len(years), 12), dtype=float)
        year_idx = {y: i for i, y in enumerate(years)}
        for (y, m), v in g.items():
            z[year_idx[y], m - 1] = v
        text = np.array([[f"{v * 100:.2f}%" for v in row] for row in z], dtype=object)
        month_win_rate = float((g.to_numpy() > 0).mean()) if len(g) else 0.0
        yearly = d.groupby("year")["total"].sum()
        year_win_rate = float((yearly.to_numpy() > 0).mean()) if len(yearly) else 0.0
        return MonthlyHeatmap(
            years=[int(y) for y in years],
            months=months,
            z=z,
            text=text,
            month_win_rate=month_win_rate,
            year_win_rate=year_win_rate,
        )

    @staticmethod
    def _build_symbol_returns(dailys: pd.DataFrame) -> SymbolReturns:
        if dailys.empty or "symbol" not in dailys.columns or "return" not in dailys.columns:
            return SymbolReturns(symbols=[], values=np.array([], dtype=float))
        sr = dailys.groupby("symbol")["return"].sum().sort_values(ascending=True)
        return SymbolReturns(symbols=[str(s) for s in sr.index.tolist()], values=sr.to_numpy(dtype=float))

    @staticmethod
    def _build_pairs_dist(wb: WeightBacktest) -> PairsDist:
        agg = wb.aggregated_pairs
        pnl_pct: dict[str, np.ndarray] = {}
        holds: dict[str, np.ndarray] = {}
        if agg.empty or "交易方向" not in agg.columns:
            return PairsDist(pnl_pct=pnl_pct, holds=holds)
        for direction, sub in agg.groupby("交易方向"):
            key = str(direction)
            # 盈亏比例单位 BP → 百分比
            pnl_pct[key] = np.round(sub["盈亏比例"].to_numpy(dtype=float) / 100.0, 4)
            holds[key] = sub["持仓K线数"].to_numpy(dtype=int)
        return PairsDist(pnl_pct=pnl_pct, holds=holds)

    # -------------------------------------------------------- cached (按需)
    @cached_property
    def curves_voladj(self) -> dict[str, Curve]:
        """波动率归一后的同名曲线；scale = target_vol / (daily.std · √yearly_days)。

        「超额」特殊处理：定义为 ``norm(多头) − norm(基准)``（多头、基准先各自归一化、
        再逐日相减），而非对原始超额(策略−基准)整体归一化。这样归一超额曲线恰好等于
        图上归一化多头与归一化基准两条线之差，口径自洽；其年化波动率不再等于 target_vol。
        """
        out: dict[str, Curve] = {}
        sqrt_yd = float(np.sqrt(self.yearly_days))
        for key, c in self.curves.items():
            if key == "超额":
                continue  # 由 norm(多头) − norm(基准) 派生，循环结束后单独构造
            std = float(np.std(c.daily, ddof=1)) if c.daily.size > 1 else 0.0
            annual_vol = std * sqrt_yd
            scale = (self._target_vol / annual_vol) if annual_vol > 0 else 1.0
            out[key] = _build_curve(c.daily * scale)
        if "多头" in out and "基准" in out:
            out["超额"] = _build_curve(out["多头"].daily - out["基准"].daily)
        return out

    @cached_property
    def drawdowns(self) -> list[dict]:
        """top_drawdowns 明细（基于多空日收益序列）。"""
        total = self.curves["多空"].daily
        series = pd.Series(total, index=pd.DatetimeIndex(self.dates))
        df = top_drawdowns(series, top=10)
        # top_drawdowns 的日期列是 datetime.date、数值列是 numpy 标量；统一转 JSON 安全类型，
        # 使 result.drawdowns 本身即可 json.dumps（不依赖 to_dict 兜底）。
        return [_json_safe(rec) for rec in df.to_dict("records")]

    @cached_property
    def key_trades(self) -> KeyTrades:
        kt = self._wb.key_trades(3)
        best: dict[int, list[KeyTrade]] = {}
        worst: dict[int, list[KeyTrade]] = {}
        if kt.empty:
            return KeyTrades(best=best, worst=worst)
        for rec in kt.to_dict("records"):
            year = int(rec["year"])
            trade = KeyTrade(
                symbol=str(rec["symbol"]),
                open_dt=pd.Timestamp(rec["开仓时间"]).isoformat(),
                close_dt=pd.Timestamp(rec["平仓时间"]).isoformat(),
                direction=str(rec["交易方向"]),
                pnl=float(rec["盈亏比例"]) / 10000.0,  # BP → 原始小数
                hold_bars=int(rec["持仓K线数"]),
                count=int(rec["count"]),
            )
            bucket = best if rec["kind"] == "best" else worst
            bucket.setdefault(year, []).append(trade)
        return KeyTrades(best=best, worst=worst)

    @cached_property
    def verdict(self) -> dict:
        """history 模式判定（逐年）。yearly_returns 复用其 yearly_metrics。"""
        return self._wb.is_good_strategy(mode="history")

    @cached_property
    def verdict_recent(self) -> dict:
        """recent 模式判定（尾部 recent_days 天）。"""
        return self._wb.is_good_strategy(mode="recent")

    @cached_property
    def yearly_returns(self) -> YearlyReturns:
        """逐年绝对/超额收益，复用 verdict 的 yearly_metrics（不额外计算）。"""
        ym = sorted(self.verdict.get("yearly_metrics") or [], key=lambda m: m["year"])
        return YearlyReturns(
            years=[int(m["year"]) for m in ym],
            abs_returns=np.array([float(m["abs_return"]) for m in ym], dtype=float),
            alpha_returns=np.array([float(m["alpha_return"]) for m in ym], dtype=float),
        )

    @cached_property
    def rolling(self) -> RollingMetrics:
        """多空日收益的滚动窗口指标（夏普/年化/年化波动率），x 轴为窗口结束日。"""
        from wbt.utils.rolling_daily_performance import rolling_daily_performance

        window = 252
        daily = self.curves["多空"].daily
        empty = np.array([], dtype=float)
        if daily.size == 0:
            return RollingMetrics(window=window, edt=empty, sharpe=empty, annual_return=empty, annual_vol=empty)
        df = pd.DataFrame({"dt": self.dates, "total": daily})
        roll = rolling_daily_performance(df, "total", window=window, min_periods=100, yearly_days=self.yearly_days)
        if roll.empty:
            return RollingMetrics(window=window, edt=empty, sharpe=empty, annual_return=empty, annual_vol=empty)
        return RollingMetrics(
            window=window,
            edt=pd.to_datetime(roll["edt"]).to_numpy(),
            sharpe=roll["夏普"].to_numpy(dtype=float),
            annual_return=roll["年化"].to_numpy(dtype=float),
            annual_vol=roll["年化波动率"].to_numpy(dtype=float),
        )

    @cached_property
    def segment_comparison(self) -> dict[str, dict]:
        """近 1 年 vs 全样本的关键指标对比（多空口径），值为 stats dict。"""
        out: dict[str, dict] = {"全样本": self.stats}
        n = len(self.dates)
        if n > 0:
            sdt = pd.Timestamp(self.dates[max(0, n - self.yearly_days)])
            # 区间过短等异常时降级为仅全样本。segment_stats 的错误在 Rust FFI 边界统一
            # 转成 PyException（Python Exception 基类），无法收窄到更具体的类型，故这里
            # 捕获 Exception 但留一行 debug 痕迹，避免真实 bug 被静默成"近1年面板消失"。
            try:
                out["近1年"] = self._wb.segment_stats(sdt=sdt, kind="多空")
            except Exception:
                logger.debug("segment_stats 近1年区间计算失败，降级为仅全样本", exc_info=True)
        return out

    # ---------------------------------------------------------------- to_dict
    def to_dict(self, *, full: bool = False) -> dict:
        out: dict[str, Any] = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbol_count": self.symbol_count,
            "weight_type": self.weight_type,
            "yearly_days": self.yearly_days,
            "dates": [pd.Timestamp(d).isoformat() for d in self.dates],
            "year_starts": [pd.Timestamp(d).isoformat() for d in self.year_starts],
            "curves": {k: _json_safe(c) for k, c in self.curves.items()},
            "return_dist": {
                "values_pct": _json_safe(self.return_dist.values_pct),
                "mean_pct": self.return_dist.mean_pct,
                "std_pct": self.return_dist.std_pct,
            },
            "monthly": {
                "years": self.monthly.years,
                "months": self.monthly.months,
                "z": _json_safe(self.monthly.z),
                "text": _json_safe(self.monthly.text),
                "month_win_rate": self.monthly.month_win_rate,
                "year_win_rate": self.monthly.year_win_rate,
            },
            "symbol_returns": {
                "symbols": self.symbol_returns.symbols,
                "values": _json_safe(self.symbol_returns.values),
            },
            "pairs_dist": {
                "pnl_pct": {k: _json_safe(v) for k, v in self.pairs_dist.pnl_pct.items()},
                "holds": {k: _json_safe(v) for k, v in self.pairs_dist.holds.items()},
            },
            "stats": _json_safe(self.stats),
            "stats_by_side": _json_safe(self.stats_by_side),
        }
        if full:
            out["curves_voladj"] = {k: _json_safe(c) for k, c in self.curves_voladj.items()}
            out["drawdowns"] = _json_safe(self.drawdowns)
            out["key_trades"] = {
                "best": {str(y): _json_safe(rows) for y, rows in self.key_trades.best.items()},
                "worst": {str(y): _json_safe(rows) for y, rows in self.key_trades.worst.items()},
            }
            out["verdict"] = _json_safe(self.verdict)
            out["yearly_returns"] = {
                "years": self.yearly_returns.years,
                "abs_returns": _json_safe(self.yearly_returns.abs_returns),
                "alpha_returns": _json_safe(self.yearly_returns.alpha_returns),
            }
            out["rolling"] = {
                "window": self.rolling.window,
                "edt": [pd.Timestamp(d).isoformat() for d in self.rolling.edt],
                "sharpe": _json_safe(self.rolling.sharpe),
                "annual_return": _json_safe(self.rolling.annual_return),
                "annual_vol": _json_safe(self.rolling.annual_vol),
            }
            out["segment_comparison"] = _json_safe(self.segment_comparison)
        return out
