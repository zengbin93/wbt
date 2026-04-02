from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = Path("/Volumes/jun/全A日线测试_20170101_20250429.feather")
DEFAULT_ORIG_PYTHON = Path("/Users/0xjun/Documents/cursorPro/rs_czsc/python")


@dataclass(frozen=True)
class OperationSpec:
    name: str
    runner: Callable[[Any], Any]


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    wbt_avg: float
    orig_avg: float
    ratio: float
    wbt_summary: str
    orig_summary: str


def build_operation_specs(sample_symbol: str) -> list[OperationSpec]:
    return [
        OperationSpec("stats", lambda wb: wb.stats),
        OperationSpec("daily_return", lambda wb: wb.daily_return),
        OperationSpec("dailys", lambda wb: wb.dailys),
        OperationSpec("alpha", lambda wb: wb.alpha),
        OperationSpec("pairs", lambda wb: wb.pairs),
        OperationSpec("alpha_stats", lambda wb: wb.alpha_stats),
        OperationSpec("bench_stats", lambda wb: wb.bench_stats),
        OperationSpec("long_daily_return", lambda wb: wb.long_daily_return),
        OperationSpec("short_daily_return", lambda wb: wb.short_daily_return),
        OperationSpec("long_stats", lambda wb: wb.long_stats),
        OperationSpec("short_stats", lambda wb: wb.short_stats),
        OperationSpec("symbol_dict", lambda wb: wb.symbol_dict),
        OperationSpec("get_symbol_daily", lambda wb: wb.get_symbol_daily(sample_symbol)),
        OperationSpec("get_symbol_pairs", lambda wb: wb.get_symbol_pairs(sample_symbol)),
        OperationSpec("get_top_symbols_profit", lambda wb: wb.get_top_symbols(n=5, kind="profit")),
        OperationSpec("get_top_symbols_loss", lambda wb: wb.get_top_symbols(n=5, kind="loss")),
    ]


def summarize_value(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return f"DataFrame{value.shape}"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def load_backtest_classes() -> tuple[type[Any], type[Any], Callable[..., dict[str, Any]], Callable[..., dict[str, Any]]]:
    sys.path.insert(0, str(DEFAULT_ORIG_PYTHON))
    from rs_czsc._trader.weight_backtest import WeightBacktest as OrigWeightBacktest
    from rs_czsc._rs_czsc import daily_performance as orig_daily_performance

    sys.path.insert(0, str(REPO_ROOT / "python"))
    from wbt.backtest import WeightBacktest as WbtWeightBacktest
    from wbt._wbt import daily_performance as wbt_daily_performance

    return WbtWeightBacktest, OrigWeightBacktest, wbt_daily_performance, orig_daily_performance


def measure_init(factory: Callable[[], Any], repeat: int) -> tuple[float, str]:
    times: list[float] = []
    summary = ""
    for _ in range(repeat):
        start = time.perf_counter()
        obj = factory()
        times.append(time.perf_counter() - start)
        summary = type(obj).__name__
    return statistics.mean(times), summary


def measure_operation(factory: Callable[[], Any], op: OperationSpec, repeat: int) -> tuple[float, str]:
    times: list[float] = []
    summary = ""
    for _ in range(repeat):
        obj = factory()
        start = time.perf_counter()
        value = op.runner(obj)
        times.append(time.perf_counter() - start)
        summary = summarize_value(value)
    return statistics.mean(times), summary


def measure_standalone_daily_performance(
    wb_factory: Callable[[], Any],
    fn: Callable[..., dict[str, Any]],
    repeat: int,
    yearly_days: int,
) -> tuple[float, str]:
    times: list[float] = []
    summary = ""
    for _ in range(repeat):
        wb = wb_factory()
        returns = wb.daily_return["total"].to_numpy()
        start = time.perf_counter()
        value = fn(returns, yearly_days=yearly_days)
        times.append(time.perf_counter() - start)
        summary = summarize_value(value)
    return statistics.mean(times), summary


def make_row(
    name: str,
    wbt_avg: float,
    orig_avg: float,
    wbt_summary: str,
    orig_summary: str,
) -> BenchmarkRow:
    ratio = wbt_avg / orig_avg if orig_avg > 0 else float("inf")
    return BenchmarkRow(name, wbt_avg, orig_avg, ratio, wbt_summary, orig_summary)


def print_table(rows: list[BenchmarkRow]) -> None:
    print("\n" + "=" * 110)
    print(f"{'operation':<28} {'wbt_avg(s)':>12} {'orig_avg(s)':>12} {'ratio':>10} {'wbt':>20} {'orig':>20}")
    print("-" * 110)
    for row in rows:
        print(
            f"{row.name:<28} "
            f"{row.wbt_avg:>12.4f} "
            f"{row.orig_avg:>12.4f} "
            f"{row.ratio:>10.2f}x "
            f"{row.wbt_summary:>20} "
            f"{row.orig_summary:>20}"
        )
    print("=" * 110)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare wbt vs rs_czsc performance on a real dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--weight-type", choices=["ts", "cs"], default="ts")
    parser.add_argument("--yearly-days", type=int, default=252)
    args = parser.parse_args()

    WbtWeightBacktest, OrigWeightBacktest, wbt_daily_performance, orig_daily_performance = load_backtest_classes()

    print(f"Loading data from {args.data_path}")
    dfw = pd.read_feather(args.data_path)
    print(f"Data: {dfw.shape[0]:,} rows, {dfw['symbol'].nunique()} symbols")
    print(
        "Params:"
        f" digits={args.digits}, fee_rate={args.fee_rate},"
        f" n_jobs={args.n_jobs}, weight_type={args.weight_type}, yearly_days={args.yearly_days},"
        f" repeat={args.repeat}"
    )

    sample_symbol = str(dfw["symbol"].iloc[0])
    print(f"Sample symbol for symbol-specific methods: {sample_symbol}")

    def wbt_factory() -> Any:
        return WbtWeightBacktest(
            dfw.copy(),
            digits=args.digits,
            fee_rate=args.fee_rate,
            n_jobs=args.n_jobs,
            weight_type=args.weight_type,
            yearly_days=args.yearly_days,
        )

    def orig_factory() -> Any:
        return OrigWeightBacktest(
            dfw.copy(),
            digits=args.digits,
            fee_rate=args.fee_rate,
            n_jobs=args.n_jobs,
            weight_type=args.weight_type,
            yearly_days=args.yearly_days,
        )

    rows: list[BenchmarkRow] = []

    wbt_avg, wbt_summary = measure_init(wbt_factory, args.repeat)
    orig_avg, orig_summary = measure_init(orig_factory, args.repeat)
    rows.append(make_row("init", wbt_avg, orig_avg, wbt_summary, orig_summary))

    for op in build_operation_specs(sample_symbol):
        wbt_avg, wbt_summary = measure_operation(wbt_factory, op, args.repeat)
        orig_avg, orig_summary = measure_operation(orig_factory, op, args.repeat)
        rows.append(make_row(op.name, wbt_avg, orig_avg, wbt_summary, orig_summary))

    wbt_avg, wbt_summary = measure_standalone_daily_performance(
        wbt_factory, wbt_daily_performance, args.repeat, args.yearly_days
    )
    orig_avg, orig_summary = measure_standalone_daily_performance(
        orig_factory, orig_daily_performance, args.repeat, args.yearly_days
    )
    rows.append(make_row("daily_performance", wbt_avg, orig_avg, wbt_summary, orig_summary))

    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
