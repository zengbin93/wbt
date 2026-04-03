from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = Path("/Volumes/jun/全A日线测试_20170101_20250429.feather")
DEFAULT_CZSC_ROOT = Path("/Users/0xjun/Documents/cursorPro/czsc")


@dataclass(frozen=True)
class OperationSpec:
    name: str
    kind: str
    runner: Callable[[Any], Any]
    sort_cols: tuple[str, ...] = ()


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.errors: list[tuple[str, str]] = []

    def ok(self, name: str) -> None:
        self.passed += 1

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        self.errors.append((name, detail))
        print(f"FAIL {name}: {detail}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print("\n" + "=" * 80)
        print(f"TOTAL: {total} checks, {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("\nFailed checks:")
            for name, detail in self.errors:
                print(f"- {name}: {detail}")
        print("=" * 80)
        return self.failed == 0


def build_operation_specs(sample_symbol: str) -> list[OperationSpec]:
    return [
        OperationSpec("stats", "dict", lambda wb: wb.stats),
        OperationSpec("daily_return", "df", lambda wb: wb.daily_return, ("date",)),
        OperationSpec("dailys", "df", lambda wb: wb.dailys, ("symbol", "date")),
        OperationSpec("alpha", "df", lambda wb: wb.alpha, ("date",)),
        OperationSpec("alpha_stats", "dict", lambda wb: wb.alpha_stats),
        OperationSpec("bench_stats", "dict", lambda wb: wb.bench_stats),
        OperationSpec("long_daily_return", "df", lambda wb: wb.long_daily_return, ("date",)),
        OperationSpec("short_daily_return", "df", lambda wb: wb.short_daily_return, ("date",)),
        OperationSpec("long_stats", "dict", lambda wb: wb.long_stats),
        OperationSpec("short_stats", "dict", lambda wb: wb.short_stats),
        OperationSpec("get_symbol_daily", "df", lambda wb: wb.get_symbol_daily(sample_symbol), ("date",)),
        OperationSpec("get_symbol_pairs", "pairs_df", lambda wb: wb.get_symbol_pairs(sample_symbol), ("开仓时间",)),
        OperationSpec("daily_performance", "dict", lambda wb: wb.__class__.__module__),
    ]


def install_czsc_stubs(czsc_root: Path) -> None:
    deprecated_mod = types.ModuleType("deprecated")
    deprecated_mod.deprecated = lambda *args, **kwargs: lambda func: func
    sys.modules["deprecated"] = deprecated_mod

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable=None, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_mod

    logger = types.SimpleNamespace(info=lambda *a, **k: None, add=lambda *a, **k: None)
    loguru_mod = types.ModuleType("loguru")
    loguru_mod.logger = logger
    sys.modules["loguru"] = loguru_mod

    plotly_mod = types.ModuleType("plotly")
    plotly_express_mod = types.ModuleType("plotly.express")
    plotly_mod.express = plotly_express_mod
    sys.modules["plotly"] = plotly_mod
    sys.modules["plotly.express"] = plotly_express_mod

    io_module = types.ModuleType("czsc.utils.io")
    io_module.save_json = lambda *args, **kwargs: None
    sys.modules["czsc.utils.io"] = io_module

    stats_spec = importlib.util.spec_from_file_location("czsc.utils.stats", czsc_root / "czsc" / "utils" / "stats.py")
    assert stats_spec and stats_spec.loader
    stats_module = importlib.util.module_from_spec(stats_spec)
    sys.modules["czsc.utils.stats"] = stats_module
    stats_spec.loader.exec_module(stats_module)

    czsc_utils_mod = types.ModuleType("czsc.utils")
    czsc_utils_mod.io = io_module
    czsc_utils_mod.stats = stats_module
    sys.modules["czsc.utils"] = czsc_utils_mod

    czsc_mod = types.ModuleType("czsc")
    czsc_mod.daily_performance = stats_module.daily_performance
    sys.modules["czsc"] = czsc_mod


def load_czsc_weight_backtest(czsc_root: Path) -> tuple[type[Any], Callable[..., dict[str, Any]]]:
    install_czsc_stubs(czsc_root)
    script_path = czsc_root / "czsc" / "py" / "weight_backtest.py"
    spec = importlib.util.spec_from_file_location("czsc.py.weight_backtest", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WeightBacktest, sys.modules["czsc.utils.stats"].daily_performance


def load_wbt_backtest() -> tuple[type[Any], Callable[..., dict[str, Any]]]:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from wbt import daily_performance as wbt_daily_performance
    from wbt.backtest import WeightBacktest as WbtWeightBacktest

    return WbtWeightBacktest, wbt_daily_performance


def compare_scalar(results: Results, name: str, wbt_val: Any, czsc_val: Any, tol: float) -> None:
    if wbt_val is None and czsc_val is None:
        results.ok(name)
        return
    if wbt_val is None or czsc_val is None:
        results.fail(name, f"wbt={wbt_val} vs czsc={czsc_val}")
        return
    if isinstance(wbt_val, str) and isinstance(czsc_val, str):
        if normalize_date_string(wbt_val) == normalize_date_string(czsc_val):
            results.ok(name)
        else:
            results.fail(name, f"wbt='{wbt_val}' vs czsc='{czsc_val}'")
        return
    try:
        if abs(float(wbt_val) - float(czsc_val)) <= tol:
            results.ok(name)
        else:
            results.fail(name, f"wbt={wbt_val} vs czsc={czsc_val}")
    except Exception:
        if normalize_date_string(str(wbt_val)) == normalize_date_string(str(czsc_val)):
            results.ok(name)
        else:
            results.fail(name, f"wbt={wbt_val} vs czsc={czsc_val}")


def compare_dict(
    results: Results, prefix: str, wbt_dict: dict[str, Any], czsc_dict: dict[str, Any], tol: float
) -> None:
    keys = set(wbt_dict) | set(czsc_dict)
    for key in sorted(keys):
        if key not in wbt_dict:
            results.fail(f"{prefix}[{key}]", "missing in wbt")
        elif key not in czsc_dict:
            results.fail(f"{prefix}[{key}]", "missing in czsc")
        else:
            compare_scalar(results, f"{prefix}[{key}]", wbt_dict[key], czsc_dict[key], tol)


def compare_list(results: Results, name: str, wbt_list: list[Any], czsc_list: list[Any]) -> None:
    if wbt_list == czsc_list:
        results.ok(name)
    else:
        results.fail(name, f"wbt={wbt_list} vs czsc={czsc_list}")


def normalize_date_string(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def compare_df(
    results: Results,
    prefix: str,
    wbt_df: pd.DataFrame,
    czsc_df: pd.DataFrame,
    tol: float,
    sort_cols: tuple[str, ...] = (),
) -> None:
    if wbt_df.shape != czsc_df.shape:
        results.fail(f"{prefix}.shape", f"wbt={wbt_df.shape} vs czsc={czsc_df.shape}")
        return
    results.ok(f"{prefix}.shape")

    w_cols = set(wbt_df.columns)
    c_cols = set(czsc_df.columns)
    if w_cols != c_cols:
        results.fail(f"{prefix}.columns", f"wbt={sorted(w_cols)} vs czsc={sorted(c_cols)}")
        return
    results.ok(f"{prefix}.columns")

    ordered_cols = sorted(w_cols)
    valid_sort = [c for c in sort_cols if c in w_cols]
    if valid_sort:
        wbt_df = wbt_df.sort_values(valid_sort).reset_index(drop=True)
        czsc_df = czsc_df.sort_values(valid_sort).reset_index(drop=True)

    for col in ordered_cols:
        w = wbt_df[col]
        c = czsc_df[col]
        name = f"{prefix}[{col}]"
        if pd.api.types.is_numeric_dtype(w) and pd.api.types.is_numeric_dtype(c):
            diff = (w.fillna(0) - c.fillna(0)).abs().max()
            if pd.isna(diff) or diff <= tol:
                results.ok(name)
            else:
                results.fail(name, f"max_diff={diff:.2e}")
        else:
            w_str = w.astype(str).map(normalize_date_string)
            c_str = c.astype(str).map(normalize_date_string)
            if w_str.equals(c_str):
                results.ok(name)
            else:
                results.fail(name, "stringified values differ")


def normalize_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    if "持仓数量" in df.columns:
        rows: list[dict[str, Any]] = []
        rename_map = {"symbol": "标的代码"}
        for _, row in df.iterrows():
            count = int(row["持仓数量"])
            base = row.drop(labels=["持仓数量"]).rename(index=rename_map).to_dict()
            for _ in range(count):
                rows.append(dict(base))
        out = pd.DataFrame(rows)
        ordered = [
            "标的代码",
            "交易方向",
            "开仓时间",
            "平仓时间",
            "开仓价格",
            "平仓价格",
            "持仓K线数",
            "事件序列",
            "持仓天数",
            "盈亏比例",
        ]
        return out[ordered].copy()
    return df.copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare wbt against czsc Python weight_backtest on a real dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--czsc-root", type=Path, default=DEFAULT_CZSC_ROOT)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--weight-type", choices=["ts", "cs"], default="ts")
    parser.add_argument("--yearly-days", type=int, default=252)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    dfw = pd.read_feather(args.data_path)
    print(f"Data: {dfw.shape[0]:,} rows, {dfw['symbol'].nunique()} symbols")

    CzscWeightBacktest, czsc_daily_performance = load_czsc_weight_backtest(args.czsc_root)
    WbtWeightBacktest, wbt_daily_performance = load_wbt_backtest()

    if args.n_jobs != 1:
        print("Note: forcing n_jobs=1 for czsc source comparison to avoid subprocess import issues.")

    kwargs = {
        "digits": args.digits,
        "fee_rate": args.fee_rate,
        "n_jobs": 1,
        "weight_type": args.weight_type,
        "yearly_days": args.yearly_days,
    }
    print(f"Params: {kwargs}")

    czsc = CzscWeightBacktest(dfw.copy(), **kwargs)
    wbt = WbtWeightBacktest(dfw.copy(), **kwargs)
    sample_symbol = str(dfw["symbol"].iloc[0])
    print(f"Sample symbol: {sample_symbol}")

    results = Results()
    for op in build_operation_specs(sample_symbol):
        if op.name == "daily_performance":
            returns = wbt.daily_return["total"].to_numpy()
            compare_dict(
                results,
                op.name,
                wbt_daily_performance(returns, yearly_days=args.yearly_days),
                czsc_daily_performance(returns, yearly_days=args.yearly_days),
                args.tol,
            )
            continue

        print(f"Checking {op.name}")
        try:
            wbt_val = op.runner(wbt)
        except Exception as e:
            results.fail(op.name, f"wbt exception: {type(e).__name__}: {e}")
            continue
        try:
            czsc_val = op.runner(czsc)
        except Exception as e:
            results.fail(op.name, f"czsc exception: {type(e).__name__}: {e}")
            continue
        if op.kind == "df":
            compare_df(results, op.name, wbt_val, czsc_val, args.tol, op.sort_cols)
        elif op.kind == "pairs_df":
            compare_df(
                results, op.name, normalize_pairs_df(wbt_val), normalize_pairs_df(czsc_val), args.tol, op.sort_cols
            )
        elif op.kind == "dict":
            compare_dict(results, op.name, wbt_val, czsc_val, args.tol)
        elif op.kind == "list":
            compare_list(results, op.name, wbt_val, czsc_val)
        else:
            raise ValueError(f"Unknown operation kind: {op.kind}")

    return 0 if results.summary() else 1


if __name__ == "__main__":
    raise SystemExit(main())
