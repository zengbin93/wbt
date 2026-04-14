"""逐行对比 wbt.yearly_return 与 vista.calculate_yearly_returns 的输出。

前置条件：
- 真实持仓权重数据（默认 /Volumes/jun/全A日线测试_20170101_20250429.feather）
- vista 源码可读（默认 /Users/0xjun/Documents/cursorPro/vista）

用法：
    uv run python python/scripts/compare_yearly_return_with_vista.py \\
        --data-path /path/to/data.feather \\
        --min-days 120
"""

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

DEFAULT_DATA_PATH = Path("/Volumes/jun/全A日线测试_20170101_20250429.feather")
DEFAULT_VISTA_ROOT = Path("/Users/0xjun/Documents/cursorPro/vista")
# bit-exact 一致所允许的 f64 累计误差下限（两侧都是 f64 从左往右复利，理论上应完全相等）
TOLERANCE = 1e-12


@dataclass(frozen=True)
class ComparisonResult:
    wbt_rows: int
    vista_rows: int
    matched: bool
    max_abs_diff: float


class _MockWb:
    """WeightBacktest Protocol：只暴露 daily_return 属性给 vista 侧使用"""

    def __init__(self, daily_return: pd.DataFrame) -> None:
        self.daily_return = daily_return


def _install_loguru_stub() -> None:
    """vista.utils.yearly_return 依赖 loguru，避免强制装依赖，直接 stub。"""
    if "loguru" in sys.modules:
        return
    fake_logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    fake_loguru = types.ModuleType("loguru")
    fake_loguru.logger = fake_logger  # type: ignore[attr-defined]
    sys.modules["loguru"] = fake_loguru


def load_vista_calculate_yearly_returns(
    vista_root: Path = DEFAULT_VISTA_ROOT,
) -> Callable[..., pd.DataFrame]:
    """导入 vista.utils.yearly_return.calculate_yearly_returns（不污染全局 sys.path）"""
    _install_loguru_stub()
    module_path = vista_root / "vista" / "utils" / "yearly_return.py"
    if not module_path.exists():
        raise FileNotFoundError(f"vista yearly_return.py not found: {module_path}")
    spec = importlib.util.spec_from_file_location("_vista_yearly_return", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_yearly_returns  # type: ignore[no-any-return]


def run_comparison(
    dfw: pd.DataFrame,
    min_days: int = 120,
    digits: int = 2,
    fee_rate: float = 0.0002,
    n_jobs: int = 4,
    weight_type: str = "ts",
    vista_root: Path = DEFAULT_VISTA_ROOT,
) -> ComparisonResult:
    """跑 wbt 回测，然后同一 daily_return 宽表喂给 vista，逐行对比两侧的 yearly_return 输出。"""
    from wbt import WeightBacktest

    wb = WeightBacktest(
        dfw.copy(),
        digits=digits,
        fee_rate=fee_rate,
        n_jobs=n_jobs,
        weight_type=weight_type,
    )

    wbt_yr = wb.yearly_return(min_days=min_days).reset_index(drop=True).copy()

    vista_fn = load_vista_calculate_yearly_returns(vista_root)
    mock = _MockWb(wb.daily_return.copy())
    vista_yr = vista_fn(mock, min_days=min_days).reset_index(drop=True).copy()

    # year 列：wbt 为 int32（Rust 来），vista 为 Python int（pandas int64）→ 统一 int64
    wbt_yr["year"] = wbt_yr["year"].astype("int64")
    vista_yr["year"] = vista_yr["year"].astype("int64")

    if wbt_yr.shape != vista_yr.shape:
        print(f"shape mismatch: wbt={wbt_yr.shape}, vista={vista_yr.shape}", file=sys.stderr)
        _dump_diff_head(wbt_yr, vista_yr)
        return ComparisonResult(wbt_yr.shape[0], vista_yr.shape[0], False, float("inf"))

    try:
        pd.testing.assert_frame_equal(wbt_yr[["year", "symbol"]], vista_yr[["year", "symbol"]])
    except AssertionError as err:
        print(f"year/symbol columns mismatch:\n{err}", file=sys.stderr)
        _dump_diff_head(wbt_yr, vista_yr)
        return ComparisonResult(wbt_yr.shape[0], vista_yr.shape[0], False, float("inf"))

    abs_diff = (wbt_yr["return"] - vista_yr["return"]).abs()
    max_abs_diff = float(abs_diff.max()) if len(abs_diff) else 0.0
    matched = max_abs_diff <= TOLERANCE

    if not matched:
        print(f"return values differ: max_abs_diff={max_abs_diff:.3e}", file=sys.stderr)
        diff = wbt_yr.copy()
        diff["vista_return"] = vista_yr["return"]
        diff["abs_diff"] = abs_diff
        print("Top 10 by abs_diff:", file=sys.stderr)
        print(diff.sort_values("abs_diff", ascending=False).head(10).to_string(), file=sys.stderr)

    return ComparisonResult(wbt_yr.shape[0], vista_yr.shape[0], matched, max_abs_diff)


def _dump_diff_head(wbt_yr: pd.DataFrame, vista_yr: pd.DataFrame) -> None:
    print("wbt head:", file=sys.stderr)
    print(wbt_yr.head(10).to_string(), file=sys.stderr)
    print("vista head:", file=sys.stderr)
    print(vista_yr.head(10).to_string(), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vista-root", type=Path, default=DEFAULT_VISTA_ROOT)
    parser.add_argument("--min-days", type=int, default=120)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--fee-rate", type=float, default=0.0002)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--weight-type", choices=["ts", "cs"], default="ts")
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"data_path not found: {args.data_path}", file=sys.stderr)
        return 2
    if not args.vista_root.exists():
        print(f"vista_root not found: {args.vista_root}", file=sys.stderr)
        return 2

    print(f"Loading data from {args.data_path}")
    dfw = pd.read_feather(args.data_path)
    print(f"Data: {dfw.shape[0]:,} rows, {dfw['symbol'].nunique()} symbols")
    print(
        f"Params: digits={args.digits}, fee_rate={args.fee_rate}, n_jobs={args.n_jobs},"
        f" weight_type={args.weight_type}, min_days={args.min_days}"
    )

    result = run_comparison(
        dfw,
        min_days=args.min_days,
        digits=args.digits,
        fee_rate=args.fee_rate,
        n_jobs=args.n_jobs,
        weight_type=args.weight_type,
        vista_root=args.vista_root,
    )

    print("\n" + "=" * 80)
    print(f"wbt rows:              {result.wbt_rows}")
    print(f"vista rows:            {result.vista_rows}")
    print(f"max_abs_diff(return):  {result.max_abs_diff:.3e}")
    print(f"tolerance:             {TOLERANCE:.3e}")
    print(f"matched:               {result.matched}")
    print("=" * 80)

    return 0 if result.matched else 1


if __name__ == "__main__":
    raise SystemExit(main())


# quiet lint on unused Any
_ = Any
