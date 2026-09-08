"""用独立进程测量合成数据的 pandas 输入、IPC 往返和宽表物化成本。

在 python/ 下构建 release 扩展后运行；每个 case/repeat 使用全新子进程。
结果是诊断基线，不代表真实策略负载，也不用于 CI 性能阈值。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

CASES = (
    "pandas_ipc",
    "pandas_init",
    "arrow_init",
    "stats",
    "daily_return_first",
    "daily_return_repeat",
    "ipc_encode_cached",
    "ipc_decode",
)


def peak_rss_mib() -> float:
    import resource

    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def run_case(args: argparse.Namespace) -> dict:
    import numpy as np
    import pandas as pd

    from wbt import WeightBacktest
    from wbt._df_convert import arrow_bytes_to_pd_df, pandas_to_arrow_bytes
    from wbt._wbt import PyWeightBacktest

    # 每个 symbol 每 stride 天参与一天，每天两根 bar，保证首日也产生日收益。
    # symbol 错开参与日，使稀疏数据仍覆盖完整日期轴（symbols >= stride）。
    offsets = [np.arange(i % args.stride, args.days, args.stride) for i in range(args.symbols)]
    sid = np.repeat(np.arange(args.symbols), [len(values) * 2 for values in offsets])
    day = np.repeat(np.concatenate(offsets), 2)
    bar = np.tile([0, 1], len(day) // 2)
    del offsets
    df = pd.DataFrame(
        {
            "dt": pd.Timestamp("2020-01-01")
            + pd.to_timedelta(day, unit="D")
            + pd.to_timedelta(570 + bar * 330, unit="m"),
            "symbol": np.array([f"S{i:05d}" for i in range(args.symbols)])[sid],
            "weight": np.where((day // 20 + sid) % 2 == 0, 0.5, -0.5),
            "price": 100 + day * 0.01 + np.sin(day + sid + bar) * 0.1,
        }
    )
    del sid, day, bar
    kwargs = {"digits": 2, "fee_rate": 0.0002, "n_jobs": 1, "weight_type": "ts", "yearly_days": 252}

    if args.case == "pandas_ipc":
        operation = partial(pandas_to_arrow_bytes, df)
    elif args.case == "pandas_init":
        operation = partial(WeightBacktest, df, **kwargs)
    elif args.case == "arrow_init":
        payload = pandas_to_arrow_bytes(df)
        operation = partial(PyWeightBacktest.from_arrow, payload, **kwargs)
    else:
        wb = WeightBacktest(df, **kwargs)
        if args.case == "stats":
            operation = partial(getattr, wb, "stats")
        elif args.case == "daily_return_first":
            operation = partial(getattr, wb, "daily_return")
        elif args.case == "daily_return_repeat":
            _ = wb.daily_return
            del _
            operation = partial(getattr, wb, "daily_return")
        elif args.case == "ipc_encode_cached":
            _ = wb._inner.daily_return()
            del _
            operation = wb._inner.daily_return
        else:
            payload = wb._inner.daily_return()
            operation = partial(arrow_bytes_to_pd_df, payload)

    peak_before = peak_rss_mib()
    start = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - start
    peak_after = peak_rss_mib()
    if isinstance(result, pd.DataFrame) and result.shape != (args.days, args.symbols + 2):
        raise ValueError(f"unexpected daily_return shape: {result.shape}")
    # 保持结果活到测量后；shape 等结果检查不计入时间或峰值。
    return {
        "seconds": elapsed,
        "process_peak_before_mib": peak_before,
        "process_peak_after_mib": peak_after,
        "rows": len(df),
        "result_shape": list(result.shape) if isinstance(result, pd.DataFrame) else None,
        "result_bytes": len(result) if isinstance(result, bytes) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--days", type=int, default=252)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--label", default="working-tree", help="记录被测版本；脚本不会自动切换或构建版本")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.symbols, args.days, args.stride, args.repeat) < 1:
        parser.error("symbols/days/stride/repeat must be positive")
    if args.stride > min(args.symbols, args.days):
        parser.error("stride must not exceed symbols or days")
    if sys.platform not in ("darwin", "linux"):
        parser.error("RSS measurement currently supports macOS and Linux")
    if args.case:
        print(json.dumps(run_case(args)))
        return

    report = {
        "label": args.label,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "polars")},
        "threads": {name: os.environ.get(name) for name in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS")},
        "symbols": args.symbols,
        "days": args.days,
        "stride": args.stride,
        "repeat": args.repeat,
        "cases": {},
    }
    for case in CASES:
        samples = []
        for _ in range(args.repeat):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--case", case],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            samples.append(json.loads(completed.stdout))
        report["cases"][case] = {
            "median_seconds": statistics.median(sample["seconds"] for sample in samples),
            "median_process_peak_mib": statistics.median(sample["process_peak_after_mib"] for sample in samples),
            "samples": samples,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
