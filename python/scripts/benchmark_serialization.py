"""可复跑的 BacktestResult JSON / MessagePack 文件读写基准。"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import wbt


def make_result() -> wbt.BacktestResult:
    rng = np.random.default_rng(353)
    rows = []
    for symbol in ("AAA", "BBB"):
        for day in range(40):
            for hour in range(4):
                dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour + 9, minutes=30)
                rows.append(
                    {
                        "dt": dt.isoformat(sep=" "),
                        "symbol": symbol,
                        "weight": round(rng.uniform(-0.5, 0.5), 2),
                        "price": round(100 + rng.normal(0, 2), 4),
                    }
                )
    return wbt.WeightBacktest(pd.DataFrame(rows), n_jobs=1).to_result()


def samples(operation, rounds: int) -> list[int]:
    operation()
    values = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        operation()
        values.append(time.perf_counter_ns() - started)
    return values


def summarize(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered) / 1_000_000,
        "min_ms": ordered[0] / 1_000_000,
        "p95_ms": ordered[round((len(ordered) - 1) * 0.95)] / 1_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    result = make_result()
    report: dict[str, object] = {"rounds": args.rounds, "python": __import__("sys").version, "formats": {}}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for full in (False, True):
            paths = {"json": root / f"result-{full}.json", "msgpack": root / f"result-{full}.msgpack"}
            writers = {
                "json": lambda path=paths["json"], is_full=full: result.dump_json(path, full=is_full),
                "msgpack": lambda path=paths["msgpack"], is_full=full: result.dump_msgpack(path, full=is_full),
            }
            readers = {
                "json": lambda path=paths["json"]: wbt.load_json(path),
                "msgpack": lambda path=paths["msgpack"]: wbt.load_msgpack(path),
            }
            for writer in writers.values():
                writer()
            wbt.assert_payload_equal(readers["json"](), readers["msgpack"]())
            report["formats"][str(full)] = {
                name: {
                    "write": summarize(samples(writers[name], args.rounds)),
                    "read": summarize(samples(readers[name], args.rounds)),
                    "bytes": paths[name].stat().st_size,
                }
                for name in paths
            }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
