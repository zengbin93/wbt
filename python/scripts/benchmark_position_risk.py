"""Reproducible release-only position-risk benchmark and standalone HTML report."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def make_case(seed, times, symbols, density, shuffled):
    rng = np.random.default_rng(seed)
    updates = max(1, round(symbols * density))
    timestamp_ids = np.repeat(np.arange(times), updates)
    symbol_ids = np.tile(np.arange(symbols), times) if density == 1 else rng.integers(0, symbols, len(timestamp_ids))
    weights = rng.normal(0, 0.4, len(timestamp_ids))
    weights[rng.random(len(weights)) < 0.1] = 0.0
    weights[rng.random(len(weights)) < 0.03] = np.nan
    names = np.array([f"S{symbol:05d}" for symbol in range(symbols)])
    frame = pd.DataFrame(
        {
            "dt": pd.Timestamp("2026-09-01") + pd.to_timedelta(timestamp_ids, unit="min"),
            "symbol": names[symbol_ids],
            "weight": weights,
        }
    )
    if shuffled:
        duplicates = frame.iloc[::10].copy()
        duplicates["weight"] = rng.normal(0, 2, len(duplicates))
        frame = pd.concat([frame, duplicates], ignore_index=True).sample(frac=1, random_state=seed)
    return frame.reset_index(drop=True)


def summary(samples):
    return {
        "median_ms": statistics.median(samples) * 1000,
        "min_ms": min(samples) * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
        "samples_seconds": samples,
    }


def benchmark_case(name, frame, repeats, warmups, seed):
    from position_risk_reference import calculate_position_risk as python_risk
    from position_risk_reference import calculate_prepared

    from wbt import _wbt, calculate_position_risk
    from wbt._df_convert import arrow_bytes_to_pd_df
    from wbt.position_risk import _prepare_frame, _to_arrow

    prepared = _prepare_frame(frame)
    payload = _to_arrow(prepared)
    output, _, is_debug = _wbt._profile_position_risk(payload)
    if is_debug:
        raise RuntimeError("Benchmark requires a release extension; debug assertions are enabled")
    actual = calculate_position_risk(frame)
    expected = python_risk(frame)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    difference = np.abs(actual.iloc[:, 1:].to_numpy() - expected.iloc[:, 1:].to_numpy())
    finite_difference = difference[np.isfinite(difference)]
    functions = {
        "rust_e2e": lambda: calculate_position_risk(frame),
        "python_e2e": lambda: python_risk(frame),
        "python_core": lambda: calculate_prepared(prepared),
        "normalize": lambda: _prepare_frame(frame),
        "python_ipc_encode": lambda: _to_arrow(prepared),
        "python_ipc_decode": lambda: arrow_bytes_to_pd_df(output),
        "native_profile_call": lambda: _wbt._profile_position_risk(payload),
    }
    samples = {key: [] for key in [*functions, "rust_ipc_decode", "rust_core", "rust_ipc_encode", "conversion_total"]}
    for _ in range(warmups):
        for function in functions.values():
            function()
    order_rng = random.Random(seed)
    for _ in range(repeats):
        keys = list(functions)
        order_rng.shuffle(keys)
        for key in keys:
            start = perf_counter()
            value = functions[key]()
            elapsed = perf_counter() - start
            samples[key].append(elapsed)
            if key == "native_profile_call":
                for phase, duration in zip(["rust_ipc_decode", "rust_core", "rust_ipc_encode"], value[1], strict=True):
                    samples[phase].append(duration)
            del value
        samples["conversion_total"].append(
            sum(
                samples[key][-1]
                for key in [
                    "python_ipc_encode",
                    "rust_ipc_decode",
                    "rust_ipc_encode",
                    "python_ipc_decode",
                ]
            )
        )
    measurements = {key: summary(values) for key, values in samples.items()}
    return {
        "name": name,
        "rows": len(frame),
        "times": frame.dt.nunique(),
        "symbols": frame.symbol.nunique(),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "max_abs_error": float(finite_difference.max()) if len(finite_difference) else 0.0,
        "measurements": measurements,
        "e2e_speedup": measurements["python_e2e"]["median_ms"] / measurements["rust_e2e"]["median_ms"],
        "core_speedup": measurements["python_core"]["median_ms"] / measurements["rust_core"]["median_ms"],
    }


def render_report(data):
    rows, details = [], []
    for case in data["cases"]:
        measurements = case["measurements"]
        cells = [html.escape(case["name"]), f"{case['rows']:,}", f"{case['times']:,}", f"{case['symbols']:,}"]
        for key in ["python_e2e", "rust_e2e", "python_core", "rust_core", "conversion_total"]:
            cells.append(f"{measurements[key]['median_ms']:.3f}")
        cells += [f"{case['e2e_speedup']:.2f}×", f"{case['core_speedup']:.2f}×"]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        phase_rows = "".join(
            f"<tr><td>{html.escape(key)}</td><td>{value['min_ms']:.4f}</td>"
            f"<td>{value['median_ms']:.4f}</td><td>{value['p95_ms']:.4f}</td></tr>"
            for key, value in measurements.items()
        )
        details.append(
            f"<details><summary>{html.escape(case['name'])} — 分层统计 / ms</summary>"
            f"<table><tr><th>阶段</th><th>Min</th><th>Median</th><th>P95</th></tr>{phase_rows}</table></details>"
        )
    metadata = html.escape(json.dumps(data["metadata"], ensure_ascii=False, indent=2))
    embedded = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>WBT · Position Risk 性能报告</title><style>
body{{font:16px/1.65 system-ui,sans-serif;background:#f3f6fa;color:#20304a;margin:0}}
main{{max-width:1320px;margin:40px auto;padding:0 28px}}h1{{font-size:36px;line-height:1.2}}
h2{{margin-top:34px}}header{{background:#102a43;color:#fff;padding:36px;border-radius:16px}}
.tag{{color:#67e8cf}}section,details{{background:white;padding:20px 24px;border-radius:12px;margin:18px 0}}
.scroll,details{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dfe7ef;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}th{{background:#eef4f9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
summary{{cursor:pointer;font-weight:600}}code{{background:#e8eef4;padding:2px 5px}}.note{{border-left:4px solid #13a88a}}
</style></head><body><main><header><div class="tag">WBT / SKZ-721 · RELEASE BENCHMARK</div>
<h1>持仓风险度：Rust 与同算法 Python</h1><p>先验证语义，再测量性能。不预设加速结论，不把算法差异计入语言加速。</p></header>
<section class="note"><h2>结论边界</h2><p>以下是本机、固定 mock、预热后重复测量的结果。
Python 对照按 Rust 的「稳定排序 + 聚合树」重新实现，不使用飞书原 Python 源码。
这不是对所有 pandas/NumPy 实现或所有硬件的速度承诺；小输入可能由转换开销主导。
速比 = Python 中位数 / Rust 中位数，小于 1 表示 Rust 更慢。</p></section>
<section class="scroll"><h2>主结果（中位数，毫秒）</h2><table><tr><th>场景</th><th>N 行</th><th>T 时间</th><th>S 品种</th>
<th>Python E2E</th><th>Rust E2E</th><th>Python 核心</th><th>Rust 核心</th><th>转换合计</th><th>E2E 速比</th><th>核心速比</th></tr>{"".join(rows)}</table></section>
<section><h2>计时范围与公平性</h2><ul>
<li>两端共享相同原始输入、输入规范化、七个 float64 指标和 datetime 输出契约。逐场景断言所有列数值一致（rtol/atol 1e-12、NaN 同位）。</li>
<li>E2E 是实际公共调用独立计时，包含输入规范化、计算、输出 DataFrame；Rust 还包含四段 Arrow IPC 转换与 PyO3 调用。</li>
<li>核心 = 已规范化的本语言 DataFrame → 输出 DataFrame：包含字段提取、品种编码、校验（Rust）、稳定排序、树更新、结果物化；不是仅树循环。</li>
<li>Rust 内部 Instant 分别测 IPC 解码、完整核心、IPC 编码；普通公共调用不插入计时器。Python 核心独立计时。</li>
<li>转换合计 = Python IPC 编码 + Rust IPC 解码 + Rust IPC 编码 + Python IPC 解码的每轮分段和；不把 E2E−核心的噪声残差当作转换。</li>
<li>分段来自独立调用，不应精确相加为 E2E。PyO3 调用/返回字节拷贝、计时器等未归入四段转换；native_profile_call 给出整个仪表化边界调用供检查。</li>
<li>固定 seed、同一进程、预热、每轮随机交错顺序、保留全部样本及 min/median/p95。数据生成、正确性断言、导入、编译和 HTML 生成不计时。</li>
<li>这是墙钟耗时，未绑定 CPU、未控制系统负载/温度；少量重复的 P95 只是描述统计，不是置信区间。未测峰值内存或并发吞吐。</li>
</ul></section>
<section><h2>实现与正确性</h2><p>时间必须先升序；品种未出现前为空仓，NaN 沿用历史，跨日不断仓，零表示平仓。
同时间同品种最后非缺失值胜出。允许杠杆；空头为零时多空比为 NaN。空输入保留八列，拒绝缺失键和无限权重，保留纳秒与时区。</p>
<p>连续 Vec 聚合树维护总/多/空/净/最大/平方和，更新复杂度 O(log S)，总复杂度 O(N log N + N log S + T)，空间 O(N+S+T)，不创建 T×S 网格。
重算父节点避免历史增减累积误差；float64 仍存在舍入与溢出限制。没有 unsafe、fast-math 或并行归约。</p>
<p>测试使用手算预期和独立稠密历史 oracle（每时刻每品种重选历史最后有效记录并用 math.fsum），不只验证两棵树互相一致。
覆盖乱序、隔夜、重复、缺失、全空仓、杠杆、纳秒/时区、空输入、错误输入、大仓平仓恢复小仓；具体运行证据见随 PR 提交的 verification.txt。</p>
<p>实现前参考 Rust Book iterator performance、Cargo profiles、PyO3 0.28 performance/parallelism 官方文档；来源及设计取舍见 docs/position_risk.md。</p></section>
<h2>分场景分层数据</h2>{"".join(details)}
<section><h2>环境、构建与复现</h2><pre>{metadata}</pre><p>运行命令与锁定依赖见 docs/position_risk.md；JSON 保留全部原始样本。</p></section>
<details><summary>内嵌完整 JSON（离线可审计）</summary><pre>{embedded}</pre></details>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/position_risk"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.warmups < 1:
        parser.error("use at least two repetitions and one warmup")
    root = Path(__file__).resolve().parents[2]
    sources = [
        "src/core/position_risk.rs",
        "src/python.rs",
        "python/wbt/position_risk.py",
        "python/scripts/position_risk_reference.py",
        "python/scripts/benchmark_position_risk.py",
    ]
    from wbt import _wbt

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "packages": {
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "polars", "maturin"]
        },
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True).strip(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "extension_sha256": hashlib.sha256(Path(_wbt.__file__).read_bytes()).hexdigest(),
        "cargo_lock_sha256": hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest(),
        "build": "maturin develop --release; opt-level=3, lto=fat, codegen-units=1; debug assertions checked off",
        "command": " ".join(sys.argv),
    }
    cases = [("tiny", 4, 3, 1.0, False), ("small-dense", 100, 10, 1.0, False)]
    if not args.quick:
        cases += [
            ("dense-100k", 1000, 100, 1.0, False),
            ("dense-500k", 2000, 250, 1.0, False),
            ("sparse-1000-symbols", 20000, 1000, 0.003, False),
            ("shuffled-duplicates", 2000, 100, 0.2, True),
        ]
    data = {"metadata": metadata, "cases": []}
    for index, (name, times, symbols, density, shuffled) in enumerate(cases):
        frame = make_case(args.seed + index, times, symbols, density, shuffled)
        result = benchmark_case(name, frame, args.repeats, args.warmups, args.seed + index)
        data["cases"].append(result)
        print(f"{name}: E2E {result['e2e_speedup']:.2f}x; core {result['core_speedup']:.2f}x", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "report.html").write_text(render_report(data), encoding="utf-8")
    (args.output / "cargo-lock.txt").write_bytes((root / "Cargo.lock").read_bytes())


if __name__ == "__main__":
    main()
