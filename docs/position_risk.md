# Position risk: design and reproducibility

`wbt.calculate_position_risk(frame)` accepts a pandas DataFrame with `dt`,
`symbol`, and `weight`, and returns the sorted union of observed timestamps with
`total_risk`, `long_risk`, `short_risk`, `net_exposure`, `max_single_risk`,
`herfindahl`, and `long_short_ratio`.

## Contract

- Sort by time before carrying positions forward, including overnight gaps.
- Before a symbol's first observation its position is zero. A missing weight
  carries the previous position; zero is an explicit close.
- For duplicate time/symbol rows, the last nonmissing weight in original input
  order wins. This resolves the otherwise unspecified duplicate/missing overlap.
- No leverage clipping. A zero short exposure produces a NaN long/short ratio.
- Null time/symbol keys and infinite weights are rejected. Invalid datetime or
  nonnumeric weight values raise errors rather than silently disappearing.
- Empty input returns the same eight-column schema with zero rows.
- Input is not mutated. Timestamp precision/timezone survives the IPC boundary;
  no milliseconds-only normalization is applied.

The task owner confirmed on 2026-09-08 that the document's chronological semantics
take precedence and that the Python comparison must be newly implemented using
the Rust algorithm, not copied/adapted from the original document's Python code.

```python
import pandas as pd
from wbt import calculate_position_risk

frame = pd.DataFrame({
    "dt": ["2026-09-08 09:30", "2026-09-08 09:00"],
    "symbol": ["IF", "RB"],
    "weight": [0.3, -0.2],
})
result = calculate_position_risk(frame)
assert result.total_risk.round(2).tolist() == [0.2, 0.5]
```

Rust callers use `wbt::core::position_risk::calculate_position_risk(&frame)`.
The native input uses a Polars Datetime column and string symbol column;
weights are strictly cast to float64. The Python wrapper additionally parses
datetime/numeric inputs and validates keys before serialization. Additional
input columns are ignored by the Python wrapper.

## Algorithm and tradeoffs

1. Intern symbols in first-appearance order, extract typed timestamp/weight
   events, and stable-sort events by timestamp.
2. Store each symbol's current six additive/max statistics in the leaves of a
   contiguous binary aggregation tree. Apply each nonmissing update to its leaf
   and recompute its ancestors. Emit the root only after all events at a time.
3. Materialize seven float64 columns plus the original datetime dtype.

For N input rows, S symbols and T unique timestamps: O(N log N + N log S + T)
time and O(N + S + T) auxiliary storage, rather than allocating a T × S grid.
This also avoids repeatedly scanning all S symbols to recover a maximum when
the previous largest position closes. Recomputing parent sums (rather than
subtracting old exposure from a running total) avoids historical cancellation
drift after large positions close. Floating-point reductions still have ordinary
float64 rounding/overflow limits; bitwise equality is not promised.

No unsafe indexing, new dependency, fast-math, explicit SIMD, or parallel
reduction is introduced. Stable order and a fixed tree keep reductions
deterministic. Release uses the repository's existing opt-level 3, fat LTO and
one codegen unit. The PyO3 boundary retains the repository's Arrow IPC convention
and detaches from Python while decoding, computing and encoding.

## Implementation research

Official documentation consulted before implementation:

- Rust Book, comparing iterator and loop performance:
  https://doc.rust-lang.org/book/ch13-04-performance.html
  — typed iteration does not require replacing safe abstractions with unsafe loops.
- Cargo release profiles:
  https://doc.rust-lang.org/cargo/reference/profiles.html
  — benchmark optimized release artifacts, not debug binaries.
- PyO3 0.28 performance guidance:
  https://pyo3.rs/v0.28.2/performance.html
  — avoid per-element Python object conversions; use a typed boundary.
- PyO3 parallelism / detaching:
  https://pyo3.rs/v0.28.2/parallelism.html
  — native-only work should not hold the interpreter attachment unnecessarily.

The segment-tree choice is a project-specific design decision, not a claim of
speedup based on those sources. Only measured results establish performance.

## Recorded results (2026-09-08)

On Apple M1 / CPython 3.12.13, seven measured repetitions after two warmups:
the 500k-row dense case is 1199.793 ms in Python versus 68.580 ms in Rust
end-to-end (17.49×); complete core times are 1176.004 versus 34.103 ms (34.48×).
The 12-row case is slower through Rust (1.027 versus 0.950 ms, 0.93×). The six
cases cover dense, sparse, and shuffled duplicate inputs with zero observed
numeric differences. See the HTML/JSON for every phase, min/median/P95, raw
samples and qualifications; these are not hardware-independent guarantees.

## Verification and benchmark reproduction

From the repository root (requires Rust and Python 3.10+):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r docs/benchmarks/position_risk/python-requirements.txt
cp docs/benchmarks/position_risk/cargo-lock.txt Cargo.lock
cd python
VIRTUAL_ENV="$PWD/../.venv" ../.venv/bin/maturin develop --release --skip-install
cd ..
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q
cargo test --lib
cargo fmt --all -- --check
cargo clippy --all -- -D warnings -A non_snake_case
.venv/bin/basedpyright --project python/pyproject.toml --pythonpath "$PWD/.venv/bin/python"
PYTHONPATH=python .venv/bin/python python/scripts/benchmark_position_risk.py \
  --seed 721 --repeats 7 --warmups 2 --output docs/benchmarks/position_risk
```

Use `--quick --repeats 2 --warmups 1 --output target/position-risk-smoke` for a
short benchmark smoke test. The runner rejects an extension with debug
assertions enabled. The actual checked-in results use the recorded Rust/Python
versions, machine, source/extension hashes, lockfile and all raw timing samples.
Dependency snapshots describe the measured environment, not new minimum package
requirements. Restore the existing Cargo.lock instead if it contains local
dependency changes you need to preserve; exact replication requires the snapshot.

Optional browser verification (requires Playwright's Chromium):

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/python python/scripts/verify_position_risk_report.py
```

This checks the offline report's result table and expandable phase details, no
page errors, and no page-level horizontal overflow at desktop/mobile widths. The
first expanded mobile-table check caught an overflow; the report now scrolls
wide tables within their own containers.

Artifacts are in `docs/benchmarks/position_risk/`: `report.html` (offline,
self-contained), `results.json`, dependency snapshots and `verification.txt`.
No external services, market data, credentials or database are used by the tests
or mock generator. Imports, generation, assertions and report rendering are
outside measured intervals. The library's public call has no instrumentation;
the private `_wbt._profile_position_risk` returns bytes, three phase durations
(IPC decode, complete DataFrame-to-DataFrame core, IPC encode), and a debug flag.

The baseline uses identical stable event ordering, symbol IDs, tree topology,
missing/duplicate semantics and output construction. It is interpreted CPython,
not a vectorized pandas challenger. Both paths share input normalization only;
correctness also uses hand-calculated fixtures and a separate dense history
oracle with `math.fsum`. Core times include sorting, symbol encoding, tree
allocation and result materialization, not just arithmetic. Phase timings are
independent experiments and cannot be added to exactly reconstruct E2E; wrapper
and PyO3 byte copying overhead is not silently labeled as core compute.

## Regression discoveries

- The first public API test failed because the function did not exist, then
  passed after the initial implementation.
- Expanded tests found that Polars' general integer-to-Datetime cast drops
  timezone metadata without its optional timezones feature. Constructing the
  output via `into_datetime` preserves the existing metadata without adding a
  new dependency feature or altering epoch timestamps.
- Nullable `pd.NA`, non-string keys and infinities required explicit boundary
  validation. Native nonnumeric weights initially became null under Polars'
  default non-strict cast; a failing direct-IPC regression drove strict casting.
- The ordinary float64 limitation remains: highly cancelling simultaneous
  positions may lose precision, and large finite weights may overflow their
  square/aggregate. Closing a large position does not leave accumulated residue
  in otherwise small subsequent exposures.
