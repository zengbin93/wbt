"""
wbt vs rs_czsc 全面数据对齐测试

覆盖范围：
- 所有参数组合：weight_type (ts/cs), digits (1/2/3), fee_rate (0/0.0002/0.001), n_jobs (1/4/8), yearly_days (242/252/365)
- 所有属性/方法：stats, daily_return, dailys, alpha, alpha_stats, bench_stats, pairs,
                 long_daily_return, short_daily_return, long_stats, short_stats,
                 symbol_dict, get_symbol_daily, get_symbol_pairs, get_top_symbols
- 逐行逐列对比，精度到 1e-10
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd

# ── Imports ──────────────────────────────────────────────────
sys.path.insert(0, "/Users/0xjun/Documents/cursorPro/rs_czsc/python")
from rs_czsc._trader.weight_backtest import WeightBacktest as OrigWeightBacktest
from rs_czsc._rs_czsc import daily_performance as orig_daily_performance

sys.path.insert(0, "/Users/0xjun/Documents/vsPro/wbt/python")
from wbt.backtest import WeightBacktest as WbtWeightBacktest
from wbt._wbt import daily_performance as wbt_daily_performance

DATA_PATH = "/Volumes/jun/全A日线测试_20170101_20250429.feather"
TOLERANCE = 1e-10

# ── Helpers ──────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1

    def fail(self, name, detail):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"    FAIL {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TOTAL: {total} checks, {self.passed} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed checks:")
            for name, detail in self.errors:
                print(f"  - {name}: {detail}")
        print(f"{'='*60}")
        return self.failed == 0


def compare_scalar(results, name, wbt_val, orig_val, tol=TOLERANCE):
    """Compare two scalar values."""
    if isinstance(wbt_val, str) and isinstance(orig_val, str):
        if wbt_val == orig_val:
            results.ok(name)
        else:
            results.fail(name, f"wbt='{wbt_val}' vs orig='{orig_val}'")
        return

    if wbt_val is None and orig_val is None:
        results.ok(name)
        return
    if wbt_val is None or orig_val is None:
        results.fail(name, f"wbt={wbt_val} vs orig={orig_val}")
        return

    try:
        w = float(wbt_val)
        o = float(orig_val)
        if abs(w - o) <= tol:
            results.ok(name)
        else:
            results.fail(name, f"wbt={w} vs orig={o}, diff={abs(w-o):.2e}")
    except (TypeError, ValueError):
        if str(wbt_val) == str(orig_val):
            results.ok(name)
        else:
            results.fail(name, f"wbt={wbt_val} vs orig={orig_val}")


def compare_dict(results, prefix, wbt_dict, orig_dict, tol=TOLERANCE):
    """Compare two dicts key-by-key."""
    all_keys = set(list(wbt_dict.keys()) + list(orig_dict.keys()))
    for k in sorted(all_keys):
        if k not in wbt_dict:
            results.fail(f"{prefix}[{k}]", "missing in wbt")
        elif k not in orig_dict:
            results.fail(f"{prefix}[{k}]", "missing in orig")
        else:
            compare_scalar(results, f"{prefix}[{k}]", wbt_dict[k], orig_dict[k], tol)


def compare_df(results, prefix, wbt_df, orig_df, sort_cols=None, tol=TOLERANCE):
    """Compare two DataFrames row-by-row, column-by-column."""
    # Shape
    if wbt_df.shape != orig_df.shape:
        results.fail(f"{prefix}.shape", f"wbt={wbt_df.shape} vs orig={orig_df.shape}")
        return

    results.ok(f"{prefix}.shape")

    # Columns
    wbt_cols = set(wbt_df.columns)
    orig_cols = set(orig_df.columns)
    if wbt_cols != orig_cols:
        missing = orig_cols - wbt_cols
        extra = wbt_cols - orig_cols
        if missing:
            results.fail(f"{prefix}.columns", f"missing: {missing}")
        if extra:
            results.fail(f"{prefix}.columns", f"extra: {extra}")
        common = wbt_cols & orig_cols
    else:
        common = wbt_cols
        results.ok(f"{prefix}.columns")

    # Sort if needed
    if sort_cols:
        valid_sort = [c for c in sort_cols if c in common]
        if valid_sort:
            wbt_df = wbt_df.sort_values(valid_sort).reset_index(drop=True)
            orig_df = orig_df.sort_values(valid_sort).reset_index(drop=True)

    # Compare each column
    for col in sorted(common):
        w = wbt_df[col]
        o = orig_df[col]
        name = f"{prefix}[{col}]"

        if w.dtype in ['float64', 'float32'] and o.dtype in ['float64', 'float32']:
            diff = (w - o).abs()
            max_diff = diff.max()
            if np.isnan(max_diff):
                # Check NaN positions match
                w_nan = w.isna()
                o_nan = o.isna()
                if (w_nan == o_nan).all():
                    non_nan = ~w_nan
                    if non_nan.sum() > 0:
                        max_diff_nn = (w[non_nan] - o[non_nan]).abs().max()
                        if max_diff_nn <= tol:
                            results.ok(name)
                        else:
                            results.fail(name, f"max_diff={max_diff_nn:.2e} (NaN positions match)")
                    else:
                        results.ok(name)
                else:
                    results.fail(name, f"NaN positions differ: wbt has {w_nan.sum()}, orig has {o_nan.sum()}")
            elif max_diff <= tol:
                results.ok(name)
            else:
                n_mismatch = (diff > tol).sum()
                results.fail(name, f"max_diff={max_diff:.2e}, {n_mismatch}/{len(w)} rows differ")
        elif w.dtype == o.dtype or (str(w.dtype).startswith('datetime') and str(o.dtype).startswith('datetime')):
            if (w == o).all():
                results.ok(name)
            else:
                n_mismatch = (w != o).sum()
                results.fail(name, f"{n_mismatch}/{len(w)} rows differ")
        else:
            # Different dtypes, cast to string
            if (w.astype(str) == o.astype(str)).all():
                results.ok(name)
            else:
                n_mismatch = (w.astype(str) != o.astype(str)).sum()
                results.fail(name, f"{n_mismatch}/{len(w)} rows differ (dtype: wbt={w.dtype} orig={o.dtype})")


def compare_list(results, name, wbt_list, orig_list):
    """Compare two lists."""
    if wbt_list == orig_list:
        results.ok(name)
    else:
        results.fail(name, f"wbt has {len(wbt_list)} items, orig has {len(orig_list)} items")
        if len(wbt_list) == len(orig_list):
            for i, (w, o) in enumerate(zip(wbt_list, orig_list)):
                if w != o:
                    results.fail(f"{name}[{i}]", f"wbt='{w}' vs orig='{o}'")
                    if i > 5:
                        results.fail(f"{name}[...]", "truncated")
                        break


# ── Test Scenarios ───────────────────────────────────────────

def run_scenario(results, dfw, label, digits, fee_rate, n_jobs, weight_type, yearly_days):
    """Run a full comparison for one parameter set."""
    print(f"\n{'─'*60}")
    print(f"  Scenario: {label}")
    print(f"  digits={digits}, fee_rate={fee_rate}, n_jobs={n_jobs}, weight_type='{weight_type}', yearly_days={yearly_days}")
    print(f"{'─'*60}")

    prefix = label

    try:
        t0 = time.perf_counter()
        wbt = WbtWeightBacktest(dfw.copy(), digits=digits, fee_rate=fee_rate, n_jobs=n_jobs,
                                 weight_type=weight_type, yearly_days=yearly_days)
        wbt_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        orig = OrigWeightBacktest(dfw.copy(), digits=digits, fee_rate=fee_rate, n_jobs=n_jobs,
                                   weight_type=weight_type, yearly_days=yearly_days)
        orig_time = time.perf_counter() - t0

        print(f"  Time: wbt={wbt_time:.3f}s, orig={orig_time:.3f}s, ratio={wbt_time/orig_time:.2f}x")
    except Exception as e:
        results.fail(f"{prefix}.init", f"Exception: {e}")
        traceback.print_exc()
        return

    # 1. stats (dict, 29 keys)
    try:
        compare_dict(results, f"{prefix}.stats", wbt.stats, orig.stats)
    except Exception as e:
        results.fail(f"{prefix}.stats", f"Exception: {e}")

    # 2. daily_return (DataFrame)
    try:
        compare_df(results, f"{prefix}.daily_return", wbt.daily_return, orig.daily_return, sort_cols=["date"])
    except Exception as e:
        results.fail(f"{prefix}.daily_return", f"Exception: {e}")

    # 3. dailys (DataFrame)
    try:
        compare_df(results, f"{prefix}.dailys", wbt.dailys, orig.dailys, sort_cols=["symbol", "date"])
    except Exception as e:
        results.fail(f"{prefix}.dailys", f"Exception: {e}")

    # 4. alpha (DataFrame)
    try:
        compare_df(results, f"{prefix}.alpha", wbt.alpha, orig.alpha, sort_cols=["date"])
    except Exception as e:
        results.fail(f"{prefix}.alpha", f"Exception: {e}")

    # 5. pairs (DataFrame)
    try:
        compare_df(results, f"{prefix}.pairs", wbt.pairs, orig.pairs,
                    sort_cols=["symbol", "开仓时间", "交易方向"])
    except Exception as e:
        results.fail(f"{prefix}.pairs", f"Exception: {e}")

    # 6. alpha_stats (dict)
    try:
        compare_dict(results, f"{prefix}.alpha_stats", wbt.alpha_stats, orig.alpha_stats)
    except Exception as e:
        results.fail(f"{prefix}.alpha_stats", f"Exception: {e}")

    # 7. bench_stats (dict)
    try:
        compare_dict(results, f"{prefix}.bench_stats", wbt.bench_stats, orig.bench_stats)
    except Exception as e:
        results.fail(f"{prefix}.bench_stats", f"Exception: {e}")

    # 8. long_daily_return (DataFrame)
    try:
        compare_df(results, f"{prefix}.long_daily_return", wbt.long_daily_return, orig.long_daily_return,
                    sort_cols=["date"])
    except Exception as e:
        results.fail(f"{prefix}.long_daily_return", f"Exception: {e}")

    # 9. short_daily_return (DataFrame)
    try:
        compare_df(results, f"{prefix}.short_daily_return", wbt.short_daily_return, orig.short_daily_return,
                    sort_cols=["date"])
    except Exception as e:
        results.fail(f"{prefix}.short_daily_return", f"Exception: {e}")

    # 10. long_stats (dict)
    try:
        compare_dict(results, f"{prefix}.long_stats", wbt.long_stats, orig.long_stats)
    except Exception as e:
        results.fail(f"{prefix}.long_stats", f"Exception: {e}")

    # 11. short_stats (dict)
    try:
        compare_dict(results, f"{prefix}.short_stats", wbt.short_stats, orig.short_stats)
    except Exception as e:
        results.fail(f"{prefix}.short_stats", f"Exception: {e}")

    # 12. symbol_dict (list)
    try:
        compare_list(results, f"{prefix}.symbol_dict", wbt.symbol_dict, orig.symbol_dict)
    except Exception as e:
        results.fail(f"{prefix}.symbol_dict", f"Exception: {e}")

    # 13. get_symbol_daily (per-symbol DataFrame)
    try:
        test_symbols = wbt.symbols[:3]  # Test first 3 symbols
        for sym in test_symbols:
            wbt_sd = wbt.get_symbol_daily(sym)
            orig_sd = orig.get_symbol_daily(sym)
            compare_df(results, f"{prefix}.get_symbol_daily({sym})", wbt_sd, orig_sd, sort_cols=["date"])
    except Exception as e:
        results.fail(f"{prefix}.get_symbol_daily", f"Exception: {e}")

    # 14. get_symbol_pairs (per-symbol DataFrame)
    try:
        for sym in test_symbols:
            wbt_sp = wbt.get_symbol_pairs(sym)
            orig_sp = orig.get_symbol_pairs(sym)
            if wbt_sp.shape[0] > 0 and orig_sp.shape[0] > 0:
                compare_df(results, f"{prefix}.get_symbol_pairs({sym})", wbt_sp, orig_sp,
                           sort_cols=["开仓时间"])
            else:
                if wbt_sp.shape == orig_sp.shape:
                    results.ok(f"{prefix}.get_symbol_pairs({sym})")
                else:
                    results.fail(f"{prefix}.get_symbol_pairs({sym})",
                                 f"shape wbt={wbt_sp.shape} orig={orig_sp.shape}")
    except Exception as e:
        results.fail(f"{prefix}.get_symbol_pairs", f"Exception: {e}")

    # 15. get_top_symbols (list)
    try:
        for kind in ["profit", "loss"]:
            for n in [1, 3, 5]:
                wbt_top = wbt.get_top_symbols(n=n, kind=kind)
                orig_top = orig.get_top_symbols(n=n, kind=kind)
                compare_list(results, f"{prefix}.get_top_symbols(n={n},kind={kind})", wbt_top, orig_top)
    except Exception as e:
        results.fail(f"{prefix}.get_top_symbols", f"Exception: {e}")


def run_daily_performance_test(results):
    """Test the standalone daily_performance function."""
    print(f"\n{'─'*60}")
    print(f"  Standalone daily_performance() function")
    print(f"{'─'*60}")

    # Use a known return series
    df = pd.read_feather(DATA_PATH)
    from wbt._wbt import PyWeightBacktest as WbtInner
    from wbt._df_convert import pandas_to_arrow_bytes, arrow_bytes_to_pd_df

    wbt_wb = WbtWeightBacktest(df.copy(), digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts", yearly_days=252)
    returns = wbt_wb.daily_return["total"].to_numpy()

    for yearly_days in [242, 252, 365]:
        wbt_dp = wbt_daily_performance(returns, yearly_days=yearly_days)
        orig_dp = orig_daily_performance(returns, yearly_days=yearly_days)
        compare_dict(results, f"daily_performance(yearly_days={yearly_days})", wbt_dp, orig_dp)


# ── Main ─────────────────────────────────────────────────────

def main():
    results = Results()

    print("Loading data...")
    dfw = pd.read_feather(DATA_PATH)
    print(f"Data: {dfw.shape[0]:,} rows, {dfw['symbol'].nunique()} symbols")

    # ── Scenario 1: Default parameters (baseline) ──
    run_scenario(results, dfw, "S1_default",
                 digits=2, fee_rate=0.0002, n_jobs=8, weight_type="ts", yearly_days=252)

    # ── Scenario 2: Cross-section strategy ──
    run_scenario(results, dfw, "S2_cs",
                 digits=2, fee_rate=0.0002, n_jobs=8, weight_type="cs", yearly_days=252)

    # ── Scenario 3: Higher precision digits ──
    run_scenario(results, dfw, "S3_digits3",
                 digits=3, fee_rate=0.0002, n_jobs=4, weight_type="ts", yearly_days=252)

    # ── Scenario 4: Lower precision digits ──
    run_scenario(results, dfw, "S4_digits1",
                 digits=1, fee_rate=0.0002, n_jobs=4, weight_type="ts", yearly_days=252)

    # ── Scenario 5: Zero fee rate ──
    run_scenario(results, dfw, "S5_zero_fee",
                 digits=2, fee_rate=0.0, n_jobs=4, weight_type="ts", yearly_days=252)

    # ── Scenario 6: Higher fee rate ──
    run_scenario(results, dfw, "S6_high_fee",
                 digits=2, fee_rate=0.001, n_jobs=4, weight_type="ts", yearly_days=252)

    # ── Scenario 7: Single thread ──
    run_scenario(results, dfw, "S7_single_thread",
                 digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252)

    # ── Scenario 8: Different yearly_days ──
    run_scenario(results, dfw, "S8_yearly365",
                 digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts", yearly_days=365)

    # ── Scenario 9: CS + digits=3 + high fee ──
    run_scenario(results, dfw, "S9_cs_d3_hfee",
                 digits=3, fee_rate=0.001, n_jobs=8, weight_type="cs", yearly_days=242)

    # ── Scenario 10: Subset - single symbol ──
    single_sym = dfw[dfw["symbol"] == "000001.SZ"].copy()
    run_scenario(results, single_sym, "S10_single_symbol",
                 digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252)

    # ── Scenario 11: Subset - 10 symbols ──
    top10 = dfw["symbol"].unique()[:10]
    small_df = dfw[dfw["symbol"].isin(top10)].copy()
    run_scenario(results, small_df, "S11_10symbols",
                 digits=2, fee_rate=0.0002, n_jobs=4, weight_type="ts", yearly_days=252)

    # ── Standalone daily_performance ──
    run_daily_performance_test(results)

    # ── Final Summary ──
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
