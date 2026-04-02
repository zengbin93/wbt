from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "perf_compare_real_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("perf_compare_real_data", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_operation_specs_covers_public_return_value_accessors() -> None:
    module = load_module()
    ops = module.build_operation_specs("000001.SZ")
    names = [op.name for op in ops]
    assert names == [
        "stats",
        "daily_return",
        "dailys",
        "alpha",
        "pairs",
        "alpha_stats",
        "bench_stats",
        "long_daily_return",
        "short_daily_return",
        "long_stats",
        "short_stats",
        "symbol_dict",
        "get_symbol_daily",
        "get_symbol_pairs",
        "get_top_symbols_profit",
        "get_top_symbols_loss",
    ]
