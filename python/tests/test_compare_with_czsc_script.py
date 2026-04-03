from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_with_czsc_real_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_with_czsc_real_data", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_operation_specs_matches_public_contract() -> None:
    module = load_module()
    ops = module.build_operation_specs("000001.SZ")
    assert [op.name for op in ops] == [
        "stats",
        "daily_return",
        "dailys",
        "alpha",
        "alpha_stats",
        "bench_stats",
        "long_daily_return",
        "short_daily_return",
        "long_stats",
        "short_stats",
        "get_symbol_daily",
        "get_symbol_pairs",
        "daily_performance",
    ]
