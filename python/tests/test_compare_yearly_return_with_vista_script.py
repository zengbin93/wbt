"""Integration test for scripts/compare_yearly_return_with_vista.py

契约测试：任何环境下都跑，验证脚本的公共函数可用。
真实数据对比：仅在 /Volumes/jun/ 数据 + vista 源码都可访问时运行，
             要求 wbt.yearly_return 与 vista.calculate_yearly_returns 逐行 bit-exact 一致。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_yearly_return_with_vista.py"
REAL_DATA_PATH = Path("/Volumes/jun/全A日线测试_20170101_20250429.feather")
VISTA_ROOT = Path("/Users/0xjun/Documents/cursorPro/vista")


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_yearly_return_with_vista", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_script_exposes_public_api() -> None:
    module = _load_script()
    assert callable(module.run_comparison)
    assert callable(module.main)
    assert callable(module.load_vista_calculate_yearly_returns)


@pytest.mark.skipif(
    not REAL_DATA_PATH.exists(),
    reason=f"real data not available: {REAL_DATA_PATH}",
)
@pytest.mark.skipif(
    not (VISTA_ROOT / "vista" / "utils" / "yearly_return.py").exists(),
    reason=f"vista repo not available: {VISTA_ROOT}",
)
def test_yearly_return_matches_vista_row_by_row_on_real_data() -> None:
    """真实数据（全 A 日线 2017-2025，840 万行 × 5351 symbol）逐行 bit-exact 对比。"""
    module = _load_script()
    dfw = pd.read_feather(REAL_DATA_PATH)
    result = module.run_comparison(dfw, min_days=120, n_jobs=4, vista_root=VISTA_ROOT)

    # 行数必须一致
    assert result.wbt_rows == result.vista_rows > 0, (
        f"rows mismatch: wbt={result.wbt_rows}, vista={result.vista_rows}"
    )
    # 允许的最大绝对误差 = TOLERANCE (1e-12)；实测 0.0（bit-exact）
    assert result.matched, f"return values differ: max_abs_diff={result.max_abs_diff:.3e}"
    # 额外断言：本仓库最佳水平是 bit-exact（不是 1e-12 内近似），若未来偶然退化也能抓到
    assert result.max_abs_diff == 0.0, f"expected bit-exact, got max_abs_diff={result.max_abs_diff:.3e}"
