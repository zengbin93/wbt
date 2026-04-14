"""WeightBacktest.yearly_return integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wbt import WeightBacktest


def _multi_year_dfw() -> pd.DataFrame:
    """跨两年（2023 + 2024）的持仓权重数据，两个 symbol，每日 4 根 K 线。"""
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    dates = [f"2023-12-{20 + i:02d}" for i in range(10)] + [f"2024-01-{1 + i:02d}" for i in range(10)]
    for sym in ["SYM_A", "SYM_B"]:
        for d in dates:
            for h in range(4):
                dt = f"{d} {9 + h:02d}:30:00"
                w = round(float(rng.uniform(-0.3, 0.3)), 2)
                p = 100.0 + float(rng.normal(0, 2))
                rows.append({"dt": dt, "symbol": sym, "weight": w, "price": round(p, 4)})
    return pd.DataFrame(rows)


def test_yearly_return_returns_dataframe_with_expected_columns(wb: WeightBacktest) -> None:
    result = wb.yearly_return(min_days=1)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["year", "symbol", "return"]


def test_yearly_return_default_min_days_filters_short_fixture(wb: WeightBacktest) -> None:
    # 15 天数据远少于默认 min_days=120 → 空结果，schema 保留
    result = wb.yearly_return()
    assert result.empty
    assert list(result.columns) == ["year", "symbol", "return"]


def test_yearly_return_single_year_contains_symbols_and_total(wb: WeightBacktest) -> None:
    result = wb.yearly_return(min_days=1)
    assert (result["year"] == 2024).all()
    syms = set(result["symbol"].tolist())
    assert {"SYM_A", "SYM_B", "total"}.issubset(syms)


def test_yearly_return_multi_year_matches_vista_formula() -> None:
    df = _multi_year_dfw()
    wb = WeightBacktest(df, digits=2, fee_rate=0.0002, n_jobs=1, weight_type="ts", yearly_days=252)

    result = wb.yearly_return(min_days=5)
    assert set(result["year"].unique()) == {2023, 2024}

    # vista 公式：(1+r).prod() - 1，按 (year, symbol) 自行复算一致
    daily = wb.daily_return.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")
    for _, row in result.iterrows():
        year = int(row["year"])
        sym = str(row["symbol"])
        actual = float(row["return"])
        series = daily[daily.index.year == year][sym].dropna()
        expected = float((1 + series).prod() - 1)
        assert abs(actual - expected) < 1e-10, f"(year={year}, sym={sym}) expected {expected}, got {actual}"


def test_yearly_return_sorted_by_year_then_symbol() -> None:
    df = _multi_year_dfw()
    wb = WeightBacktest(df, n_jobs=1)
    result = wb.yearly_return(min_days=5)
    year_sym = list(zip(result["year"].tolist(), result["symbol"].tolist(), strict=True))
    assert year_sym == sorted(year_sym)
