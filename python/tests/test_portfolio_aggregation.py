"""F04: total, alpha and review returns share the TS/CS portfolio series."""

import numpy as np
import polars as pl
import pytest

from wbt import WeightBacktest


@pytest.fixture(params=["ts", "cs"])
def mode(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["fixed_symbols", "changing_symbols"])
def sparse(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["all_long", "long_short"])
def short(request):
    return request.param


@pytest.fixture(params=["pandas", "polars", "csv", "parquet", "arrow"])
def portfolio(request, mode, sparse, short, tmp_path):
    dates = ["2023-12-28", "2023-12-29", "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02", "2024-01-03"]
    b_len = 2 if sparse else 7
    df = pl.DataFrame(
        {
            "dt": [d + " 09:00:00" for d in dates + dates[:b_len]],
            "symbol": ["A"] * 7 + ["B"] * b_len,
            "weight": [0.5] * 7 + ([-0.25] if short else [0.5]) * b_len,
            "price": [100.0, 90.0, 99.0, 108.9, 98.01, 107.811, 118.5921]
            + [100.0, 80.0, 88.0, 96.8, 77.44, 85.184, 93.7024][:b_len],
        }
    )
    path = request.param
    if path == "pandas":
        data = df.to_pandas()
    elif path == "polars":
        data = df
    else:
        data = tmp_path / f"weights.{path}"
        if path == "csv":
            df.write_csv(data)
        elif path == "parquet":
            df.write_parquet(data)
        else:
            df.write_ipc(data)
    return WeightBacktest(data, weight_type=mode, fee_rate=0, n_jobs=1)


def expected_returns(mode, sparse, short):
    # A: 0.5 * (-10%, 10%, 10%, -10%, 10%, 10%).
    a = np.array([-0.05, 0.05, 0.05, -0.05, 0.05, 0.05])
    # B: price returns (-20%, 10%, 10%, -20%, 10%, 10%).
    b = np.array([0.05, -0.025, -0.025, 0.05, -0.025, -0.025] if short else [-0.10, 0.05, 0.05, -0.10, 0.05, 0.05])
    if sparse:
        b[1:] = 0
    count = np.array([2, 1, 1, 1, 1, 1] if sparse else [2] * 6)
    total = (a + b) / count if mode == "ts" else a + b
    # Benchmark remains the equal-weight mean of active symbols, in both modes.
    benchmark = np.array([-0.15, 0.10, 0.10, -0.10 if sparse else -0.15, 0.10, 0.10])
    return total, benchmark


def test_alpha_uses_portfolio_total(portfolio, mode, sparse, short):
    total, benchmark = expected_returns(mode, sparse, short)
    dr = portfolio.daily_return
    alpha = portfolio.alpha
    assert alpha["date"].tolist() == dr["date"].tolist()
    np.testing.assert_allclose(dr["total"], total, atol=1e-12)
    np.testing.assert_allclose(alpha["策略"], total, atol=1e-12)
    np.testing.assert_allclose(alpha["基准"], benchmark, atol=1e-12)
    np.testing.assert_allclose(alpha["超额"], total - benchmark, atol=1e-12)
    result = portfolio.to_result()
    np.testing.assert_allclose(result.curves["多空"].daily, total, atol=1e-12)
    np.testing.assert_allclose(result.curves["超额"].daily, total - benchmark, atol=1e-12)


def test_review_yearly_and_recent_returns_use_portfolio_total(portfolio, mode, sparse, short):
    total, _ = expected_returns(mode, sparse, short)
    yearly = [float(total[:3].sum()), float(total[3:].sum())]
    assert portfolio.stats["绝对收益"] == pytest.approx(round(sum(yearly), 4))
    yr = portfolio.yearly_return(min_days=1)
    np.testing.assert_allclose(yr.loc[yr["symbol"] == "total", "return"], yearly, atol=1e-12)
    history = portfolio.is_good_strategy(mode="history", min_year_days=1)
    assert [m["year"] for m in history["yearly_metrics"]] == [2023, 2024]
    np.testing.assert_allclose([m["abs_return"] for m in history["yearly_metrics"]], yearly, atol=1e-12)
    recent = portfolio.is_good_strategy(mode="recent", recent_days=4, min_history_days=0)
    assert recent["recent_abs_return"] == pytest.approx(float(total[-4:].sum()), abs=1e-12)
    result = portfolio.to_result()
    np.testing.assert_allclose(result.yearly_returns.abs_returns, yearly, atol=1e-12)
    np.testing.assert_allclose([m["abs_return"] for m in result.verdict["yearly_metrics"]], yearly, atol=1e-12)
    if sparse and not short:
        # A fixed mean silently flips the 2023 CS loss (-5%) into a gain (+2.5%).
        assert yearly[0] == pytest.approx(-0.05 if mode == "cs" else 0.025)


def test_recent_window_includes_changing_symbol_count(portfolio, mode, sparse, short):
    total, _ = expected_returns(mode, sparse, short)
    recent = portfolio.is_good_strategy(mode="recent", recent_days=6, min_history_days=0)
    assert recent["recent_actual_days"] == 6
    assert recent["recent_abs_return"] == pytest.approx(float(total.sum()), abs=1e-12)
