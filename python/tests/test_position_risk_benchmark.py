import runpy
from pathlib import Path

import pandas as pd


def test_mock_and_report_are_reproducible(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "benchmark_position_risk.py"
    assert path.exists(), "reproducible benchmark/report runner is missing"
    benchmark = runpy.run_path(str(path))
    first = benchmark["make_case"](17, 20, 5, 0.3, True)
    second = benchmark["make_case"](17, 20, 5, 0.3, True)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.columns) == {"dt", "symbol", "weight"}
    assert not first.dt.is_monotonic_increasing
    report = benchmark["render_report"]({"metadata": {"cpu": "<unsafe>"}, "cases": []})
    assert "&lt;unsafe&gt;" in report
    assert "<unsafe>" not in report
    assert "<!doctype html>" in report.lower()
