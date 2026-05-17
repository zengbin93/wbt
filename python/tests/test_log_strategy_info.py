from __future__ import annotations

import logging

import pandas as pd
import pytest
from loguru import logger as _loguru_logger


@pytest.fixture
def loguru_to_caplog(caplog: pytest.LogCaptureFixture):
    """把 loguru 输出桥到标准 logging 以便 caplog 捕获。"""

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = _loguru_logger.add(_Sink(), level="DEBUG")
    yield caplog
    _loguru_logger.remove(handler_id)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A"] * 3 + ["B"] * 3,
            "dt": pd.date_range("2024-01-01", periods=3).tolist() * 2,
            "weight": [0.1, -0.2, 0.0, 0.3, 0.0, None],
        }
    )


def test_normal_df_logs_basic_info(loguru_to_caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.log_strategy_info import log_strategy_info

    with loguru_to_caplog.at_level(logging.INFO):
        log_strategy_info("S1", _sample_df())
    text = "\n".join(rec.message for rec in loguru_to_caplog.records)
    assert "策略 S1 数据详情" in text
    assert "品种数量: 2" in text


def test_empty_df_warns_only(loguru_to_caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.log_strategy_info import log_strategy_info

    with loguru_to_caplog.at_level(logging.WARNING):
        log_strategy_info("S2", pd.DataFrame(columns=["symbol", "dt", "weight"]))
    assert any("数据为空" in rec.message for rec in loguru_to_caplog.records)


def test_quality_warning_for_nan_and_zero(loguru_to_caplog: pytest.LogCaptureFixture) -> None:
    from wbt.utils.log_strategy_info import log_strategy_info

    with loguru_to_caplog.at_level(logging.WARNING):
        log_strategy_info("S3", _sample_df())
    assert any("数据质量提醒" in rec.message for rec in loguru_to_caplog.records)
