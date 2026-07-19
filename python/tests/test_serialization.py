from __future__ import annotations

import json

import numpy as np
import pytest

import wbt
from wbt import WeightBacktest
from wbt.result import BacktestResult, _json_safe
from wbt.serialization import FORMAT, FORMAT_VERSION, load_msgpack, to_msgpack


@pytest.fixture
def result(wb: WeightBacktest) -> BacktestResult:
    return wb.to_result()


# ---------------------------------------------------------------------------
# 非有限浮点：to_dict 必须是严格 JSON（allow_nan=False 不报错）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("full", [False, True])
def test_to_dict_is_strict_json(result: BacktestResult, full: bool) -> None:
    payload = result.to_dict(full=full)
    # allow_nan=False：一旦出现 NaN / Infinity 会抛 ValueError
    json.dumps(payload, allow_nan=False)


def test_json_safe_coerces_non_finite() -> None:
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None
    assert _json_safe(np.float64("nan")) is None
    assert _json_safe(np.float64("inf")) is None
    # 有限值原样保留
    assert _json_safe(1.5) == 1.5
    assert _json_safe(np.float64(2.0)) == 2.0
    # 数组内的 NaN 也被清洗
    assert _json_safe(np.array([1.0, np.nan, np.inf])) == [1.0, None, None]
    # bool 不被误当作 float
    assert _json_safe(True) is True


# ---------------------------------------------------------------------------
# MessagePack 读写
# ---------------------------------------------------------------------------
def test_roundtrip_via_methods(result: BacktestResult, tmp_path) -> None:
    path = tmp_path / "backtest_results.msgpack"
    result.dump_msgpack(path, full=True)
    payload = load_msgpack(path)
    assert payload["symbol_count"] == result.symbol_count
    assert len(payload["dates"]) == len(result.dates)
    assert set(payload["curves"].keys()) == {"多空", "多头", "空头", "基准", "超额"}


def test_roundtrip_keeps_format_constants(result: BacktestResult, tmp_path) -> None:
    path = tmp_path / "format.msgpack"
    path.write_bytes(result.to_msgpack(full=False))
    payload = load_msgpack(path)
    assert FORMAT == "wbt.backtest_result"
    assert FORMAT_VERSION == 1
    assert payload["symbol_count"] == result.symbol_count


def test_full_flag_controls_payload(result: BacktestResult, tmp_path) -> None:
    compact_path = tmp_path / "compact.msgpack"
    full_path = tmp_path / "full.msgpack"
    compact_path.write_bytes(result.to_msgpack(full=False))
    full_path.write_bytes(result.to_msgpack(full=True))
    compact = load_msgpack(compact_path)
    full = load_msgpack(full_path)
    assert "drawdowns" not in compact
    assert "drawdowns" in full


def test_top_level_helpers_exposed(result: BacktestResult) -> None:
    assert wbt.to_msgpack is to_msgpack
    data = wbt.to_msgpack(result)
    assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# 错误封装被拒绝
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\x81\xa6format\xa4nope", "unexpected format"),
        (
            b"\x83\xa6format\xb3wbt.backtest_result\xaeformat_version\xcd\x03\xe7\xa7payload\x80",
            "unsupported format_version",
        ),
        (b"\x93\x01\x02\x03", "expected mapping"),
    ],
)
def test_invalid_envelope_rejected(tmp_path, data: bytes, message: str) -> None:
    path = tmp_path / "bad.msgpack"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        load_msgpack(path)
