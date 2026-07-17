from __future__ import annotations

import json

import msgpack
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


def test_envelope_has_format_and_version(result: BacktestResult) -> None:
    data = result.to_msgpack(full=False)
    envelope = msgpack.unpackb(data, raw=False)
    assert envelope["format"] == FORMAT == "wbt.backtest_result"
    assert envelope["format_version"] == FORMAT_VERSION == 1
    assert isinstance(envelope["payload"], dict)


def test_full_flag_controls_payload(result: BacktestResult) -> None:
    compact = msgpack.unpackb(result.to_msgpack(full=False), raw=False)["payload"]
    full = msgpack.unpackb(result.to_msgpack(full=True), raw=False)["payload"]
    assert "drawdowns" not in compact
    assert "drawdowns" in full


def test_top_level_helpers_exposed(result: BacktestResult) -> None:
    assert wbt.to_msgpack is to_msgpack
    data = wbt.to_msgpack(result)
    assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# 错误封装被拒绝
# ---------------------------------------------------------------------------
def test_wrong_format_rejected(tmp_path) -> None:
    path = tmp_path / "bad_format.msgpack"
    path.write_bytes(msgpack.packb({"format": "nope", "format_version": 1, "payload": {}}, use_bin_type=True))
    with pytest.raises(ValueError, match="unexpected format"):
        load_msgpack(path)


def test_unknown_version_rejected(tmp_path) -> None:
    path = tmp_path / "bad_version.msgpack"
    path.write_bytes(msgpack.packb({"format": FORMAT, "format_version": 999, "payload": {}}, use_bin_type=True))
    with pytest.raises(ValueError, match="unsupported format_version"):
        load_msgpack(path)


def test_non_mapping_envelope_rejected(tmp_path) -> None:
    path = tmp_path / "bad_envelope.msgpack"
    path.write_bytes(msgpack.packb([1, 2, 3], use_bin_type=True))
    with pytest.raises(ValueError):
        load_msgpack(path)
