"""BacktestResult 的 MessagePack 交换格式读写。

定位：**完整嵌套结果对象**（``BacktestResult.to_dict``）在 Python/Rust 之间的二进制
交换格式。不替代 Arrow IPC / Parquet 处理收益曲线、rolling、drawdowns、key_trades 等
列式表格热数据。

封装格式（envelope）::

    {
        "format": "wbt.backtest_result",
        "format_version": 1,
        "payload": result.to_dict(full=full),
    }

第一版 ``load_msgpack`` 返回 ``dict`` payload，不反构造成 ``BacktestResult``：后者的懒加载
字段依赖私有源对象 ``_wb``，落盘结果无从恢复，强行还原只会得到半残对象。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from wbt.result import BacktestResult

FORMAT = "wbt.backtest_result"
FORMAT_VERSION = 1


def _require_msgpack():
    try:
        import msgpack
    except ImportError as e:  # pragma: no cover - 依赖缺失路径
        raise ImportError(
            "msgpack is required for BacktestResult MessagePack I/O; install with `pip install wbt[msgpack]`"
        ) from e
    return msgpack


def to_msgpack(result: BacktestResult, *, full: bool = True) -> bytes:
    """把 ``BacktestResult`` 编码为 MessagePack 字节。"""
    msgpack = _require_msgpack()
    return cast(bytes, msgpack.packb(_envelope(result, full=full), use_bin_type=True))


def to_json(result: BacktestResult, *, full: bool = True) -> bytes:
    """把 ``BacktestResult`` 编码为 UTF-8 JSON 字节。"""
    return json.dumps(_envelope(result, full=full), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def dump_msgpack(result: BacktestResult, path: str | Path, *, full: bool = True) -> None:
    """把 ``BacktestResult`` 写为 ``.msgpack`` 文件。"""
    Path(path).write_bytes(to_msgpack(result, full=full))


def dump_json(result: BacktestResult, path: str | Path, *, full: bool = True) -> None:
    """把结果写为带版本 envelope 的 ``.json`` 文件。"""
    Path(path).write_bytes(to_json(result, full=full))


def load_msgpack(path: str | Path) -> dict[str, Any]:
    """读取 ``.msgpack`` 文件，校验封装头后返回 ``dict`` payload。

    ``format`` 不匹配或 ``format_version`` 未知时抛 ``ValueError``。
    """
    msgpack = _require_msgpack()
    envelope = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    return _unwrap(envelope)


def load_json(path: str | Path) -> dict[str, Any]:
    """读取并校验带版本 envelope 的 JSON 文件。"""
    return _unwrap(json.loads(Path(path).read_bytes()))


def assert_payload_equal(left: Any, right: Any, path: str = "payload") -> None:
    """递归比较 wire payload，值和 Python 类型都必须相同。"""
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}: {type(left).__name__} != {type(right).__name__}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}: {left.keys()!r} != {right.keys()!r}")
        for key in left:
            assert_payload_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}: {len(left)} != {len(right)}")
        for index, value in enumerate(left):
            assert_payload_equal(value, right[index], f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def _envelope(result: BacktestResult, *, full: bool) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "payload": _normalize_payload(result.to_dict(full=full)),
    }


def _normalize_payload(payload: Any) -> dict[str, Any]:
    from wbt.result import _json_safe

    normalized = _json_safe(payload)
    if not isinstance(normalized, dict):
        raise TypeError("BacktestResult payload must be a mapping")
    _validate_json_value(normalized)
    return normalized


def _validate_json_value(value: Any, path: str = "payload") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"non-finite float at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"object key at {path} must be a string, got {type(key).__name__}")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _unwrap(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValueError(f"invalid msgpack envelope: expected mapping, got {type(envelope).__name__}")
    fmt = envelope.get("format")
    if type(fmt) is not str or fmt != FORMAT:
        raise ValueError(f"unexpected format {fmt!r}, expected {FORMAT!r}")
    version = envelope.get("format_version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version {version!r}, expected {FORMAT_VERSION}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid msgpack envelope: missing or malformed payload")
    _validate_json_value(payload)
    return payload
