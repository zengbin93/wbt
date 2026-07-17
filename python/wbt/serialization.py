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
    envelope = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "payload": result.to_dict(full=full),
    }
    return cast(bytes, msgpack.packb(envelope, use_bin_type=True))


def dump_msgpack(result: BacktestResult, path: str | Path, *, full: bool = True) -> None:
    """把 ``BacktestResult`` 写为 ``.msgpack`` 文件。"""
    Path(path).write_bytes(to_msgpack(result, full=full))


def load_msgpack(path: str | Path) -> dict[str, Any]:
    """读取 ``.msgpack`` 文件，校验封装头后返回 ``dict`` payload。

    ``format`` 不匹配或 ``format_version`` 未知时抛 ``ValueError``。
    """
    msgpack = _require_msgpack()
    envelope = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    return _unwrap(envelope)


def _unwrap(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValueError(f"invalid msgpack envelope: expected mapping, got {type(envelope).__name__}")
    fmt = envelope.get("format")
    if fmt != FORMAT:
        raise ValueError(f"unexpected format {fmt!r}, expected {FORMAT!r}")
    version = envelope.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version {version!r}, expected {FORMAT_VERSION}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid msgpack envelope: missing or malformed payload")
    return payload
