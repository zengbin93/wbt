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
from typing import TYPE_CHECKING, Any

from wbt._wbt import load_msgpack as _load_msgpack
from wbt._wbt import to_msgpack as _to_msgpack

if TYPE_CHECKING:
    from wbt.result import BacktestResult

FORMAT = "wbt.backtest_result"
FORMAT_VERSION = 1


def to_msgpack(result: BacktestResult, *, full: bool = True) -> bytes:
    """把 ``BacktestResult`` 编码为 MessagePack 字节。"""
    return _to_msgpack(result.to_dict(full=full))


def dump_msgpack(result: BacktestResult, path: str | Path, *, full: bool = True) -> None:
    """把 ``BacktestResult`` 写为 ``.msgpack`` 文件。"""
    Path(path).write_bytes(to_msgpack(result, full=full))


def load_msgpack(path: str | Path) -> dict[str, Any]:
    """读取 ``.msgpack`` 文件，校验封装头后返回 ``dict`` payload。

    ``format`` 不匹配或 ``format_version`` 未知时抛 ``ValueError``。
    """
    return _load_msgpack(str(path))
