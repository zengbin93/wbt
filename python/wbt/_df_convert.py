import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import Union

def pandas_to_arrow_bytes(df: Union[pd.DataFrame, pd.Series]) -> bytes:
    """
    将 Pandas DataFrame 转换为 Arrow 字节流

    参数:
        df (pd.DataFrame): 输入的 Pandas DataFrame

    返回:
        bytes: 序列化后的 Arrow 字节流
    """
    # 将 Pandas DataFrame 转换为 PyArrow Table
    table = pa.Table.from_pandas(df)

    # 序列化为 Arrow IPC 文件格式的字节流
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

    # 返回字节流
    return sink.getvalue().to_pybytes()


def arrow_bytes_to_pd_df(arrow_bytes: bytes) -> pd.DataFrame:
    """
    将 Arrow 字节流转换回 Pandas DataFrame

    参数:
        arrow_bytes (bytes): 输入的 Arrow 字节流

    返回:
        pd.DataFrame: 还原的 Pandas DataFrame
    """
    # 使用 Arrow BufferReader 读取字节流
    buffer = pa.BufferReader(arrow_bytes)

    # 通过 IPC 文件格式读取 Arrow 表
    with ipc.open_file(buffer) as reader:
        table = reader.read_all()

    # 转换 Arrow Table 为 Pandas DataFrame
    return table.to_pandas()
