"""打印策略数据的详细信息。"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def log_strategy_info(strategy: str, df: pd.DataFrame) -> None:
    """打印策略数据详情，包括每个品种的数据详情。"""
    logger.info("-" * 100)
    if df.empty:
        logger.warning(f"策略 {strategy} 数据为空！")
        return

    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)

    logger.info(f"策略 {strategy} 数据详情：")
    logger.info(f"  总记录数: {len(df)}")
    logger.info(f"  时间范围: {df['dt'].min()} ~ {df['dt'].max()}; 时间点数: {df['dt'].nunique()}")
    logger.info(f"  品种数量: {df['symbol'].nunique()}")
    logger.info("  品种详情:")
    for symbol in sorted(df["symbol"].unique()):
        sub = df[df["symbol"] == symbol]
        if "weight" in sub.columns:
            ws = sub["weight"].describe()
            logger.info(
                f"    {symbol}: 记录数={len(sub)}, 时间={sub['dt'].min()}~{sub['dt'].max()}, "
                f"权重范围=[{ws['min']:.3f}, {ws['max']:.3f}], 平均={ws['mean']:.3f}"
            )
        else:
            logger.info(f"    {symbol}: 记录数={len(sub)}, 时间={sub['dt'].min()}~{sub['dt'].max()}")

    if "weight" in df.columns:
        null_w = int(df["weight"].isnull().sum())
        zero_w = int((df["weight"] == 0).sum())
        if null_w or zero_w:
            logger.warning(f"  数据质量提醒: 空权重={null_w}, 零权重={zero_w}")
    logger.info("-" * 100)
